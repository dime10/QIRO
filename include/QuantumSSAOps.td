#ifndef QUANTUM_SSA_OPS
#define QUANTUM_SSA_OPS

include "QuantumSSADialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

// Base class for "gates", i.e. operations executable on a quantum computer
class QuantumSSA_Gate<string mnemonic, list<OpTrait> traits = []> : QuantumSSA_Op<mnemonic, traits> {
    let description = [{
        A "gate" op is one that supports the instation of itself without requiring
        immediate application to specific qubits. That means that the target qubits
        need not necessarily be supplied (signifying the op is on hold) and need to
        be tagged 'optional' in ODS, with a fixed name (used by the verifier): "qbs".
        Similarly, the output, which has as type any of the quantum op types, will
        only be generated for an op on hold, and thus needs to be tagged 'optional'
        as well, with the fixed name: "op".
        Note that the verifier will check that only one of the two is given/returned.
    }];

    string additionalVerifier = "";
    let verifier = additionalVerifier#[{
        if (this->qbs()) {
            if(this->qbs().getType() != this->res().getType())
                return this->emitOpError() << "requires same type for input and return state!";
        } else if (!this->res().getType().isa<U1Type>() &&
                   !this->res().getType().isa<U2Type>() &&
                   !this->res().getType().isa<COpType>() &&
                   !this->res().getType().isa<CircType>()) {
            return this->emitOpError() << "must return itself when no target given!";
        }

        return success();
    }];
}

def AllocOp : QuantumSSA_Op<"alloc", [NoSideEffect]> {
    let summary = "Allocate a qubit (state).";
    let description = [{
        The allocation command registers a new qubit with the system, which
        is guaranteed to be in the |0> state, and returns it as a state.

        This operation takes no inputs and returns a value of type 'qstate'.

        Example:

        ```mlir
        // Allocate single qubit
        %qb = "qs.alloc"() : () -> !qs.qstate
        // OR in custom assembly format
        %qb = qs.alloc -> !qs.qstate
        ```
    }];

    let results = (outs
        Qstate_Type : $qb
    );

    let assemblyFormat = [{
        attr-dict `->` type($qb)
    }];
}

def AllocRegOp : QuantumSSA_Op<"allocreg", [NoSideEffect]> {
    let summary = "Allocate a qubit register (state).";
    let description = [{
        This allocation command registers a new qubit register with the system,
        which is guaranteed to be in the |0..> state, and returns it as a state.

        This operation takes an integer attribute as input specifying the size,
        and returns a value of type 'rstate<size>'.

        Example:

        ```mlir
        // Allocate qubit register
        %reg = "qs.allocreg"() {static_size=4} : () -> !qs.rstate<4>
        // OR in custom assembly format
        %reg = qs.allocreg(4) -> !qs.rstate<4>
        ```
    }];

    let arguments = (ins
        Optional<Index> : $size,
        OptionalAttr<Confined<APIntAttr, [IntNonNegative]>> : $static_size
    );

    let results = (outs
        Rstate_Type : $reg
    );

    let verifier = [{
        if (this->size() && this->static_size())
            return this->emitOpError() << "cannot provide register size as attribute AND operand!";
        if (!this->size() && !this->static_size())
            return this->emitOpError() << "requires a register size!";

        // check return type matches type of argument
        RstateType regType = this->reg().getType().cast<RstateType>();
        if (this->size() && regType.getNumQubits())
            return this->emitOpError() << "can't return type with static param if size is dynamic!";
        if (this->static_size() && !regType.getNumQubits())
            return this->emitOpError() << "must return type with static param if size is static!";

        return success();
    }];

    let assemblyFormat = [{
        `(` custom<IntParam>($size, type($size), $static_size) `)` attr-dict `->` type(results)
    }];
}

def FreeOp : QuantumSSA_Op<"free"> {
    let summary = "Deallocate a qubit.";
    let description = [{
        The free command deallocates a qubit with the system. The qubit is not
        required to be in the |0> state, but it must be disentangled from all
        other qubits.

        This operation takes one operand of type 'qstate'.

        Example:

        ```mlir
        // Deallocate single qubit
        "qs.free"(%qb) : (!qs.qstate) -> ()
        // OR in custom assembly format
        qs.free %qb : !qs.qstate
        ```
    }];

    let arguments = (ins
        Qstate_Type
    );

    let assemblyFormat = [{
        operands attr-dict `:` type(operands)
    }];
}

def FreeRegOp : QuantumSSA_Op<"freereg"> {
    let summary = "Deallocate a qubit register.";
    let description = [{
        This free command deallocates a qubit register with the system. The register
        is not required to be in the |0..> state, but every qubit inside must be
        disentangled from all other qubits.

        This operation takes one operand of type 'rstate'.

        Example:

        ```mlir
        // Allocate qubit register
        "qs.freereg"(%reg) : (!qs.rstate<4>) -> ()
        // OR in custom assembly format
        qs.freereg %reg : !qs.rstate<4>
        ```
    }];

    let arguments = (ins
        Rstate_Type : $reg
    );

    let assemblyFormat = [{
        operands attr-dict `:` type(operands)
    }];
}

def ExtractOp : QuantumSSA_Op<"extract", [NoSideEffect]> {
    let summary = "Extract qubits from register.";
    let description = [{
        This op generates new ssa values for single qubits inside a register(-like).
        The remaining part of the register is always returned as a new valuen, even
        when empty. The variadic indices can be either all constant or all dynamic.
        To avoid aliasing, the input register is invalidated.

        Example:

        ```mlir
        %r = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>

        // Extract single qubit from register
        %q, %rem = "qs.extract"(%r) {const_idx=[2]} : (!qs.rstate<4>) -> (!qs.qstate, !qs.rstate<3>)
        // OR in custom assembly format
        %q, %rem = qs.extract %r[2] : !qs.rstate<4> -> !qs.qstate, !qs.rstate<3>

        // Extract qubit with dynamic index from Qlist
        func @foo(%l : !qs.rstate<>, %i : index) {
            %q, %rem = "qs.extract"(%l, %i) : (!qs.rstate<>, index) -> (!qs.qstate, !qs.rstate<>)
            // OR in custom assembly format
            %q, %rem = qs.extract %l{%i} : !qs.rstate<>, index -> !qs.qstate, !qs.rstate<>
            return
        }
    }];

    let arguments = (ins
        Rstate_Type : $reg,
        Variadic<Index> : $dyn_idx,
        OptionalAttr<Confined<NonNegI64ArrayAttr, [ArrayMinCount<1>]>> : $const_idx
    );

    let results = (outs
        Variadic<Qstate_Type> : $qbs,
        Rstate_Type : $rem
    );

    let verifier = [{
        if (this->const_idx() && this->dyn_idx().size())
            return this->emitOpError() << "takes indices as attributes or operands, but not both!";
        if (!this->const_idx() && !this->dyn_idx().size())
            return this->emitOpError() << "needs at least 1 index, either as attribute or operand!";

        // check num indices/qubits match
        unsigned numIdx = this->const_idx() ? (*this->const_idx()).size() : this->dyn_idx().size();
        if (numIdx != this->qbs().size())
            return this->emitOpError() << "with differing number of qubits vs indices, qubits: "
                                       << this->qbs().size() << ", indices: " << numIdx << "!";

        // return type checking
        if (auto numQubits = this->reg().getType().cast<RstateType>().getNumQubits()) {
            int regsize = *numQubits;

            // bounds checking (if possible)
            if(this->const_idx()) {
                for (auto idxAttr : *this->const_idx()) {
                    int64_t idx = idxAttr.dyn_cast<IntegerAttr>().getInt();
                    if (idx >= regsize)
                        return this->emitOpError() << "detected index value out of bounds, idx: "
                                                   << idx << ", reg size: " << regsize << "!";
                }
            }

            if (auto size = this->rem().getType().cast<RstateType>().getNumQubits())
                if (*size != regsize - numIdx)
                    return this->emitOpError() << "output reg size != (input reg size - # qubits)!";
        }

        // invalidate register value (no more uses beyond this point)
        Value::user_range users = this->reg().getUsers();
        if (dyn_cast_or_null<ExtractOp>(*users.begin()) != *this) {
            return this->emitError() << "The register operand is invalid beyond this op, "
                                        "but detected further use!";
        }

        return success();
    }];

    let assemblyFormat = [{
        $reg ($const_idx^)? (`{` $dyn_idx^ `}`)? attr-dict `:` type($reg) `->` type(results)
    }];
}

def CombineStatOp : QuantumSSA_Op<"scombine", [NoSideEffect]> {
    let summary = "Merge qubits into register (statically).";
    let description = [{
        This op generates a new ssa values for for a register(-like) state obtained
        from a previous register value and any number of additional qubits.
        The provided indices indicating insertion points for each of the qubits must
        be provided as an integer array attribute. To avoid aliasing, all quantum
        input arguments (qubits & register states) are invalidated.

        Example:

        ```mlir
        %r = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        %q = "qs.alloc"() : () -> !qs.qstate

        // Add single qubit to register
        %r2 = "qs.scombine"(%r, %q) {const_idx=[2]} : (!qs.rstate<4>, !qs.qstate) -> !qs.rstate<5>
        // OR in custom assembly format
        %r2 = qs.scombine %r[2], %q : !qs.rstate<4>, !qs.qstate -> !qs.rstate<5>
    }];

    let arguments = (ins
        Rstate_Type : $reg,
        Confined<NonNegI64ArrayAttr, [ArrayMinCount<1>]> : $const_idx,
        Variadic<Qstate_Type> : $qbs
    );

    let results = (outs
        Rstate_Type : $newreg
    );

    let verifier = [{
        // check num indices/qubits match
        unsigned numIdx = this->const_idx().size();
        if (numIdx != this->qbs().size())
            return this->emitOpError() << "with differing number of qubits vs indices, qubits: "
                                       << this->qbs().size() << ", indices: " << numIdx << "!";

        // return type checking
        if (auto numQubits = this->reg().getType().cast<RstateType>().getNumQubits()) {
            int regsize = *numQubits;
            if (auto size = this->newreg().getType().cast<RstateType>().getNumQubits())
                if (*size != regsize + numIdx)
                    return this->emitOpError() << "output reg size != (input reg size + # qubits)!";

            // bounds checking
            llvm::SmallVector<int64_t, 8> idxArray; idxArray.reserve(numIdx);
            for (auto idxAttr : this->const_idx())
                idxArray.push_back(idxAttr.dyn_cast<IntegerAttr>().getInt());
            std::sort(idxArray.begin(), idxArray.end());
            for (int64_t idx : idxArray) {
                if (idx > regsize)
                        return this->emitOpError() << "detected index value out of bounds, idx: "
                                                   << idx << ", reg size: " << regsize << "!";
                regsize++;
            }
        }

        // invalidate register and qubit values (no more uses beyond this point)
        Value::user_range users = this->reg().getUsers();
        if (dyn_cast_or_null<CombineStatOp>(*users.begin()) != *this) {
            return this->emitError() << "The register operand is invalid beyond this op, "
                                        "but detected further use!";
        }
        for (auto arg : this->qbs()) {
            Value::user_range users = arg.getUsers();
            if (dyn_cast_or_null<CombineStatOp>(*users.begin()) != *this) {
                std::string errMsg("Operand value ");
                llvm::raw_string_ostream stream(errMsg);
                AsmState state(this->getParentOp());
                arg.printAsOperand(stream, state);
                errMsg += " is invalid beyond this op, but detected further use!";
                return this->emitError() << errMsg;
            }
        }

        return success();
    }];

    let assemblyFormat = [{
        $reg $const_idx `,` $qbs attr-dict `:` type(operands) `->` type(results)
    }];
}

def CombineDynOp : QuantumSSA_Op<"dcombine", [NoSideEffect, SameVariadicOperandSize]> {
    let summary = "Merge qubits into register (dynamically).";
    let description = [{
        This op generates a new ssa values for for a register(-like) state obtained
        from a previous register value and any number of additional qubits.
        The provided indices indicating insertion points for each of the qubits must
        be provided as ssa operands. To avoid aliasing, all quantum input arguments
        (qubits & register states) are invalidated.

        Example:

        ```mlir
        // Add qubit to Qlist with dynamic index
        func @foo(%l : !qs.rstate<>, %i : index, %q : !qs.qstate) {
            %l2 = "qs.dcombine"(%l, %i, %q) : (!qs.rstate<>, index, !qs.qstate) -> !qs.rstate<>
            // OR in custom assembly format
            %l2 = qs.dcombine %l{%i}, %q : !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>
            return
        }
    }];

    let arguments = (ins
        Rstate_Type : $reg,
        Variadic<Index> : $dyn_idx,
        Variadic<Qstate_Type> : $qbs
    );

    let results = (outs
        Rstate_Type : $newreg
    );

    let verifier = [{
        if (!this->qbs().size())
            return this->emitOpError() << "needs at least 1 qubit value to insert!";

        // return type checking
        if (auto numQubits = this->reg().getType().cast<RstateType>().getNumQubits()) {
            int regsize = *numQubits;
            int numIdx = this->dyn_idx().size();
            if (auto size = this->newreg().getType().cast<RstateType>().getNumQubits())
                if (*size != regsize + numIdx)
                    return this->emitOpError() << "output reg size != (input reg size - # qubits)!";
        }

        // invalidate register and qubit values (no more uses beyond this point)
        Value::user_range users = this->reg().getUsers();
        if (dyn_cast_or_null<CombineDynOp>(*users.begin()) != *this) {
            return this->emitError() << "The register operand is invalid beyond this op, "
                                        "but detected further use!";
        }
        for (auto arg : this->qbs()) {
            Value::user_range users = arg.getUsers();
            if (dyn_cast_or_null<CombineDynOp>(*users.begin()) != *this) {
                std::string errMsg("Operand value ");
                llvm::raw_string_ostream stream(errMsg);
                AsmState state(this->getParentOp());
                arg.printAsOperand(stream, state);
                errMsg += " is invalid beyond this op, but detected further use!";
                return this->emitError() << errMsg;
            }
        }

        return success();
    }];

    let assemblyFormat = [{
        $reg `[` $dyn_idx `]` `,` $qbs attr-dict `:` type($reg) `,` type($qbs) `->` type(results)
    }];
}

def HOp : QuantumSSA_Gate<"H", [Unitary, Hermitian, NoSideEffect]> {
    let summary = "Hadamard gate.";
    let description = [{
        The Hadamard gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,  1],
             [1, -1]] / sqrt(2)

        This operation takes a qubit or register state as input and returns an
        updated state. Alternatively, if no target is provided, the operation
        is returned.

        Example:

        ```mlir
        %0 = "qs.alloc"() : () -> !qs.qstate
        %1 = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        // Apply a Hadamard gate to qubit %0
        "qs.H"(%0) : (!qs.qstate) -> !qs.qstate
        // Apply a Hadamard gate to each qubit in register %1
        "qs.H"(%1) : (!qs.rstate<4>) -> !qs.rstate<4>
        // OR in custom assembly format
        qs.H %1 : !qs.rstate<4> -> !qs.rstate<4>
        // Generate Hadamard op without application to qubits (to be manipulated via meta-ops)
        $op = "qs.H"() : () -> !qs.u1
        // OR in custom assembly format
        %op = qs.H -> !qs.u1
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        AnyTypeOf<[QData_Type, U1_Type]> : $res
    );

    let assemblyFormat = [{
        ($qbs^ `:` type($qbs))? `->` type(results) attr-dict
    }];
}

def XOp : QuantumSSA_Gate<"X", [Unitary, Hermitian, NoSideEffect]> {
    let summary = "NOT gate.";
    let description = [{
        The NOT gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[0, 1],
             [1, 0]]

        This operation takes a qubit or register state as input and returns an
        updated state. Alternatively, if no target is provided, the operation
        is returned.

        Example:

        ```mlir
        %0 = "qs.alloc"() : () -> !qs.qstate
        %1 = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        // Apply a NOT gate to qubit %0
        "qs.X"(%0) : (!qs.qstate) -> !qs.qstate
        // Apply a NOT gate to each qubit in register %1
        "qs.X"(%1) : (!qs.rstate<4>) -> !qs.rstate<4>
        // OR in custom assembly format
        qs.X %1 : !qs.rstate<4> -> !qs.rstate<4>
        // Generate NOT op without application to qubits (to be manipulated via meta-ops)
        $op = "qs.X"() : () -> !qs.u1
        // OR in custom assembly format
        %op = qs.X -> !qs.u1
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        AnyTypeOf<[QData_Type, U1_Type]> : $res
    );

    let assemblyFormat = [{
        ($qbs^ `:` type($qbs))? `->` type(results) attr-dict
    }];
}

def RzOp : QuantumSSA_Gate<"RZ", [Unitary, NoSideEffect]> {
    let summary = "Z-Rotation gate.";
    let description = [{
        The Rz gate is a single Qubit unitary.
        It rotates a qubit state by an angle φ along the z-axis.
        Its 2x2 matrix representation in the computational basis is given by:
            [[e^-iφ/2,    0  ],
             [   0   , e^iφ/2]]

        This operation takes a qubit or register state as input and returns an
        updated state. The parameter φ is mandatory and can be provided as an
        attribute or operand. Alternatively, if no target is provided, the
        operation is returned.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !qs.qstate
        %1 = "q.allocreg"() {size=4} : () -> !qs.rstate<4>
        %phi = constant 0.1 : f64
        // Apply a z-rotation by angle 0.1 to qubit %0
        "qs.RZ"(%phi, %0) : (f64, !qs.qstate) -> !qs.qstate
        // Apply a z-rotation by angle 0.1 to each qubit in register %1
        "qs.RZ"(%phi, %1) : (f64, !qs.rstate<4>) -> !qs.rstate<4>
        // OR in custom assembly format
        qs.RZ(%phi) %1 : f64, !qs.rstate<4> -> !qs.rstate<4>
        // Generate RZ gate without application to qubits (to be manipulated via meta-ops)
        $op = "qs.RZ"(%phi) : (f64) -> !qs.u1
        // OR in custom assembly format
        %op = qs.RZ(%phi) : f64 -> !qs.u1
        ```
    }];

    let arguments = (ins
        AnyFloat : $phi,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        AnyTypeOf<[QData_Type, U1_Type]> : $res
    );

    let assemblyFormat = [{
        `(` $phi `)` ($qbs^)? `:` type(operands) `->` type(results) attr-dict
    }];
}

def ROp : QuantumSSA_Gate<"R", [Unitary, NoSideEffect]> {
    let summary = "Rotation gate.";
    let description = [{
        The R gate is a single Qubit unitary.
        It rotates a qubit state by an angle φ about the |1> state.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,   0 ],
             [0, e^iφ]]

        This operation takes a qubit or register state as input and returns an
        updated state. The parameter φ is mandatory and must be provided as an
        operand. Alternatively, if no target is provided, the operation is returned.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !qs.qstate
        %1 = "q.allocreg"() {size=4} : () -> !qs.rstate<4>
        %phi = constant 0.1 : f64
        // Apply a rotation by angle 0.1 to qubit %0
        "qs.R"(%phi, %0) : (f64, !qs.qstate) -> !qs.qstate
        // Apply a z-rotation by angle 0.1 to each qubit in register %1
        "qs.R"(%phi, %1) : (f64, !qs.rstate<4>) -> !qs.rstate<4>
        // OR in custom assembly format
        qs.R(%phi) %1 : f64, !qs.rstate<4> -> !qs.rstate<4>
        // Generate R gate without application to qubits (to be manipulated via meta-ops)
        $op = "qs.R"(%phi) : (f64) -> !qs.u1
        // OR in custom assembly format
        %op = qs.R(%phi) : f64 -> !qs.u1
        ```
    }];

    let arguments = (ins
        AnyFloat : $phi,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        AnyTypeOf<[QData_Type, U1_Type]> : $res
    );

    let assemblyFormat = [{
        `(` $phi `)` ($qbs^)? `:` type(operands) `->` type(results) attr-dict
    }];
}

def CNotOp : QuantumSSA_Gate<"CX", [Unitary, Hermitian, NoSideEffect, AttrSizedOperandSegments]> {
    let summary = "CNOT gate.";
    let description = [{
        The CNOT gate is a two Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 4x4 matrix representation in the computational basis is given by:
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]

        As with other basic operations, the target is allowed to be either a qubit or
        a register state (in which case the gate is applied to every qubit in the register).
        The control however must be a singular qubit state. Alternatively, if no target is
        provided, the operation is returned. Note that the control qubit MUST NOT appear as
        one of target qubits (not checked atm).

        Example:

        ```mlir
        %0 = "qs.alloc"() : () -> !qs.qstate
        %1 = "qs.alloc"() : () -> !qs.qstate
        %2 = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        // Apply a CNOT to target %1 and control %0
        "qs.CX"(%0, %1) : (!qs.qstate, !qs.qstate) -> (!qs.qstate, !qs.qstate)
        // Apply a CNOT with control %0 to each qubit in register %2
        "qs.CX"(%0, %2) : (!qs.qstate, !qs.rstate<4>) -> (!qs.qstate, !qs.rstate<4>)
        // OR in custom assembly format
        qs.CX %0, %2 : !qs.qstate, !qs.rstate<4> -> !qs.qstate, !qs.rstate<4>
        // Generate CNOT gate without application to qubits (to be manipulated via meta-ops)
        $op = "qs.CX"() : () -> !qs.u2
        // OR in custom assembly format
        $op = qs.CX -> !qs.u2
        ```
    }];

    let arguments = (ins
        Optional<Qstate_Type> : $ctrl,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        Optional<Qstate_Type> : $new_ctrl,
        AnyTypeOf<[QData_Type, U2_Type]> : $res
    );

    let additionalVerifier = [{
        auto res = this->res().getType();

        if (this->ctrl() && !this->qbs() || !this->ctrl() && this->qbs())
            return this->emitOpError() << "accepts either both control and target or neither!";

        if (!res.isa<U2Type>() && !this->new_ctrl() || res.isa<U2Type>() && this->new_ctrl())
            return this->emitOpError() << "must return either both qb/reg states or neither!";

        if (!res.isa<U2Type>() && this->qbs().getType() != res)
            return this->emitOpError() << "2nd input qb/reg type must match 2nd output type!";
    }];

    let assemblyFormat = [{
        ($ctrl^ `,` $qbs `:` type($ctrl) `,` type($qbs))? `->` type(results) attr-dict
    }];
}

def SwapOp : QuantumSSA_Gate<"SWAP", [Unitary, Hermitian, Commutative, AttrSizedOperandSegments]> {
    let summary = "SWAP gate.";
    let description = [{
        The SWAP gate is a two Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 4x4 matrix representation in the computational basis is given by:
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]

        This operation takes two single qubit states as input and returns
        their updated states. Alternatively, if no targets are the provided,
        the operation is returned with type 'u2'.

        Example:

        ```mlir
        %0 = "qs.alloc"() : () -> !qs.qstate
        %1 = "qs.alloc"() : () -> !qs.qstate
        // Apply a SWAP to qubits %0 and %1
        "q.SWAP"(%0, %1) : (!q.qstate, !q.qstate) -> (!q.qstate, !q.qstate)
        // OR in custom assembly format
        q.SWAP %0, %1 : !q.qstate, !q.qstate -> !q.qstate, !q.qstate
        ```
    }];

    let arguments = (ins
        Optional<Qstate_Type> : $qbs2,
        Optional<Qstate_Type> : $qbs
    );

    let results = (outs
        Optional<Qstate_Type> : $new_qbs2,
        AnyTypeOf<[Qstate_Type, U2_Type]> : $res
    );

    let additionalVerifier = [{
        auto res = this->res().getType();

        if (this->qbs2() && !this->qbs() || !this->qbs2() && this->qbs())
            return this->emitOpError() << "accepts either both qubit operands or neither!";

        if (!res.isa<U2Type>() && !this->new_qbs2() || res.isa<U2Type>() && this->new_qbs2())
            return this->emitOpError() << "must return either both qubit states or neither!";
    }];

    let assemblyFormat = [{
        ($qbs2^ `,` $qbs `:` type($qbs2) `,` type($qbs))? `->` type(results) attr-dict
    }];
}

def ReturnStateOp : QuantumSSA_Op<"return", [Terminator]> {
    let summary = "Return qubit states from a circuit.";
    let description = [{
        This is a terminator opperation for quantum circuits, which, compared to the
        'quantum' dialect version, is able to return updated qubit states from a circuit
        call. This is required as all SSA values inside the circuit live in their own
        scope.
    }];

    let arguments = (outs
        Variadic<QData_Type> : $retvals
    );

    let assemblyFormat = [{
       operands attr-dict `:` type(operands)
    }];
}

def CircuitOp : QuantumSSA_Op<"circ", [Symbol, FunctionLike, IsolatedFromAbove, AffineScope,
                                       DeclareOpInterfaceMethods<CallableOpInterface>]> {
    let summary = "Create a function-like quantum circuit.";
    let description = [{
        A quantum circuit defines a custom operation on the quantum computer by
        grouping multiple elementary gates or other circuit calls into one unit.
        The main use of this is to reuse a gate sequence in multiple places in the
        program, as well as apply the operation to different and differently-sized
        qubit sets (if the circuit is defined parametrically).

        The body of the circuit is contained within a region of a single block, so
        no traditional MLIR control flow via block branching can be used. However,
        the circuit op does define an affine scope for its body, which allows for
        control flow via the affine dialect, such as affine.for and affine.if.

        Example:

        ```mlir
        "qs.circ"() ({
        ^bb0(%r0: !qs.rstate<>):
            %r1 = "qs.H"(%r0) : (!qs.rstate<>) -> !qs.rstate<>
            "qs.return"(%r1) : (!qs.rstate<>) -> ()
        }) {sym_name="h", type=(!q.rstate<>) -> !qs.rstate<>} : () -> ()
        // OR in custom assembly
        qs.circ @h(%r0: !qs.rstate<>) -> !qs.rstate<> {
            %r1 = "qs.H"(%r0) : (!qs.rstate<>) -> !qs.rstate<>
            "qs.return"(%r1) : (!qs.rstate<>) -> ()
        }
        ```
    }];

    let regions = (region
        AnyRegion : $gates
    );

    let extraClassDeclaration = [{
        // FunctionLike trait needs access to the functions below.
        friend class OpTrait::FunctionLike<CircuitOp>;

        FunctionType getType() {
            return OpTrait::FunctionLike<CircuitOp>::getType();
        }

        // Required hooks for the input/output type enumeration in FunctionLike.
        unsigned getNumFuncArguments() {
            return this->getType().getNumInputs();
        }
        unsigned getNumFuncResults() {
            return this->getType().getNumResults();
        }

        // Verifies the type of the function for op specific invariants. Optional.
        LogicalResult verifyType();

        // Verifies the body of the function for op specific invariants. Optional.
        LogicalResult verifyBody();
    }];

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def CircuitValueOp : QuantumSSA_Op<"getval", [NoSideEffect]> {
    let summary = "Generate an SSA value for a circuit.";
    let description = [{
        Internal operation to generate SSA values of circuits given their
        symbol name. The values are required to perform meta operations on
        the circuits.
    }];

    let arguments = (ins
        FlatSymbolRefAttr : $circref
    );

    let results = (outs
        Circuit_Type : $circval
    );

    let verifier = [{
        Operation *circOp = this->resolveCallable();
        if (!circOp)
            return this->emitOpError() << "could not resolve given symbol!";

        if (!isa<CircuitOp>(circOp))
            return this->emitOpError() << "symbol does not reference a 'CircuitOp'!";

        return success();
    }];

    let extraClassDeclaration = [{
        Operation *resolveCallable() {
            return SymbolTable::lookupNearestSymbolFrom(this->getOperation(), this->circref());
        }
    }];

    let assemblyFormat = [{
        $circref attr-dict `->` type($circval)
    }];
}

def CallCircOp : QuantumSSA_Op<"call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "Perform direct call to quantum circuit with given qubit arguments.";
    let description = [{
        This operation is used to perform a call to a function-like quantum
        circuit, which is referenced via a symbol attribute.

        The operands, which can be of any type, are expected to match the
        arguments defined in the circuit signature. It returns the updated
        state for all qubits and registers given as input.

        Example:

        ```mlir
        qs.circ @h(%r0: !qs.rstate<>) -> !qs.rstate<> {
            %r1 = "qs.H"(%r0) : (!qs.rstate<>) -> !qs.rstate<>
            "qs.return"(%r1) : (!qs.rstate<>) -> ()
        }

        %0 = "qs.allocreg"() {n=4} : () -> !qs.rstate<4>

        // Call circuit with qubit register %0
        %1 = "qs.call"(%0) {circref=@h} : (!qs.rstate<4>) -> !qs.rstate<4>
        // OR in custom assembly
        %1 = qs.call @h(%0) : !qs.rstate<4> -> !qs.rstate<4>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr : $circref,
        Variadic<AnyType> : $args
    );

    let results = (outs
        Variadic<QData_Type> : $results
    );

    let verifier = [{
        // check circuit symbol
        Operation *callable = cast<CallOpInterface>(this->getOperation()).resolveCallable();
        if (!callable)
            return this->emitOpError() << "could not resolve given symbol!";

        if (!isa<CircuitOp>(callable))
            return this->emitOpError() << "symbol does not reference a 'CircuitOp'!";

        // make sure function arguments match
        CircuitOp circOp = cast<CircuitOp>(callable);
        FunctionType circType = circOp.OpTrait::FunctionLike<CircuitOp>::getType();

        // compare number of arguments
        if (circType.getNumInputs() != this->args().size())
            return this->emitOpError() << "has wrong number of circuit arguments! "
                                          "Circuit requires: " << circType.getNumInputs() << ". "
                                          "Got: " << this->args().size() << ".";
        // compare number of results
        if (circType.getNumResults() != this->results().size())
            return this->emitOpError() << "has wrong number of circuit results! "
                                          "Circuit requires: " << circType.getNumResults() << ". "
                                          "Got: " << this->results().size() << ".";

        // compare each parameter type:
        //  - in general all types must match the circuit signature exactly, with 1 exception
        //  - registers can appear in 2 forms in the circuit signature:
        //      as !qs.rstate<n> or !qs.rstate<>, both of which can be matched with either, however
        //      only in the case where both have size information will the given register be checked
        //      to be at least as large as required, from which the first n qubits will be used
        std::string errMsg = "detected wrong argument type! Expected: ";
        auto sigType = circType.getInputs().begin();
        for (auto argType : this->args().getTypes()) {
            if (auto sigRegType = sigType->dyn_cast<RstateType>()) {
                if (auto argRegType = argType.dyn_cast<RstateType>())
                    if (argRegType.getNumQubits() && sigRegType.getNumQubits())
                        if (*argRegType.getNumQubits() < *sigRegType.getNumQubits())
                            return this->emitOpError() << "too few qubits in register! Expected: "
                                                       << *sigRegType.getNumQubits() << ". Got: "
                                                       << *argRegType.getNumQubits() << ".";
                else
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
            } else {
                if (*sigType != argType)
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
            }
            sigType++;
        }

        // compare each return type to input types
        auto retType = this->results().getTypes().begin();
        for (auto argType : this->args().getTypes()) {
            if (argType.isa<QstateType>() || argType.isa<RstateType>()) {
                if (retType == this->results().getTypes().end())
                    this->emitOpError() << "has too few return values! Every QData argument "
                                           "needs to be returned in its updated state.";
                if (*retType != argType)
                    this->emitOpError() << "has mismatched return type! Requires: " << argType
                                        << ". Got: " << *retType << ".";
                retType++;
            }
        }

        return success();
    }];

    let assemblyFormat = [{
        $circref `(` $args `)` attr-dict `:` type($args) `->` type($results)
    }];
}

def ApplyCircOp : QuantumSSA_Op<"apply"> {
    let summary = "Apply quantum circuit to given arguments.";
    let description = [{
        In contrast to the call operation which uses a symbol to reference the
        circuit, this operation calls a circuit via the provided SSA operand.
        Typically, such a value is created to be manipulated via quantum meta
        operations, and is ultimately applied to qubits via this op.

        The circuit can be any value of type '!qs.circ' or a type derived from
        it, such as '!qs.cop<n, !qs.circ>.

        The remaining operands, which can be of any type, will be forwarded
        to the circuit, and must match its declared arguments. It returns the
        updated state for all qubits and registers given as input.

        Example:

        ```mlir
        qs.circ @h(%r0: !qs.rstate<>) -> !qs.rstate<> {
            %r1 = "qs.H"(%r0) : (!qs.rstate<>) -> !qs.rstate<>
            "qs.return"(%r1) : (!qs.rstate<>) -> ()
        }

        %0 = "qs.allocreg"() {n=4} : () -> !qs.rstate<4>
        %c = qs.getval @h -> !qs.circ<(!qs.rstate<>) -> !qs.rstate<>>

        // Apply circuit with qubit register %0
        %1 = "qs.apply"(%c, %0) : (!qs.rstate<4>) -> !qs.rstate<4>
        // OR in custom assembly
        %1 = qs.apply @h(%0) : !qs.rstate<4> -> !qs.rstate<4>
        ```
    }];

    let arguments = (ins
        QCirc_Type : $circval,
        Variadic<AnyType> : $args
    );

    let results = (outs
        Variadic<QData_Type> : $results
    );

    let verifier = [{
        if (auto copType = this->circval().getType().dyn_cast<COpType>())
            if (!copType.getBaseType() || !copType.getBaseType().isa<CircType>())
                return this->emitOpError() << "can only be applied to circuit(-derived) types!";

        // make sure function arguments match
        Operation *circValOp = this->circval().getDefiningOp();
        while (!isa<CircuitValueOp>(circValOp)) {
            if (auto cop = dyn_cast<ControlOp>(circValOp))
                circValOp = cop.heldOp().getDefiningOp();
            else
                circValOp = cast<AdjointOp>(circValOp).heldOp().getDefiningOp();
        }
        Operation *circOp = cast<CircuitValueOp>(circValOp).resolveCallable();
        FunctionType circType = cast<CircuitOp>(circOp).OpTrait::FunctionLike<CircuitOp>::getType();

        auto funargTypes = circType.getInputs();
        auto funresTypes = circType.getResults();

        MLIRContext *ctx = getContext();
        if (filterRegSize(funargTypes, ctx) != filterRegSize(this->args().getType(), ctx))
            return emitOpError("failed to verify that circuit input types match argument types!");
        if (filterRegSize(funresTypes, ctx) != filterRegSize(this->results().getType(), ctx))
            return emitOpError("failed to verify that circuit return types match result types!");

        return success();
    }];

    let extraClassDeclaration = [{
        static TypeRange filterRegSize(TypeRange types, MLIRContext *ctx) {
            Type rstate = RstateType::get(ctx, llvm::None);
            llvm::SmallVector<Type, 8> filteredTypes;
            filteredTypes.reserve(types.size());
            for (auto type : types) {
                if (type.isa<RstateType>())
                    filteredTypes.push_back(rstate);
                else
                    filteredTypes.push_back(type);
            }
            return filteredTypes;
        }
    }];

    let assemblyFormat = [{
        $circval `(` $args `)` attr-dict `:` type($circval) `(` type($args) `->` type($results) `)`
    }];
}

def ControlOp : QuantumSSA_Gate<"ctrl", [Meta, NoSideEffect, AttrSizedOperandSegments]> {
    let summary = "Control meta-op.";
    let description = [{
        This meta operation transforms a quantum op that has not been applied
        yet into a controlled operation. The target is optional again so that
        the op can further be transformed with meta-ops.
        A controlled operation only executes its base op if the state of the
        control qubit (or register) is |1..>.

        Accepted inputs are: individual quantum operations, quantum circuits,
        or already controlled ops. The result is a controlled op (if no target
        was given).

        Example:

        ```mlir
        %0 = "qs.alloc"() : () -> !qs.state
        %1 = "qs.alloc"() : () -> !qs.state
        %2 = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        %op = "qs.H"() : () -> !qs.u1
        // Control H op on qubit %0 and apply to qubit %1
        "qs.ctrl"(%op, %0, %1) : (!qs.u1, !qs.state, !qs.state) -> (!qs.state, !qs.state)
        // Control H op on qubit %0 and apply to all qubits in register %2
        "qs.ctrl"(%op, %0, %2) : (!qs.u1, !qs.state, !qs.rstate<4>) -> (!qs.state, !qs.rstate<4>)
        // Control H op on all of register %2 and apply to qubit %1
        "qs.ctrl"(%op, %2, %1) : (!qs.u1, !qs.rstate<4>, !qs.qstate) -> (!qs.rstate<4>, !qs.qstate)

        // Control H op on qubit %1 but leave application open
        %cop = "qs.ctrl"(%op, %1) : (!qs.u1, !qs.state) -> !qs.cop<1, !qs.u1>
        // OR in custom assembly format
        %cop = qs.ctrl %op, %1 : !qs.u1, !qs.state -> !qs.cop<1, !qs.u1>
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        QData_Type : $ctrls,
        Optional<Qstate_Type> : $qbs2,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QData_Type : $new_ctrls,
        Optional<Qstate_Type> : $new_qbs2,
        AnyTypeOf<[QData_Type, ControlledOp_Type]> : $res
    );

    let additionalVerifier = [{
        auto heldOp = this->heldOp().getType();
        auto res = this->res().getType();

        if (this->qbs2() && !this->qbs())
            return this->emitOpError() << "does not accept 2nd qubit arg without 1st!";

        // check presence of operands depending on heldOp type (u1 vs u2)
        if (!res.isa<COpType>()) {
            if (heldOp.isa<U1Type>() && (this->qbs2() || this->new_qbs2()))
                return this->emitOpError() << "does not accept second qubit arg for U1 op!";
            if (heldOp.isa<U2Type>() && (!this->qbs2() || !this->new_qbs2()))
                return this->emitOpError() << "requires second qubit arg for U2 op!";
        } else {
            if (this->new_qbs2())
                return this->emitOpError() << "cannot return 2nd qubit state when returning op!";
        }

        // make sure user is not trying to apply a circuit op by specifying a target
        if (this->qbs() && (heldOp.isa<CircType>() || heldOp.isa<COpType>() &&
                            heldOp.cast<COpType>().getBaseType().isa<CircType>()))
            return this->emitOpError() << "does not accept target qubits when controlling circuit!";

        // obtain total number of control qubits from input
        unsigned totalctrls = 0;

        auto ctrls = this->ctrls().getType();
        if (ctrls.isa<QstateType>()) {
            totalctrls++;
        } else if (auto regType = ctrls.dyn_cast<RstateType>()) {
            if (auto numQubits = regType.getNumQubits())
                totalctrls += *numQubits;
            else
                totalctrls = -1; // disable # of ctrl qubits check when no size is provided
        }

        if (auto cop = heldOp.dyn_cast<COpType>()) {
            if (auto numCtrls = cop.getNumCtrls())
                totalctrls += *numCtrls;
            else
                totalctrls = -1; // disable # of ctrl qubits check if base op is 'cop' without #
        }

        // perform several checks on the return type, if it is a COpType
        if (auto output = this->res().getType().dyn_cast<COpType>()) {
            // assert correct number of control qubits
            if (totalctrls != -1) {
                if (!output.getNumCtrls())
                    return this->emitOpError() << "requires # ctrl qubits in output if available!";
                else if (*output.getNumCtrls() != totalctrls)
                    return this->emitOpError() << "mismatched # of ctrls in return type vs inputs!";
            } else {
                if (output.getNumCtrls())
                    return this->emitOpError() << "unexpected # ctrl qubits in ouput (not verif.)!";
            }
            // assert proper base type
            if (heldOp.isa<COpType>()) {
                if (heldOp.cast<COpType>().getBaseType() != output.getBaseType())
                    return this->emitOpError() << "must preserve base type when given 'COp'!";
            } else {
                if (heldOp != output.getBaseType())
                    return this->emitOpError() << "return base type must match given operation!";
            }
        }
    }];

    let assemblyFormat = [{
        $heldOp `,` $ctrls
        custom<OneOrTwoQbs>($qbs2, $qbs)
        attr-dict `:` type(operands) `->` type(results)
    }];
}

def AdjointOp : QuantumSSA_Gate<"adj", [Meta, NoSideEffect, AttrSizedOperandSegments]> {
    let summary = "(Hermitian) adjoint meta-op.";
    let description = [{
        This meta operation transforms a quantum op that has not been applied
        yet into a complex-conjugated, transposed version. The target is optional
        again so that the op can further be transformed with meta-ops.
        Since all quantum gates are unitary, the resulting op is equal to the
        inverse operation.

        Accepted inputs are: individual quantum operations, quantum circuits,
        or controlled ops. The result is the same type as the input (if no
        target was given).

        Example:

        ```mlir
        %0 = "qs.alloc"() : () -> !qs.qstate
        %1 = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        %op = "qs.H"() : () -> !qs.u1
        // Invert H op and apply to qubit %0
        "qs.adj"(%op, %0) : (!qs.u1, !qs.qstate) -> !qs.qstate
        // Invert H op apply to all qubits in register %1
        "qs.adj"(%op, %1) : (!qs.u1, !qs.rstate<4>) -> !qs.rstate<4>
        // Invert H op but leave application open
        %opdg = "qs.adj"(%op) : (!qs.u1) -> !qs.u1
        // OR in custom assembly format
        %opdg = qs.adj %op : !qs.u1 -> !qs.u1
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        Optional<Qstate_Type> : $qbs2,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        Optional<Qstate_Type> : $new_qbs2,
        AnyTypeOf<[QData_Type, QOp_Type]> : $res
    );

    string additionalVerifier = [{
        auto heldOp = this->heldOp().getType();
        auto res = this->res().getType();
        bool resIsOp = !(res.isa<QstateType>() || res.isa<RstateType>());

        if (this->qbs2() && !this->qbs())
            return this->emitOpError() << "does not accept 2nd qubit arg without 1st!";

        // check presence of operands depending on heldOp type (u1 vs u2)
        if (!resIsOp) {
            if (heldOp.isa<U1Type>() && (this->qbs2() || this->new_qbs2()))
                return this->emitOpError() << "does not accept second qubit arg for U1 op!";
            if (heldOp.isa<U2Type>() && (!this->qbs2() || !this->new_qbs2()))
                return this->emitOpError() << "requires second qubit arg for U2 op!";
        } else {
            if (this->new_qbs2())
                return this->emitOpError() << "cannot return 2nd qubit state when returning op!";
        }

        // make sure user is not trying to apply a circuit op by specifying a target
        if (this->qbs() && (heldOp.isa<CircType>() || heldOp.isa<COpType>() &&
                            heldOp.cast<COpType>().getBaseType().isa<CircType>()))
            return this->emitOpError() << "does not accept target qubits when inverting circuit!";

        // check ouput op type matches input op type
        if (resIsOp && heldOp != res)
            return this->emitOpError() << "given operation type must match output type!";
    }];

    let assemblyFormat = [{
        $heldOp custom<OneOrTwoQbs>($qbs2, $qbs) attr-dict `:` type(operands) `->` type(results)
    }];
}

def MeasurementOp : QuantumSSA_Op<"meas"> {
    let summary = "Measure qubits.";
    let description = [{
        This operation performs a measurement on individual or groups of qubits.
        The measurement results are returned as clasical boolean values.

        Example:
        ```mlir
        %0 = "qs.alloc"() : () -> !qs.qstate
        %1 = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        // measure single qubit
        %m = "qs.meas"(%0) : (!qs.qstate) -> (!qs.qstate, i1)
        // OR in custom assebly format
        %m = qs.meas %0 : !qs.qstate -> !qs.qstate, i1
        // measure register
        %mr = "qs.meas"(%1) : (!qs.rstate<4>) -> (!qs.rstate<4>, memref<4xi1>)
        // OR in custom assembly format
        %mr = qs.meas %1 : !qs.rstate<4> -> !qs.rstate<4>, memref<4xi1>
        ```
    }];

    let arguments = (ins
        QData_Type : $qbs
    );

    let results = (outs
        QData_Type : $qbs_out,
        AnyTypeOf<[I1, I1MemRef]> : $res
    );

    let verifier = [{
        Type qbsType = this->qbs().getType();
        Type resType = this->res().getType();

        if (qbsType != this->qbs_out().getType())
            return this->emitOpError() << "must have matching types for in/output qubit states!";

        if (auto mrefType = resType.dyn_cast<MemRefType>()) {
            if (!mrefType.hasRank() || mrefType.getRank() != 1)
                return this->emitOpError() << "only produces ranked memrefs of 1 dimension!";

            if (auto qregType = qbsType.dyn_cast<RstateType>()) {
                if (qregType.getNumQubits()) {
                    if (mrefType.isDynamicDim(0))
                        return this->emitOpError() << "if available must provide static dim size!";
                    else if (*qregType.getNumQubits() != mrefType.getDimSize(0))
                        return this->emitOpError() << "mismatch in rstate and memref size!";
                }
            } else {
                return this->emitOpError() << "must return 'i1' when measuring qubit!";
            }
        } else if (!qbsType.isa<QstateType>()) {
            return this->emitOpError() << "must return memref when measuring multiple qubits!";
        }

        return success();
    }];

    let assemblyFormat = [{
        operands attr-dict `:` type(operands) `->` type(results)
    }];
}

#endif // QUANTUM_SSA_OPS
