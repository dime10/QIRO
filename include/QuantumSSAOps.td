#ifndef QUANTUM_SSA_OPS
#define QUANTUM_SSA_OPS

include "QuantumSSADialect.td"

// Base class for "gates", i.e. operations executable on a quantum computer
class QuantumSSA_Gate<string mnemonic, list<OpTrait> traits = []> : QuantumSSA_Op<mnemonic, traits> {
    let description = [{
        A "gate" op is one that supports the instation of itself without requiring
        immediate application to specific qubits. That means that the target qubits
        need not necessarily be supplied (signifying the op is on hold) and need to
        be tagged 'optional' in ODS, with a fixed name (used by the verifier): "qbs".
        Similarly, the output of type !qs.op will only be generated for an op on hold,
        and thus needs to be tagged 'optional' as well, with the fixed name: "op".
        Note that the verifier will check that only one of the two is given/returned.
    }];

    string additionalVerifier = "";
    let verifier = additionalVerifier#[{
        if (this->qbs()) {
            if(this->qbs().getType() != this->res().getType())
                return this->emitOpError() << "requires same type for input and return state!";
        } else if (!this->res().getType().isa<OpType>() &&
                   !this->res().getType().isa<COpType>() &&
                   !this->res().getType().isa<FunCircType>()) {
            return this->emitOpError() << "must return itself when no target given!";
        }

        return success();
    }];
}

def AllocQbOp : QuantumSSA_Op<"alloc"> {
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

def AllocRegOp : QuantumSSA_Op<"allocreg"> {
    let summary = "Allocate a qubit register (state).";
    let description = [{
        This allocation command registers a new qubit register with the system,
        which is guaranteed to be in the |0..> state, and returns it as a state.

        This operation takes an integer attribute as input specifying the size,
        and returns a value of type 'rstate<size>'.

        Example:

        ```mlir
        // Allocate qubit register
        %reg = "qs.allocreg"() {size=4} : () -> !qs.rstate<4>
        // OR in custom assembly format
        %reg = qs.allocreg(4) -> !qs.rstate<4>
        ```
    }];

    let arguments = (ins
        Confined<APIntAttr, [IntMinValue<2>]> : $size
    );

    let results = (outs
        Rstate_Type : $reg
    );

    let assemblyFormat = [{
        `(`$size`)` attr-dict `->` type($reg)
    }];
}

def HOp : QuantumSSA_Gate<"H", [Unitary, Hermitian]> {
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
        "qs.H"(%0) : (!qs.qstate) -> ()
        // Apply a Hadamard gate to each qubit in register %1
        "qs.H"(%1) : (!qs.rstate<4>) -> ()
        // OR in custom assembly format
        qs.H %1 : !qs.rstate<4>
        // Generate Hadamard op without application to qubits (to be manipulated via meta-ops)
        $op = "qs.H"() : () -> !qs.op
        // OR in custom assembly format
        %op = qs.H -> !qs.op
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrOp_Type : $res
    );

    let assemblyFormat = [{
        ($qbs^ `:` type($qbs))? `->` type(results) attr-dict
    }];
}

def XOp : QuantumSSA_Gate<"X", [Unitary, Hermitian]> {
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
        "qs.X"(%0) : (!qs.qstate) -> ()
        // Apply a NOT gate to each qubit in register %1
        "qs.X"(%1) : (!qs.rstate<4>) -> ()
        // OR in custom assembly format
        qs.X %1 : !qs.rstate<4>
        // Generate NOT op without application to qubits (to be manipulated via meta-ops)
        $op = "qs.X"() : () -> !qs.op
        // OR in custom assembly format
        %op = qs.X -> !qs.op
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrOp_Type : $res
    );

    let assemblyFormat = [{
        ($qbs^ `:` type($qbs))? `->` type(results) attr-dict
    }];
}

def RzOp : QuantumSSA_Gate<"RZ", [Unitary]> {
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
        // Apply a z-rotation by angle 0.1 to qubit %0
        "qs.RZ"(%0) {phi=0.1}: (!qs.qstate) -> ()
        // Apply a z-rotation by angle 0.1 to each qubit in register %1
        "qs.RZ"(%1) {phi=0.1} : (!qs.rstate<4>) -> ()
        // OR in custom assembly format
        qs.RZ(0.1) %1 : !qs.rstate<4>
        // Generate RZ gate without application to qubits (to be manipulated via meta-ops)
        $op = "qs.RZ"() {phi=0.1} : () -> !qs.op
        // OR in custom assembly format
        %op = qs.RZ(0.1) -> !qs.op
        ```
    }];

    let arguments = (ins
        AnyFloat : $phi,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrOp_Type : $res
    );

    let assemblyFormat = [{
        `(` $phi `)` ($qbs^)? `:` type(operands) `->` type(results) attr-dict
    }];
}

def ROp : QuantumSSA_Gate<"R", [Unitary]> {
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
    }];

    let arguments = (ins
        AnyFloat : $phi,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrOp_Type : $res
    );

    let assemblyFormat = [{
        `(` $phi `)` ($qbs^)? `:` type(operands) `->` type(results) attr-dict
    }];
}

def CNotOp : QuantumSSA_Gate<"CX", [Unitary, Hermitian]> {
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
        "qs.CX"(%0, %1) : (!qs.qstate, !qs.qstate) -> ()
        // Apply a CNOT with control %0 to each qubit in register %2
        "qs.CX"(%0, %2) : (!qs.qstate, !qs.rstate<4>) -> ()
        // OR in custom assembly format
        qs.CX %0, %2 : !qs.qstate, !qs.rstate<4>
        // Generate CNOT gate without application to qubits (to be manipulated via meta-ops)
        $op = "qs.CX"(%0) : (!qs.qstate) -> !qs.cop<1>
        // OR in custom assembly format
        $op = qs.CX %0 : !qs.qstate -> !qs.cop<1>
        ```
    }];

    let arguments = (ins
        Qstate_Type : $ctrl,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrCOp_Type : $res
    );

    let additionalVerifier = [{
        // check that target and control are not the same
        if (this->qbs() && this->qbs() == this->ctrl()) {
            return this->emitOpError() << "can't use the same value as control AND target!";
        }
    }];

    let assemblyFormat = [{
        operands `:` type(operands) `->` type(results) attr-dict
    }];
}

def ReturnStateOp : QuantumSSA_Op<"return", [Terminator]> {
    let summary = "Return qubit states from a function.";
    let description = [{
        This is an alternatice terminator opperation that can be
        used to return qubit states from functions, since SSA values
        inside functions are isolated from the rest of the program.
    }];

    let arguments = (outs
        Variadic<QData_Type>
    );

    let assemblyFormat = [{
       operands attr-dict `:` type(operands)
    }];
}

def FunCircOp : QuantumSSA_Op<"funcirc"> {
    let summary = "Create a quantum circuit from a function.";
    let description = [{
        A "function" quantum circuit is a collection of gates that are grouped within
        a function body. The function arguments allow the circuit to be fully parametric
        if desired. The goal of this op is to generate a value with "circuit" type, to be
        passed to meta operations. In a sense it is the SSA equivalent of the Circuit/
        ParametricCircuitOp, as these cannot produce valid operations within the value
        semantics framework.
        In order to actually execute the circuit, one can use the apply operation on the
        result of this op (or the result of subsequent meta ops), as well as directly
        apply std.call on the function name with appropriate parameters.

        Example:

        ```mlir
        func @test(%qb : !qs.state) -> !qs.state {
            qs.return %qb : !qs.state
        }
        %circ = "qs.funcirc"() {"fun"=@test} : () -> !qs.fcirc<(!qs.state) -> !qs.state>
        // OR in custom assembly form
        %circ = qs.funcirc @test -> !qs.fcirc<(!qs.state) -> !qs.state>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr : $fun,
        OptionalAttr<Confined<APIntAttr, [IntPositive]>> : $n
    );

    let results = (outs
        FunctionCircuit_Type : $funcirc
    );

    let verifier = [{
        // make sure the function exists
        auto funptr = this->resolveCallable();
        if (!funptr)
            return this->emitOpError() << "cannot resolve given function!";

        // make sure function arguments match
        auto retType = this->funcirc().getType().dyn_cast<FunCircType>();
        auto funtype = funptr->getAttrOfType<TypeAttr>("type").getValue().dyn_cast<FunctionType>();
        if (funtype != retType.getFunType())
            this->emitOpError() << "must return 'fcirc' type that matches given function type!";

        return success();
    }];

    let extraClassDeclaration = [{
        Operation *resolveCallable() {
            SymbolRefAttr callable = this->funAttr();
            return SymbolTable::lookupNearestSymbolFrom(*this, callable);
        }
    }];

    let assemblyFormat = [{
        $fun (`(` $n^ `)`)? attr-dict `->` type(results)
    }];
}

def ApplyFunCircOp : QuantumSSA_Op<"applyfc"> {
    let summary = "Apply function circuits & derivates.";
    let description = [{
        This operation used to apply a "function circuit" at the current location. It's
        the SSA equivalent of the apply operation from the 'quantum' dialect.

        The input type can either be 'fcirc' or a derivative thereof, such as
        'cop<n, fcirc>'. The remaining arguments need to match the function signature
        of the circuit exactly.

        It returns the updated state for all qubits and registers given as input.
    }];

    let arguments = (ins
        QCirc_Type : $callee,
        Variadic<AnyType> : $operands
    );

    let results = (outs
        Variadic<QData_Type> : $results
    );

    let verifier = [{
        Type circType = this->callee().getType();
        if (auto copType = circType.dyn_cast<COpType>()) {
            circType = copType.getBaseType();
            if (!circType || !circType.isa<FunCircType>())
                return this->emitOpError() << "only accepts controlled ops with base type 'fcirc'!";
        }

        auto funargTypes = circType.cast<FunCircType>().getFunType().getInputs();
        auto funresTypes = circType.cast<FunCircType>().getFunType().getResults();

        MLIRContext *ctx = getContext();
        if (filterRegSize(funargTypes, ctx) != filterRegSize(this->operands().getType(), ctx))
            return emitOpError("failed to verify that callee input types match argument types!");
        if (filterRegSize(funresTypes, ctx) != filterRegSize(this->results().getType(), ctx))
            return emitOpError("failed to verify that callee result types match result types!");

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
        $callee `(` $operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

def ControlOp : QuantumSSA_Gate<"c", [Meta]> {
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
        %op = "qs.H"() : () -> !qs.op
        // Control H op on qubit %0 and apply to qubit %1
        "qs.c"(%op, %0, %1) : (!qs.op, !qs.state, !qs.state) -> ()
        // Control H op on qubit %0 and apply to all qubits in register %2
        "qs.c"(%op, %0, %2) : (!qs.op, !qs.state, !qs.rstate<4>) -> ()
        // Control H op on all of register %2 and apply to qubit %1
        "qs.c"(%op, %2, %1) : (!qs.op, !qs.rstate<4>, !qs.qstate) -> ()

        // Control H op on qubit %1 but leave application open
        %cop = "qs.c"(%op, %1) : (!qs.op, !qs.state) -> !qs.cop<1, !qs.op>
        // OR in custom assembly format
        %cop = qs.c %op, %1 : !qs.op, !qs.state -> !qs.cop<1, !qs.op>
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        QData_Type : $ctrls,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrCOp_Type : $res
    );

    let additionalVerifier = [{
        // make sure target and control aren't identical (basic test, not comprehensive)
        if (this->qbs() && this->qbs() == this->ctrls())
            return this->emitOpError() << "can't use the same value as control AND target!";

        auto heldOp = this->heldOp().getType();
        // make sure user is not trying to apply a circuit op by specifying a target
        if (this->qbs() && (heldOp.isa<FunCircType>() || heldOp.isa<COpType>() &&
                            heldOp.cast<COpType>().getBaseType() &&
                            heldOp.cast<COpType>().getBaseType().isa<FunCircType>()))
            return this->emitOpError() << "does not accept target qubits when controlling circuit!";

        // obtain total number of control qubits from input
        unsigned totalctrls = 0;

        if (heldOp.isa<COpType>())
            totalctrls += heldOp.cast<COpType>().getNumCtrls();

        auto ctrls = this->ctrls().getType();
        if (ctrls.isa<QstateType>()) {
            totalctrls++;
        } else if (auto regType = ctrls.dyn_cast<RstateType>()) {
            if (auto numQubits = regType.getNumQubits())
                totalctrls += *numQubits;
            else
                totalctrls = -1; // disable # of ctrl qubits check when no size is provided
        } else {
            assert(false && "Unknown data-type encountered in ControlOp!");
        }

        // perform several checks on the return type, if it is a COpType
        if (this->res().getType().isa<COpType>()) {
            COpType output = this->res().getType().cast<COpType>();
            // assert correct number of control qubits
            if (totalctrls != -1 && output.getNumCtrls() != totalctrls)
                return this->emitOpError() << "with mismatched number of controls in return type "
                                              "vs inputs!";

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
        operands `:` type(operands) `->` type(results) attr-dict
    }];
}

def AdjointOp : QuantumSSA_Gate<"adj", [Meta]> {
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
        %op = "qs.H"() : () -> !qs.op
        // Invert H op and apply to qubit %0
        "qs.adj"(%op, %0) : (!qs.op, !qs.qstate) -> ()
        // Invert H op apply to all qubits in register %1
        "qs.adj"(%op, %1) : (!qs.op, !qs.rstate<4>) -> ()
        // Invert H op but leave application open
        %opdg = "qs.adj"(%op) : (!qs.op) -> !qs.op
        // OR in custom assembly format
        %opdg = qs.adj %op : !qs.op -> !qs.op
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        Optional<QData_Type> : $qbs
    );

    let results = (outs
        QDataOrQOp_Type : $res
    );

    string additionalVerifier = [{
        auto heldOp = this->heldOp().getType();
        if (!this->qbs() && heldOp != this->res().getType())
            return this->emitOpError() << "given operation type must match output type!";

        // make sure user is not trying to apply a circuit op by specifying a target
        if (this->qbs() && (heldOp.isa<FunCircType>() || heldOp.isa<COpType>() &&
                            heldOp.cast<COpType>().getBaseType().isa<FunCircType>()))
            return this->emitOpError() << "does not accept target qubits when inverting circuit!";
    }];

    let assemblyFormat = [{
        operands `:` type(operands) `->` type(results) attr-dict
    }];
}

#endif // QUANTUM_SSA_OPS
