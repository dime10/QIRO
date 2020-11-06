#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

// Base class for "gates", i.e. operations executable on a quantum computer
class Quantum_Gate<string mnemonic, list<OpTrait> traits = []> : Quantum_Op<mnemonic, traits> {
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
        if (!this->qbs() && !this->op())
            return this->emitOpError() << "needs either target qubits or return type!";
        if (this->qbs() && this->op())
            return this->emitOpError() << "cannot have both target qubits and return type!";

        return success();
    }];
}

def AllocOp : Quantum_Op<"alloc", [NoSideEffect]> {
    let summary = "Allocate a qubit.";
    let description = [{
        The allocation command registers a new qubit with the system, which
        is guaranteed to be in the |0> state.

        This operation takes no inputs and returns a value of type 'qubit'.

        Example:

        ```mlir
        // Allocate single qubit
        %qb = "q.alloc"() : () -> !q.qubit
        // OR in custom assembly format
        %qb = q.alloc -> !q.qubit
        ```
    }];

    let results = (outs
        Qubit_Type : $qb
    );

    let assemblyFormat = [{
        attr-dict `->` type($qb)
    }];
}

def AllocRegOp : Quantum_Op<"allocreg", [NoSideEffect]> {
    let summary = "Allocate a qubit register.";
    let description = [{
        This allocation command registers a new qubit register with the system,
        which is guaranteed to be in the |0..> state.

        This operation takes an integer attribute as input specifying the size,
        and returns a value of type 'qureg<size>'.

        Example:

        ```mlir
        // Allocate qubit register
        %reg = "q.allocreg"() {static_size=4} : () -> !q.qureg<4>
        // OR in custom assembly format
        %reg = q.allocreg(4) -> !q.qureg<4>
        ```
    }];

    let arguments = (ins
        Optional<Index> : $size,
        OptionalAttr<Confined<I64Attr, [IntMinValue<2>]>> : $static_size
    );

    let results = (outs
        Qureg_Type : $reg
    );

    let verifier = [{
        if (this->size() && this->static_size())
            return this->emitOpError() << "cannot provide register size as attribute AND operand!";
        if (!this->size() && !this->static_size())
            return this->emitOpError() << "requires a register size!";

        // check return type matches type of argument
        QuregType regType = this->reg().getType().cast<QuregType>();
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

def CastRegOp : Quantum_Op<"cast"> {
    let summary = "Register-size type cast.";

    let arguments = (ins Qureg_Type);
    let results = (outs Qureg_Type);

    let verifier = [{
        if (!this->getOperand().getType().cast<QuregType>().getNumQubits())
            return this->emitOpError() << "needs static-size register as input!";
        if (this->getResult().getType().cast<QuregType>().getNumQubits())
            return  this->emitOpError() << "needs to return dynamic-size register!";
        return success();
    }];

    let assemblyFormat = [{
        operands attr-dict `:` type(operands) `to` type(results)
    }];
}

def FreeOp : Quantum_Op<"free"> {
    let summary = "Deallocate a qubit.";
    let description = [{
        The free command deallocates a qubit with the system. The qubit is not
        required to be in the |0> state, but it must be disentangled from all
        other qubits.

        This operation takes one operand of type 'qubit'.

        Example:

        ```mlir
        // Deallocate single qubit
        "q.free"(%qb) : (!q.qubit) -> ()
        // OR in custom assembly format
        q.free %qb : !q.qubit
        ```
    }];

    let arguments = (ins
        Qubit_Type
    );

    let assemblyFormat = [{
        operands attr-dict `:` type(operands)
    }];
}

def FreeRegOp : Quantum_Op<"freereg"> {
    let summary = "Deallocate a qubit register.";
    let description = [{
        This free command deallocates a qubit register with the system. The register
        is not required to be in the |0..> state, but every qubit inside must be
        disentangled from all other qubits.

        This operation takes one operand of type 'qureg'.

        Example:

        ```mlir
        // Deallocate qubit register
        "q.freereg"(%reg) : (!q.qureg<4>) -> ()
        // OR in custom assembly format
        q.freereg %reg : !q.qureg<4>
        ```
    }];

    let arguments = (ins
        Qureg_Type : $reg
    );

    let assemblyFormat = [{
        operands attr-dict `:` type(operands)
    }];
}

def GenRegOp : Quantum_Op<"genreg", [NoSideEffect]> {
    let summary = "Generate a register from multiple qubits/quregs.";
    let description = [{
        This op allows to combine previously allocated qubits/quregs
        into a new register for the purpose of passing it to other operations.
        All arguments given to this op will become ivalid and must not
        used again. This is checked by a pass.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=2} : () -> !q.qureg<2>
        // Generate new register from other qubits/registers
        %reg = "q.genreg"(%0, %1) : (!q.qubit, !q.qureg<2>) -> !q.qureg<3>
        // OR in custom assembly format
        %reg = q.genreg %0, %1 : !q.qubit, !q.qureg<2> -> !q.qureg<3>
        ```
    }];

    let arguments = (ins
        Variadic<QData_Type> : $inputs
    );

    let results = (outs
        Qureg_Type : $newreg
    );

    let verifier = [{
        // require atleast two arguments to be combined together
        if (this->inputs().size() < 2)
            return this->emitOpError() << "requires >= 2 arguments to be combined into a register!";

        // tally the number of qubits from input types if available
        int totalqbs = 0;
        bool lowerbound = false;
        for (auto argType : this->inputs().getTypes()) {
            if (argType.isa<QubitType>()) {
                totalqbs++;
            } else if (auto regType = argType.dyn_cast<QuregType>()) {
                if (auto numQubits = regType.getNumQubits())
                    totalqbs += *numQubits;
                else
                    lowerbound = true;
            }
        }

        // perform as much static analysis as possible on the qubit number given the available type
        // info: for inputs with no registers of undeclared size -> exact comparison
        //       else -> qubit number from inputs provides lower bound on size of output register
        if (auto declsize = this->newreg().getType().cast<QuregType>().getNumQubits()) {
            if (!lowerbound && *declsize != totalqbs)
                return this->emitOpError() << "with mismatched number of qubits in result type vs "
                                              "sum of qubits in the input!";
            else if (*declsize < totalqbs)
                return this->emitOpError() << "with too few qubits in result type vs sum of qubits "
                                              "in the input!";
        }

        // invalidate all arguments to this operation (no more uses beyond this point)
        for (auto arg : this->inputs()) {
            Value::user_range users = arg.getUsers();
            if (dyn_cast_or_null<GenRegOp>(*users.begin()) != *this) {
                std::string errMsg("Operand value ");
                llvm::raw_string_ostream stream(errMsg);
                AsmState state(this->getParentOp());
                arg.printAsOperand(stream, state);
                errMsg += " is invalid beyond current op, but detected further use!";
                return this->emitError() << errMsg;
            }
        }

        return success();
    }];

    let assemblyFormat = [{
        operands attr-dict `:` type(operands) `->` type(results)
    }];
}

def HOp : Quantum_Gate<"H", [Unitary, Hermitian, AttrSizedOperandSegments]> {
    let summary = "Hadamard gate.";
    let description = [{
        The Hadamard gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,  1],
             [1, -1]] / sqrt(2)

        This operation takes a qubit or register as input or alternatively
        returns itself with a 'u1' type.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Apply a Hadamard gate to qubit %0
        "q.H"(%0) : (!q.qubit) -> ()
        // Apply a Hadamard gate to each qubit in register %1
        "q.H"(%1) : (!q.qureg<4>) -> ()
        // OR in custom assembly format
        q.H %1 : !q.qureg<4>
        // Generate Hadamard op without application to qubits (to be manipulated via meta-ops)
        $op = "q.H"() : () -> !q.u1
        // OR in custom assembly format
        %op = q.H -> !q.u1
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<U1_Type> : $op
    );

    let assemblyFormat = [{
        custom<OptionalQDataArg>($qbs, $range, type($range), $static_range) attr-dict
        custom<U1TypeSig>(type($qbs), type($op))
    }];
}

def XOp : Quantum_Gate<"X", [Unitary, Hermitian, AttrSizedOperandSegments]> {
    let summary = "NOT gate.";
    let description = [{
        The NOT gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[0, 1],
             [1, 0]]

        This operation takes a qubit or register as input or alternatively
        returns itself with a 'u1' type.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Apply a NOT gate to qubit %0
        "q.X"(%0) : (!q.qubit) -> ()
        // Apply a NOT gate to each qubit in register %1
        "q.X"(%1) : (!q.qureg<4>) -> ()
        // OR in custom assembly format
        q.X %1 : !q.qureg<4>
        // Generate NOT op without application to qubits (to be manipulated via meta-ops)
        $op = "q.X"() : () -> !q.u1
        // OR in custom assembly format
        %op = q.X -> !q.u1
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<U1_Type> : $op
    );

    let assemblyFormat = [{
        custom<OptionalQDataArg>($qbs, $range, type($range), $static_range) attr-dict
        custom<U1TypeSig>(type($qbs), type($op))
    }];
}

def RzOp : Quantum_Gate<"RZ", [Unitary, AttrSizedOperandSegments]> {
    let summary = "Z-Rotation gate.";
    let description = [{
        The Rz gate is a single Qubit unitary.
        It rotates a qubit state by an angle φ along the z-axis.
        Its 2x2 matrix representation in the computational basis is given by:
            [[e^-iφ/2,    0  ],
             [   0   , e^iφ/2]]

        This operation takes a qubit or register as input or alternatively
        returns itself with a 'u1' type. The parameter φ is mandatory and
        can be provided as an attribute or operand.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Apply a z-rotation by angle 0.1 to qubit %0
        "q.RZ"(%0) {phi=0.1}: (!q.qubit) -> ()
        // Apply a z-rotation by angle 0.1 to each qubit in register %1
        "q.RZ"(%1) {phi=0.1} : (!q.qureg<4>) -> ()
        // OR in custom assembly format
        q.RZ(0.1) %1 : !q.qureg<4>
        // Generate RZ gate without application to qubits (to be manipulated via meta-ops)
        $op = "q.RZ"() {phi=0.1} : () -> !q.u1
        // OR in custom assembly format
        %op = q.RZ(0.1) -> !q.u1
        ```
    }];

    let arguments = (ins
        Optional<AnyFloat> : $phi,
        OptionalAttr<F64Attr> : $static_phi,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<U1_Type> : $op
    );

    let verifier = [{
        if (this->phi() && this->static_phi())
            return this->emitOpError() << "cannot accept angle as both operand & attribute!";

        if (!this->phi() && !this->static_phi())
            return this->emitOpError() << "required angle as either operand or attribute!";

        return success();
    }];

    let assemblyFormat = [{
        `(` custom<FloatParam>($phi, type($phi), $static_phi) `)`
        custom<OptionalQDataArg>($qbs, $range, type($range), $static_range) attr-dict
        custom<U1TypeSig>(type($qbs), type($op))
    }];
}

def ROp : Quantum_Gate<"R", [Unitary, AttrSizedOperandSegments]> {
    let summary = "Rotation gate.";
    let description = [{
        The R gate is a single Qubit unitary.
        It rotates a qubit state by an angle φ about the |1> state.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,   0 ],
             [0, e^iφ]]

        This operation takes a qubit or register as input or alternatively
        returns itself with a 'u1' type. The parameter φ is mandatory and
        can be provided as an attribute or operand.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Apply a rotation by angle 0.1 to qubit %0
        "q.R"(%0) {phi=0.1}: (!q.qubit) -> ()
        // Apply a rotation by angle 0.1 to each qubit in register %1
        "q.R"(%1) {phi=0.1} : (!q.qureg<4>) -> ()
        // OR in custom assembly format
        q.R(0.1) %1 : !q.qureg<4>
        // Generate R gate without application to qubits (to be manipulated via meta-ops)
        $op = "q.R"() {phi=0.1} : () -> !q.u1
        // OR in custom assembly format
        %op = q.R(0.1) -> !q.u1
        ```
    }];

    let arguments = (ins
        Optional<AnyFloat> : $phi,
        OptionalAttr<F64Attr> : $static_phi,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<U1_Type> : $op
    );

    let verifier = [{
        if (this->phi() && this->static_phi())
            return this->emitOpError() << "cannot accept angle as both operand & attribute!";

        if (!this->phi() && !this->static_phi())
            return this->emitOpError() << "required angle as either operand or attribute!";

        return success();
    }];

    let assemblyFormat = [{
        `(` custom<FloatParam>($phi, type($phi), $static_phi) `)`
        custom<OptionalQDataArg>($qbs, $range, type($range), $static_range) attr-dict
        custom<U1TypeSig>(type($qbs), type($op))
    }];
}

def CNotOp : Quantum_Gate<"CX", [Unitary, Hermitian, AttrSizedOperandSegments]>{
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
        a register (in which case the gate is applied to every qubit in the register).
        The control however must be a singular qubit. The return type is 'u2'.
        It acts on the qubits via side-effect by changing the quantum state.
        Note that the control qubit MUST NOT appear as one of target qubits (not checked atm).

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.alloc"() : () -> !q.qubit
        %2 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Apply a CNOT to target %1 and control %0
        "q.CX"(%0, %1) : (!q.qubit, !q.qubit) -> ()
        // Apply a CNOT with control %0 to each qubit in register %2
        "q.CX"(%0, %2) : (!q.qubit, !q.qureg<4>) -> ()
        // OR in custom assembly format
        q.CX %0, %2 : !q.qubit, !q.qureg<4>
        // Generate CNOT gate without application to qubits (to be manipulated via meta-ops)
        $op = "q.CX"() : () -> !q.u2
        // OR in custom assembly format
        $op = q.CX -> !q.u2
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $ctrl,
        Variadic<Index> : $crange,
        I64ArrayAttr : $static_crange,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $qrange,
        I64ArrayAttr : $static_qrange
    );

    let results = (outs
        Optional<U2_Type> : $op
    );

    let additionalVerifier = [{
        if (this->ctrl() && !this->qbs() || !this->ctrl() && this->qbs())
            return this->emitOpError() << "accepts either both control and target or neither!";

        if (this->ctrl() && this->ctrl().getType().isa<QuregType>()
                && this->static_crange().size() != 1)
            return this->emitOpError() << "needs single qubit as control, thus if value is a "
                                          "register, must provide exactly 1 accessor index!";
    }];

    let assemblyFormat = [{
        custom<TwoQDataArg>($ctrl, $crange, type($crange), $static_crange,
                            $qbs, $qrange, type($qrange), $static_qrange)
        attr-dict custom<U2TypeSig>(type($ctrl), type($qbs), type($op))
    }];
}

def SwapOp : Quantum_Gate<"SWAP", [Unitary, Hermitian, AttrSizedOperandSegments, Commutative]> {
    let summary = "SWAP gate.";
    let description = [{
        The SWAP gate is a two Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 4x4 matrix representation in the computational basis is given by:
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]

        Both targets must be a singular qubit. The return type is 'u2'.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.alloc"() : () -> !q.qubit
        // Apply a SWAP to qubits %0 and %1
        "q.SWAP"(%0, %1) : (!q.qubit, !q.qubit) -> ()
        // OR in custom assembly format
        q.SWAP %0, %1 : !q.qubit, !q.qubit
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range,
        Optional<QData_Type> : $qbs2,
        Variadic<Index> : $range2,
        I64ArrayAttr : $static_range2
    );

    let results = (outs
        Optional<U2_Type> : $op
    );

    let additionalVerifier = [{
        if (this->qbs() && !this->qbs2() || !this->qbs() && this->qbs2())
            return this->emitOpError() << "accepts either both operands or neither!";

        std::string errMsg = "needs single qubit for both operands, thus if any value is a "
                             "register, must provide exactly 1 accessor index for that value!";
        if (this->qbs() && this->qbs().getType().isa<QuregType>()
                && this->static_range().size() != 1)
            return this->emitOpError() << errMsg;
        if (this->qbs2() && this->qbs2().getType().isa<QuregType>()
                && this->static_range2().size() != 1)
            return this->emitOpError() << errMsg;
    }];

    let assemblyFormat = [{
        custom<TwoQDataArg>($qbs, $range, type($range), $static_range,
                            $qbs2, $range2, type($range2), $static_range2)
        attr-dict custom<U2TypeSig>(type($qbs), type($qbs2), type($op))
    }];
}

def TerminatorOp : Quantum_Op<"term", [Terminator]> {
    let summary = "Marks the end of a circuit (block).";
    let description = [{
        This is an internal operation solely used to mark the end
        of an mlir "block" which is used to hold all operations in
        the circuit. A "terminator" op is required for all blocks
        that do not transfer control flow.
    }];

    let assemblyFormat = [{
       attr-dict
    }];
}

def CircuitOp : Quantum_Op<"circ", [Symbol, FunctionLike, IsolatedFromAbove,
                                    AffineScope, ImplicitQuantumTerminator,
                                    DeclareOpInterfaceMethods<CallableOpInterface>]> {
    let summary = "Create a function-like quantum circuit.";
    let description = [{
        A quantum circuit defines a custom operation on the quantum computer by
        grouping multiple elementary gates or other circuit calls into one unit.
        The main use of this is to reuse a gate sequence in multiple places in the
        program, as well as apply the operation to different and differently-sized
        qubit sets (if the circuit is defined parametrically).

        The body of the circuit is contained within a region, defines an affine
        scope for affine for loops, and is isolated from above, meaning no values
        from outside this operation can be referenced in the body.

        //While the arguments to circuit may be of any type to be as general as
        //possible, there is a a requirement for quantum registers. For each such
        //register that does not have a static size, an index argument indicating
        //a lower limit on the size of the register is required immediately following
        //the register argument.

        Note: The containing block must always be terminated with the special
              'TerminatorOp'. In custom assembly form this can be omitted.

        Example:

        ```mlir
        "q.circ"() ({
        ^bb0(%r: !q.qureg<>):
            "q.H"(%r) : (!q.qureg<>) -> ()
            "q.term"() : () -> ()
        }) {sym_name="h", type=(!q.qureg<>) -> ()} : () -> ()
        // OR in custom assembly
        q.circ @h(%r: !q.qureg<>) {
            "q.H"(%r) : (!q.qureg<>) -> ()
        }
        ```
    }];

    let arguments = (ins
        UnitAttr : $no_inline
    );

    let regions = (region
        AnyRegion : $gates
    );

    let extraClassDeclaration = [{
        // FunctionLike trait needs access to the functions below.
        friend class OpTrait::FunctionLike<CircuitOp>;

        // Required hooks for the input/output type enumeration in FunctionLike.
        unsigned getNumFuncArguments() {
            return OpTrait::FunctionLike<CircuitOp>::getType().getNumInputs();
        }
        unsigned getNumFuncResults() {
            return OpTrait::FunctionLike<CircuitOp>::getType().getNumResults();
        }

        // Verifies the type of the function for op specific invariants. Optional.
        //LogicalResult verifyType() {
        //    if (this->getNumFuncResults())
        //        return this->emitOpError() << "cannot return any values!";
        //     return success();
        //}

        // Verifies the body of the function for op specific invariants. Optional.
        //LogicalResult verifyBody();
    }];

    let builders = [
        OpBuilder<"StringRef name, FunctionType type", [{
            $_state.addAttribute(SymbolTable::getSymbolAttrName(),
                                 $_builder.getStringAttr(name));
            $_state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
            $_state.addRegion();
        }]>
    ];

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
}

def CircuitValueOp : Quantum_Op<"getval", [NoSideEffect]> {
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

def CallCircOp : Quantum_Op<"call", [AttrSizedOperandSegments,
                                     DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "Perform direct call to quantum circuit with given arguments.";
    let description = [{
        This operation is used to perform a call to a function-like quantum
        circuit, which is referenced via a symbol attribute.

        The operands, which can be of any type, are expected to match the
        arguments defined in the circuit signature.

        Example:

        ```mlir
        q.circ @h(%r: !q.qureg<>) {
            "q.H"(%r) : (!q.qureg<>) -> ()
            "q.term"() : () -> ()
        }

        %0 = "q.allocreg"() {n=4} : () -> !q.qureg<4>

        // Call circuit with qubit register %0
        "q.call"(%0) {circref=@h} : (!q.qureg<4>) -> ()
        // OR in custom assembly
        q.call @h(%0) : !q.qureg<4>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr : $circref,
        Variadic<AnyType> : $args,
        Variadic<Index> : $ranges,
        2dI64ArrayAttr : $static_ranges,
        I64ArrayAttr : $size_params
    );

    let verifier = [{
        // check 2d range attribute
        if (this->static_ranges().size() != this->args().size())
            return this->emitOpError() << "expected proper 2D static range attribute!";
        // check size paramter attribute
        if (this->size_params().size() < this->args().size())
            return this->emitOpError() << "expected proper size param attribute!";

        // check circuit symbol
        Operation *callable = cast<CallOpInterface>(this->getOperation()).resolveCallable();
        if (!callable)
            return this->emitOpError() << "could not resolve given symbol!";

        if (!isa<CircuitOp>(callable))
            return this->emitOpError() << "symbol does not reference a 'CircuitOp'!";

        // make sure function arguments match
        CircuitOp circOp = cast<CircuitOp>(callable);
        FunctionType circType = circOp.OpTrait::FunctionLike<CircuitOp>::getType();

        // compare number of arguments with function signature
        if (circType.getNumInputs() != this->size_params().size())
            return this->emitOpError() << "has wrong number of circuit arguments! "
                                          "Circuit requires: " << circType.getNumInputs() << ". "
                                          "Got: " << this->size_params().size() << ".";

        // compare each parameter type:
        //  - in general all types must match the circuit signature exactly, with 2 exceptions
        //  - index arguments can also be provided as static attributes, either all or none of them
        //  - registers can appear in 2 forms in the circuit signature:
        //      as !q.qureg<n> or !q.qureg<>, both of which can be matched with either, however
        //      only in the case where both have size information will the given register be checked
        //      to be at least as large as required, from which the first n qubits will be used
        int numIdxArgs = 0;
        int numSizeParams = this->size_params().size() - this->args().size();
        std::string errMsg = "detected wrong argument type! Expected: ";
        auto sigType = circType.getInputs().begin();
        for (auto argType : this->args().getTypes()) {
            if (sigType->isa<IndexType>()) {
                if (!argType.isa<IndexType>() && !numSizeParams)
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
                numIdxArgs++;
            } else if (auto sigRegType = sigType->dyn_cast<QuregType>()) {
                if (auto argRegType = argType.dyn_cast<QuregType>()) {
                    if (argRegType.getNumQubits() && sigRegType.getNumQubits())
                        if (*argRegType.getNumQubits() < *sigRegType.getNumQubits())
                            return this->emitOpError() << "too few qubits in register! Expected: "
                                                       << *sigRegType.getNumQubits() << ". Got: "
                                                       << *argRegType.getNumQubits() << ".";
                } else {
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
                }
            } else {
                if (*sigType != argType)
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
            }
            sigType++;
        }
        // verify enough static int attributes were provided for all index arguments
        if (numSizeParams && numIdxArgs != numSizeParams)
            return this->emitOpError() << "did not receive all index arguments as attributes!";

        return success();
    }];

    let assemblyFormat = [{
        $circref `(`
            custom<ArbitraryArgs>($args, $ranges, type($ranges), $static_ranges, $size_params)
        `)`
        attr-dict `:` type($args)
    }];
}

def ApplyCircOp : Quantum_Op<"apply", [AttrSizedOperandSegments]> {
    let summary = "Apply quantum circuit to given arguments.";
    let description = [{
        In contrast to the call operation which uses a symbol to reference the
        circuit, this operation calls a circuit via the provided SSA operand.
        Typically, such a value is created to be manipulated via quantum meta
        operations, and is ultimately applied to qubits via this op.

        The circuit can be any value of type '!q.circ' or a type derived from
        it, such as '!q.cop<n, !q.circ>.

        The remaining operands, which can be of any type, will be forwarded
        to the circuit, and must match its declared arguments. The exception
        being index operands, which can also be provided as static attributes
        for convenience.

        Example:

        ```mlir
        "q.circ"() ({
        ^bb0(%r: !q.qureg<>):
            "q.H"(%r) : (!q.qureg<>) -> ()
            "q.term"() : () -> ()
        }) {sym_name="h", type=(!q.qureg<>) -> ()} : () -> ()

        %0 = "q.allocreg"() {n=4} : () -> !q.qureg<4>
        %c = q.getval @h -> !q.circ

        // Apply circuit to qubit register %0
        "q.apply"(%c, %0) : (!q.circ, !q.qureg<4>) -> ()
        // OR in custom assembly
        q.apply %c(%0) : !q.circ(!q.qureg<4>)
        ```
    }];

    let arguments = (ins
        QCirc_Type : $circval,
        Variadic<AnyType> : $args,
        Variadic<Index> : $ranges,
        2dI64ArrayAttr : $static_ranges,
        I64ArrayAttr : $size_params
    );

    let verifier = [{
        // check 2d range attribute
        if (this->static_ranges().size() != this->args().size())
            return this->emitOpError() << "expected proper 2D static range attribute!";
        // check size paramter attribute
        if (this->size_params().size() < this->args().size())
            return this->emitOpError() << "expected proper size param attribute!";

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

        // compare number of arguments while ignoring parameter in function signature
        if (circType.getNumInputs() != this->size_params().size())
            return this->emitOpError() << "has wrong number of circuit arguments! "
                                          "Circuit requires: " << circType.getNumInputs() << ". "
                                          "Got: " << this->size_params().size() << ".";

        // compare each parameter type:
        //  - in general all types must match the circuit signature exactly, with 2 exceptions
        //  - index arguments can also be provided as static attributes, either all or none of them
        //  - registers can appear in 2 forms in the circuit signature:
        //      as !q.qureg<n> or !q.qureg<>, both of which can be matched with either
        //      only in the case where both have size information will the given register be checked
        //      to be at least as large as required, from which the first n qubits will be used
        int numIdxArgs = 0;
        int numSizeParams = this->size_params().size() - this->args().size();
        std::string errMsg = "detected wrong argument type! Expected: ";
        auto sigType = circType.getInputs().begin();
        for (auto argType : this->args().getTypes()) {
            if (sigType->isa<IndexType>()) {
                if (!argType.isa<IndexType>() && !numSizeParams)
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
                numIdxArgs++;
            } else if (auto sigRegType = sigType->dyn_cast<QuregType>()) {
                if (auto argRegType = argType.dyn_cast<QuregType>()) {
                    if (argRegType.getNumQubits() && sigRegType.getNumQubits())
                        if (*argRegType.getNumQubits() < *sigRegType.getNumQubits())
                            return this->emitOpError() << "too few qubits in register! Expected: "
                                                       << *sigRegType.getNumQubits() << ". Got: "
                                                       << *argRegType.getNumQubits() << ".";
                } else {
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
                }
            } else {
                if (*sigType != argType)
                    return this->emitOpError() << errMsg << *sigType << ". Got: " << argType << ".";
            }
            sigType++;
        }
        // verify enough static int attributes were provided for all index arguments
        if (numSizeParams && numIdxArgs != numSizeParams)
            return this->emitOpError() << "did not receive all index arguments as attributes!";

        return success();
    }];

    let assemblyFormat = [{
        $circval `(`
            custom<ArbitraryArgs>($args, $ranges, type($ranges), $static_ranges, $size_params)
        `)`
        attr-dict `:` type($circval) `(` type($args) `)`
    }];
}

def ControlOp : Quantum_Gate<"ctrl", [Meta, AttrSizedOperandSegments]> {
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
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.alloc"() : () -> !q.qubit
        %2 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        %op = "q.H"() : () -> !q.u1
        // Control H op on qubit %0 and apply to qubit %1
        "q.ctrl"(%op, %0, %1) : (!q.u1, !q.qubit, !q.qubit) -> ()
        // Control H op on qubit %0 and apply to all qubits in register %2
        "q.ctrl"(%op, %0, %2) : (!q.u1, !q.qubit, !q.qureg<4>) -> ()
        // Control H op on all of register %2 and apply to qubit %1
        "q.ctrl"(%op, %2, %1) : (!q.u1, !q.qureg<4>, !q.qubit) -> ()
        // Control H op on qubit %0 but leave application open
        %cop = "q.ctrl"(%op, %1) : (!q.u1, !q.qubit) -> !q.cop<1>
        // OR in custom assembly format
        %cop = q.ctrl %op, %1 : !q.u1, !q.qubit -> !q.cop<1>
        ```
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        QData_Type: $ctrls,
        Variadic<Index> : $crange,
        I64ArrayAttr : $static_crange,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range,
        Optional<QData_Type> : $qbs2,
        Variadic<Index> : $range2,
        I64ArrayAttr : $static_range2
    );

    let results = (outs
        Optional<ControlledOp_Type> : $op
    );

    let additionalVerifier = [{
        auto heldOp = this->heldOp().getType();

        if (this->qbs2() && !this->qbs())
            return this->emitOpError() << "does not accept 2nd qubit arg without 1st!";

        // check presence of operands depending on heldOp type (u1 vs u2)
        if (!this->op()) {
            if (heldOp.isa<U1Type>() && this->qbs2())
                return this->emitOpError() << "does not accept second qubit arg for U1 op!";
            if (heldOp.isa<U2Type>() && !this->qbs2())
                return this->emitOpError() << "requires second qubit arg for U2 op!";
        }

        // make sure user is not trying to apply a circuit op by specifying a target
        if (this->qbs() && (heldOp.isa<CircType>() || heldOp.isa<COpType>() &&
                            heldOp.cast<COpType>().getBaseType().isa<CircType>()))
            return this->emitOpError() << "does not accept target qubits when controlling circuit!";

        // obtain total number of control qubits from input
        int totalctrls = 0;

        auto ctrls = this->ctrls().getType();
        if (ctrls.isa<QubitType>()) {
            totalctrls++;
        } else if (auto regType = ctrls.dyn_cast<QuregType>()) {
            if (auto numQubits = regType.getNumQubits())
                totalctrls += *numQubits;
            else if (this->static_crange().size() == 1)
                totalctrls += 1;
            else
                totalctrls = -1; // disable # of ctrl qubits check when no size is provided
        }

        if (auto cop = heldOp.dyn_cast<COpType>()) {
            if (cop.getNumCtrls() && totalctrls != -1)
                totalctrls += *cop.getNumCtrls();
            else
                totalctrls = -1; // disable # of ctrl qubits check if base op is 'cop' without #
        }

        if (this->static_crange().size() > 1)
            totalctrls = -1; // disable # of ctrl qubits check when using register accessors

        // perform several checks on the return type
        if (this->op()) {
            COpType output = this->op().getType().cast<COpType>();
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
        $heldOp `,`
        custom<CtrlArgs>($ctrls, $crange, type($crange), $static_crange,
                         $qbs, $range, type($range), $static_range,
                         $qbs2, $range2, type($range2), $static_range2)
        attr-dict `:` type($heldOp) `,`
        custom<CtrlTypeSig>(type($ctrls), type($qbs), type($qbs2), type($op))
    }];
}

def AdjointOp : Quantum_Gate<"adj", [Meta, AttrSizedOperandSegments]> {
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
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        %op = "q.H"() : () -> !q.u1
        // Invert H op and apply to qubit %0
        "q.adj"(%op, %0) : (!q.u1, !q.qubit) -> ()
        // Invert H op apply to all qubits in register %1
        "q.adj"(%op, %1) : (!q.u1, !q.qureg<4>) -> ()
        // Invert H op but leave application open
        %opdg = "q.adj"(%op) : (!q.u1) -> !q.u1
        // OR in custom assembly format
        %opdg = q.adj %op : !q.u1 -> !q.u1
        ```
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range,
        Optional<QData_Type> : $qbs2,
        Variadic<Index> : $range2,
        I64ArrayAttr : $static_range2
    );

    let results = (outs
        Optional<QOp_Type> : $op
    );

    let additionalVerifier = [{
        auto heldOp = this->heldOp().getType();

        if (this->qbs2() && !this->qbs())
            return this->emitOpError() << "does not accept 2nd qubit arg without 1st!";

        // check presence of operands depending on heldOp type (u1 vs u2)
        if (!this->op()) {
            if (heldOp.isa<U1Type>() && this->qbs2())
                return this->emitOpError() << "does not accept second qubit arg for U1 op!";
            if (heldOp.isa<U2Type>() && !this->qbs2())
                return this->emitOpError() << "requires second qubit arg for U2 op!";
        }

        if (heldOp.isa<CircType>() && this->qbs())
            return this->emitOpError() << "does not accept target qubits when inverting circuit!";

        if (this->op() && heldOp != this->op().getType())
            return this->emitOpError() << "given operation type must match output type!";
    }];

    let assemblyFormat = [{
        $heldOp
        custom<MetaQDataArgs>($qbs, $range, type($range), $static_range,
                              $qbs2, $range2, type($range2), $static_range2)
        attr-dict `:` type($heldOp)
        custom<MetaTypeSig>(type($qbs), type($qbs2), type($op))
    }];
}

def MeasurementOp : Quantum_Op<"meas"> {
    let summary = "Measure qubits.";
    let description = [{
        This operation performs a measurement on individual or groups of qubits.
        The measurement results are returned as clasical boolean values.

        Example:
        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // measure single qubit
        %m = "q.meas"(%0) {static_range=[]} : (!q.qubit) -> i1
        // OR in custom assebly format
        %m = q.meas %0 : !q.qubit -> i1
        // measure register
        %mr = "q.meas"(%1) {static_range=[]} : (!q.qureg<4>) -> memref<4xi1>
        // OR in custom assembly format
        %mr = q.meas %1 : !q.qureg<4> -> memref<4xi1>
        ```
    }];

    let arguments = (ins
        QData_Type : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        AnyTypeOf<[I1, I1MemRef]> : $res
    );

    let verifier = [{
        Type qbsType = this->qbs().getType();
        Type resType = this->res().getType();
        int staticSize = this->static_range().size();

        if (auto mrefType = resType.dyn_cast<MemRefType>()) {
            if (!mrefType.hasRank() || mrefType.getRank() != 1)
                return this->emitOpError() << "only produces ranked memrefs of 1 dimension!";

            if (auto qregType = qbsType.dyn_cast<QuregType>()) {
                if (staticSize == 0 && qregType.getNumQubits()) {
                    if (mrefType.isDynamicDim(0))
                        return this->emitOpError() << "if available must provide static dim size!";
                    else if (*qregType.getNumQubits() != mrefType.getDimSize(0))
                        return this->emitOpError() << "mismatch in qureg and memref size!";
                } else if (staticSize == 1) {
                    return this->emitOpError() << "must return 'i1' when measuring reg. qubit!";
                }
            } else {
                return this->emitOpError() << "must return 'i1' when measuring qubit!";
            }
        } else if (!qbsType.isa<QubitType>() && staticSize != 1) {
            return this->emitOpError() << "must return memref when measuring multiple qubits!";
        }

        return success();
    }];

    let assemblyFormat = [{
        custom<QDataArg>($qbs, $range, type($range), $static_range) attr-dict
        `:` type($qbs) `->` type(results)
    }];
}

#endif // QUANTUM_OPS
