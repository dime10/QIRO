#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "QuantumInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

// Base class for "gates", i.e. operations executable on a quantum computer
class Quantum_Gate<string mnemonic, list<OpTrait> traits = []> : Quantum_Op<mnemonic, traits> {
    let description = [{
        A "gate" op is one that supports the instation of itself without requiring
        immediate application to specific qubits. That means that the target qubits
        need not necessarily be supplied (signifying the op is on hold) and need to
        be tagged 'optional' in ODS, with a fixed name (used by the verifier): "qbs".
        Similarly, the output of type !q.op will only be generated for an op on hold,
        and thus needs to be tagged 'optional' as well, with the fixed name: "op".
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

def AllocOp : Quantum_Op<"alloc"> {
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

def AllocRegOp : Quantum_Op<"allocreg"> {
    let summary = "Allocate a qubit register.";
    let description = [{
        This allocation command registers a new qubit register with the system,
        which is guaranteed to be in the |0..> state.

        This operation takes an integer attribute as input specifying the size,
        and returns a value of type 'qureg<size>'.

        Example:

        ```mlir
        // Allocate qubit register
        %reg = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // OR in custom assembly format
        %reg = q.allocreg(4) -> !q.qureg<4>
        ```
    }];

    let arguments = (ins
        Confined<APIntAttr, [IntMinValue<2>]> : $size
    );

    let results = (outs
        Qureg_Type : $reg
    );

    let assemblyFormat = [{
        `(`$size`)` attr-dict `->` type($reg)
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

def HOp : Quantum_Gate<"H", [Unitary, Hermitian, AttrSizedOperandSegments, RegAccessInterface]> {
    let summary = "Hadamard gate.";
    let description = [{
        The Hadamard gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,  1],
             [1, -1]] / sqrt(2)

        This operation takes a qubit as input and returns an 'op' type.
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
        $op = "q.H"() : () -> !q.op
        // OR in custom assembly format
        %op = q.H -> !q.op
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<Op_Type> : $op
    );

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            assert(i == 0 && "There is only 1 QData operand for the HOp!");
            return "static_range";
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_range"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({true}), 1};
        }

        static size_t getSegmentSizesArraySize() {
            return 2;
        }
    }];

    let printer = "return ::print<HOp>(p, *this);";
    let parser = "return ::parseRegAccessOps<HOp>(parser, result);";
}

def XOp : Quantum_Gate<"X", [Unitary, Hermitian, AttrSizedOperandSegments, RegAccessInterface]> {
    let summary = "NOT gate.";
    let description = [{
        The NOT gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[0, 1],
             [1, 0]]

        This operation takes a qubit as input and returns an 'op' type.
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
        $op = "q.X"() : () -> !q.op
        // OR in custom assembly format
        %op = q.X -> !q.op
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<Op_Type> : $op
    );

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            assert(i == 0 && "There is only 1 QData operand for the XOp!");
            return "static_range";
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_range"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({true}), 1};
        }

        static size_t getSegmentSizesArraySize() {
            return 2;
        }
    }];

    let printer = "return ::print<XOp>(p, *this);";
    let parser = "return ::parseRegAccessOps<XOp>(parser, result);";
}

def RzOp : Quantum_Gate<"RZ", [Unitary, AttrSizedOperandSegments, RegAccessInterface]> {
    let summary = "Z-Rotation gate.";
    let description = [{
        The Rz gate is a single Qubit unitary.
        It rotates a qubit state by an angle φ along the z-axis.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,   0 ],
             [0, e^iφ]]

        This operation takes a qubit as input and returns an 'op' type.
        The parameter φ is passed as an attribute.
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
        $op = "q.RZ"() {phi=0.1} : () -> !q.op
        // OR in custom assembly format
        %op = q.RZ(0.1) -> !q.op
        ```
    }];

    let arguments = (ins
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range,
        APFloatAttr : $phi
    );

    let results = (outs
        Optional<Op_Type> : $op
    );

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            assert(i == 0 && "There is only 1 QData operand for the RzOp!");
            return "static_range";
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_range"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({true}), 1};
        }

        static size_t getSegmentSizesArraySize() {
            return 2;
        }
    }];

    let printer = "return ::print<RzOp>(p, *this);";
    let parser = "return ::parseRegAccessOps<RzOp>(parser, result, true);";
}

def CNotOp : Quantum_Gate<"CX", [Unitary, Hermitian, AttrSizedOperandSegments, RegAccessInterface]>{
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
        The control however must be a singular qubit. The return type is a 'cop'.
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
        $op = "q.CX"(%0) : (!q.qubit) -> !q.cop<1>
        // OR in custom assembly format
        $op = q.CX %0 : !q.qubit -> !q.cop<1>
        ```
    }];

    let arguments = (ins
        QData_Type : $ctrl,            // only one qubit allowed, but with RegAccess need QData
        Optional<Index> : $crange,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $qrange,
        I64ArrayAttr : $static_crange,
        I64ArrayAttr : $static_qrange
    );

    let results = (outs
        Optional<ControlledOp_Type> : $op
    );

    let additionalVerifier = [{
        if (this->ctrl().getType().isa<QuregType>() &&
                !this->crange() && this->static_crange().size() != 1)
            return this->emitOpError() << "only accepts single qubit for the 'ctrl' operand, thus "
                                          "if value is a register, must exactly 1 accessor index!";
    }];

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            switch (i) {
                case 0:
                    return "static_crange";
                case 1:
                    return "static_qrange";
                default:
                    assert(false && "There are only 2 QData operands for the CNotOp!"); return "";
            }
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_crange", "static_qrange"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({true, true}), 2};
        }

        static size_t getSegmentSizesArraySize() {
            return 4;
        }
    }];

    let printer = "return ::print<CNotOp>(p, *this);";
    let parser = "return ::parseRegAccessOps<CNotOp>(parser, result);";
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

def CircuitOp : Quantum_Op<"circ", [Unitary, ImplicitQuantumTerminator]> {
    let summary = "Create a basic quantum circuit.";
    let description = [{
        A basic quantum circuit is a block regrouping multiple gates. No control
        flow is allowed within such a block. However, these operations can then be
        repeated, inverted, or controlled as as single unit. Note that contrary to
        ops of type 'Quantum_Gate' (basic gates & meta ops), a basic circuit is
        always considered "on hold" when defined, and thus always returns the op
        type 'circ'. Note that all qubits used within the circuit are final and
        must be defined in an enclosing scope, that is, the circuit cannot be
        reapplied to a different set of qubits. See 'ParametricCircuitOp' for this
        functionality.

        Since the circuit body consists of a block, it must always be terminated
        with the special 'TerminatorOp'. In custom assembly form this can be omitted.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=3} : () -> !q.qureg<3>
        %c = "q.circ"() ({
            "q.H"(%0) : (!q.qureg<3>) -> ()
            "q.bar"() : () -> ()
        }) : () -> !q.circ
        // OR in custom assembly
        %c = q.circ {
            "q.H"(%0) : (!q.qureg<3>) -> ()
        } -> !q.circ
        ```
    }];

    let arguments = (ins
        OptionalAttr<StrAttr> : $name
    );

    let results = (outs
        Circuit_Type
    );

    let regions = (region
        SizedRegion<1> : $gates
    );

    let assemblyFormat = [{
        (`(` $name^ `)`)? regions attr-dict `->` type(results)
    }];
}

def ParametricCircuitOp : Quantum_Op<"parcirc", [AttrSizedOperandSegments, RegAccessInterface,
                                                 DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "Instatiate a parametric quantum circuit.";
    let description = [{
        In contrast to a basic circuit, a parametric circuit is created with the
        help of a function. The function contains the body of the circuit and must
        take a non-negative circuit size parameter and a variadic list of acted-upon
        qubits/regiters as input. This op then takes the function name and arguments
        and instatiates a circuit op that can be further operated on by meta ops or
        used with the application op.

        ```mlir
        func @fun(%n : index, %qbs : !q.qureg<4>) -> () {
            %q = "q.extract"(%qbs) {idx=0} : (!q.qureg<4>) -> !q.qubit
            "q.H"(%q) : (!q.qubit) -> ()
            "q.term"() : () -> ()
        }

        %n = constant 4 : index
        %0 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        %pc = "q.parcirc"(%n, %0) {callee=@fun} : (index, !q.qureg<4>) -> !q.circ
        // OR in custom assembly format
        %pc = q.parcirc @fun(%n, %0) : (index, !q.qureg<4>) -> !q.circ
        ```
    }];

    let arguments = (ins
        Variadic<QData_Type> : $qbs,
        Variadic<Index> : $ranges,
        2dI64ArrayAttr : $static_ranges,
        FlatSymbolRefAttr : $callee,
        Confined<APIntAttr, [IntPositive]> : $n
    );

    let results = (outs
        Circuit_Type
    );

    let verifier = [{
        // tally the number of qubits from input types if available
        unsigned totalqbs = 0;
        bool lowerbound = false;
        for (auto argType : this->qbs().getTypes()) {
            if (argType.isa<QubitType>()) {
                totalqbs++;
            } else if (auto regType = argType.dyn_cast<QuregType>()) {
                if (auto numQubits = regType.getNumQubits())
                    totalqbs += *numQubits;
                else
                    lowerbound = true;
            }
        }

        // if no registers of undeclared size were provided, make sure sufficient qubits are
        // available for the circuit size parameter n
        if (!lowerbound && this->n().ugt(totalqbs))
            return this->emitOpError() << "has insufficient total qubits for size parameter n!";

        // make sure the function exists
        auto funptr = this->resolveCallable();
        if (!funptr)
            return this->emitOpError() << "cannot resolve given function!";

        // make sure function arguments match
        auto typeattr = funptr->getAttrDictionary().get("type").dyn_cast<TypeAttr>();
        auto funtype = typeattr.getValue().dyn_cast<FunctionType>();

        // compare number of arguments while ignoring parameter in function signature
        if (funtype.getNumInputs()-1 != this->qbs().size())
            return this->emitOpError() << "has wrong number of function arguments! "
                                          "Signature requires: " << funtype.getNumInputs()-1 << ". "
                                          "Got: " << this->getNumOperands() << ".";

        // compare each parameter type:
        //  - a !q.qureg must be matched with another !q.qureg, but only if both types have size
        //    information will the given register be checked to be at least as large as required
        //  - for all other types, they must match exactly
        unsigned i = 0;
        auto opargtypes = this->qbs().getTypes().begin();
        for (auto argtype : funtype.getInputs()) {
            if (i++ == 0) {
                // make sure the circuit size parameter is declared in the function
                if (!argtype.isa<IndexType>())
                    return this->emitOpError() << "needs first function argument to be the circuit "
                                                  "size with type 'index'!";
                // for the rest, skip circuit parameter (first in arg list)
                continue;
            }

            auto giventype = *opargtypes++;
            if (auto atype = argtype.dyn_cast<QuregType>()) {
                if (auto gtype = giventype.dyn_cast<QuregType>()) {
                    if (gtype.getNumQubits() && atype.getNumQubits()) {
                        if (*gtype.getNumQubits() < *atype.getNumQubits())
                            return this->emitOpError()
                                << "with insufficient qubits in register argument at position "
                                << i-2 << ", as signature expects: " << *atype.getNumQubits()
                                << ", but got: " << *gtype.getNumQubits() << "!";
                    }
                } else {
                    return this->emitOpError() << "detected wrong argument type at position "
                                               << i-2 << ", expected: " << argtype << ", "
                                               << "but got: " << giventype << "!";
                }
            } else {
                if (argtype != giventype)
                    return this->emitOpError() << "detected wrong argument type at position "
                                               << i-2 << ", expected: " << argtype << ", "
                                               << "but got: " << giventype << "!";
            }
        }

        return success();
    }];

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            assert(i == 0 && "There is only 1 QData operands for the ParametricCircuitOp!");
            return "static_ranges";
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_ranges"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({true}), 1};
        }

        static size_t getSegmentSizesArraySize() {
            return 2;
        }

        Operation *resolveCallable() {
            // If the callable isn't a value, lookup the symbol reference.
            CallInterfaceCallable callable = this->getCallableForCallee();
            if (auto symbolRef = callable.dyn_cast<SymbolRefAttr>())
                return SymbolTable::lookupNearestSymbolFrom(*this, symbolRef);
            return callable.get<Value>().getDefiningOp();
        }
    }];

    let printer = "return ::print(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
}

def ApplyCircOp : Quantum_Op<"apply"> {
    let summary = "Apply circuit to quantum computer.";
    let description = [{
        This operation is solely meant to perform the application of a circuit
        op or ops derived from a circuit op as these are always on hold when
        created and remain that way when passed to meta-ops. It is not compatible
        with other operations on hold as those require target qubits as arguments
        in order to perform application.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.alloc"() : () -> !q.qubit
        %c = "q.circ"() ({
            "q.H"(%0) : (!q.qubit) -> ()
            "q.CX"(%1, %0) : (!q.qubit, !q.qubit) -> ()
            "q.bar"() : () -> ()
        }) : () -> !q.circ
        // Apply circuit 4 times
        affine.for %i = 0 to 4 {
            "q.apply"(%c) : (!q.circ) -> ()
        }
        // OR in custom assembly
        affine.for %i = 0 to 4 {
            q.apply %c : !q.circ
        }
    }];

    let arguments = (ins
        QCirc_Type : $circ
    );

    let verifier = [{
        if (this->circ().getType().isa<COpType>() &&
            (!this->circ().getType().cast<COpType>().getBaseType() ||
             !this->circ().getType().cast<COpType>().getBaseType().isa<CircType>()))
            return this->emitOpError() << "can only be applied to circuit(-derived) types!";

        return success();
    }];

    let assemblyFormat = [{
        operands attr-dict `:` type(operands)
    }];
}

def ControlOp : Quantum_Gate<"c", [Meta, AttrSizedOperandSegments, RegAccessInterface]> {
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
        %op = "q.H"() : () -> !q.op
        // Control H op on qubit %0 and apply to qubit %1
        "q.c"(%op, %0, %1) : (!q.op, !q.qubit, !q.qubit) -> ()
        // Control H op on qubit %0 and apply to all qubits in register %2
        "q.c"(%op, %0, %2) : (!q.op, !q.qubit, !q.qureg<4>) -> ()
        // Control H op on all of register %2 and apply to qubit %1
        "q.c"(%op, %2, %1) : (!q.op, !q.qureg<4>, !q.qubit) -> ()
        // Control H op on qubit %0 but leave application open
        %cop = "q.c"(%op, %1) : (!q.op, !q.qubit) -> !q.cop<1>
        // OR in custom assembly format
        %cop = q.c %op, %1 : !q.op, !q.qubit -> !q.cop<1>
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        QData_Type: $ctrls,
        Variadic<Index> : $crange,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $qrange,
        I64ArrayAttr : $static_crange,
        I64ArrayAttr : $static_qrange
        // Variadic<QData_Type> : $extractrls
    );

    let results = (outs
        Optional<ControlledOp_Type> : $op
    );

    let additionalVerifier = [{
        auto heldOp = this->heldOp().getType();
        // make sure user is not trying to apply a circuit op by specifying a target
        if (heldOp.isa<CircType>() && this->qbs())
            return this->emitOpError() << "does not accept target qubits when controlling circuit!";

        // obtain total number of control qubits from input
        int totalctrls = 0;

        if (heldOp.isa<COpType>())
            totalctrls += heldOp.cast<COpType>().getNumCtrls();

        auto ctrls = this->ctrls().getType();
        if (ctrls.isa<QubitType>()) {
            totalctrls++;
        } else if (auto regType = ctrls.dyn_cast<QuregType>()) {
            if (auto numQubits = regType.getNumQubits())
                totalctrls += *numQubits;
            else
                totalctrls = -1; // disable # of ctrl qubits check when no size is provided
        } else {
            assert(false && "Unknown data-type encountered in ControlOp!");
        }

        // disable # of ctrl qubits check when using register accessors
        if (this->static_crange().size())
            totalctrls = -1;

        //auto xtypes = this->extractrls().getTypes();
        //for (auto xtype : xtypes) {
        //    if (xtype.isa<QubitType>())
        //        totalctrls++;
        //    else if (xtype.isa<QuregType>())
        //        totalctrls += xtype.cast<QuregType>().getNumQubits();
        //}

        // perform several checks on the return type
        if (this->op()) {
            COpType output = this->op().getType().cast<COpType>();
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

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            switch (i) {
                case 0:
                    return "static_crange";
                case 1:
                    return "static_qrange";
                default:
                    assert(false && "There are only 2 QData operands for the ControlOp!");
                    return "";
            }
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_crange", "static_qrange"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({false, true, true}), 2};
        }

        static size_t getSegmentSizesArraySize() {
            return 5;
        }
    }];

    let printer = "return ::print<ControlOp>(p, *this);";
    let parser = "return ::parseRegAccessOps<ControlOp>(parser, result);";
}

def AdjointOp : Quantum_Gate<"adj", [Meta, AttrSizedOperandSegments, RegAccessInterface]> {
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
        %op = "q.H"() : () -> !q.op
        // Invert H op and apply to qubit %0
        "q.adj"(%op, %0) : (!q.op, !q.qubit) -> ()
        // Invert H op apply to all qubits in register %1
        "q.adj"(%op, %1) : (!q.op, !q.qureg<4>) -> ()
        // Invert H op but leave application open
        %opdg = "q.adj"(%op) : (!q.op) -> !q.op
        // OR in custom assembly format
        %opdg = q.adj %op : !q.op -> !q.op
    }];

    let arguments = (ins
        QOp_Type : $heldOp,
        Optional<QData_Type> : $qbs,
        Variadic<Index> : $range,
        I64ArrayAttr : $static_range
    );

    let results = (outs
        Optional<QOp_Type> : $op
    );

    let additionalVerifier = [{
        if (this->op() && this->heldOp().getType() != this->op().getType())
            return this->emitOpError() << "given operation type must match output type!";

        if (this->heldOp().getType().isa<CircType>() && this->qbs())
            return this->emitOpError() << "does not accept target qubits when inverting circuit!";
    }];

    let extraClassDeclaration = [{
        static StringRef getAccessorAttrName(unsigned i) {
            assert(i == 0 && "There is only 1 QData operand for the AdjointOp!");
            return "static_range";
        }

        static SmallVector<StringRef, 2> getAccessorAttrNames() {
            return {"static_range"};
        }

        static std::pair<SmallVector<bool, 4>, unsigned> getRegLikeArray() {
            return {SmallVector<bool, 4>({false, true}), 1};
        }

        static size_t getSegmentSizesArraySize() {
            return 3;
        }
    }];

    let printer = "return ::print<AdjointOp>(p, *this);";
    let parser = "return ::parseRegAccessOps<AdjointOp>(parser, result);";
}

#endif // QUANTUM_OPS
