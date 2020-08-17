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
            return this->emitError() << "Need to either specify target qubits or return type!";
        if (this->qbs() && this->op())
            return this->emitError() << "Cannot specify both target qubits and return type!";

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

def ExtractOp : Quantum_Op<"extract", [NoSideEffect]> {
    let summary = "Extract single qubit from register.";
    let description = [{
        This op allows to get out individual qubits from a register.
        The register is passed as an argument, and the index as an attribute.
        Also supports extraction from a Qlist, in conjuction with accepting
        the index argument optionally as an ssa operand instead of attribute.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Extract single qubit from register
        %qb = "q.extract"(%0) {idx=2 : index} : (!q.qureg<4>) -> !q.qubit
        // OR in custom assembly format
        %qb = q.extract %0[[2]] : !q.qureg<4> -> !q.qubit
        // Extract qubit from Qlist
        func @foo(%l : !q.qlist) {
            %c = constant 2 : index
            %qb = "q.extract"(%l, %c) : (!q.qureg<4>, index) -> !q.qubit
            // OR in custom assembly format
            %qb = q.extract %l[%c] : !q.qlist, index -> !q.qubit
        }
    }];

    let arguments = (ins
        AnyTypeOf<[Qureg_Type, Qlist_Type]> : $reg,
        OptionalAttr<Confined<IndexAttr, [IntNonNegative]>> : $idxAttr,
        Optional<Index> : $idxArg
    );

    let results = (outs
        Qubit_Type
    );

    let verifier = [{
        if (this->idxAttr() && this->idxArg())
            this->emitError() << "Can supply index as either attribute or operand, but not both!";
        if (!this->idxAttr() && !this->idxArg())
            this->emitError() << "Must supply an index (either via attribute or operand)!";

        if (this->reg().getType().isa<QlistType>()) {
            return success();
        } else {
            if (this->idxArg())
                this->emitError() << "Operand index only supported on Qlist type!";
        }

        llvm::APInt idx = *(this->idxAttr());
        unsigned regsize = this->reg().getType().cast<QuregType>().getNumQubits();

        if (idx.uge(regsize))
            return this->emitError() << "index out of bounds, got "
                                     << "idx=" << *(idx.getRawData())
                                     << ", regsize=" << regsize;

        return success();
    }];

    let assemblyFormat = [{
        $reg `[` (`[`$idxAttr^`]`)? ($idxArg^)? `]` attr-dict `:` type(operands) `->` type(results)
    }];
}

def SliceRegOp : Quantum_Op<"slice", [NoSideEffect]> {
    let summary = "Extract range of qubits from register.";
    let description = [{
        This op allows to extract a continous range of qubits from a register.
        The register is passed as an argument, and the indeces as attributes.
        The endpoint is included for the left index but NOT the right index.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
        // Extract range of qubits from register
        %reg = "q.slice"(%0) {a=1, b=4} : (!q.qureg<4>) -> !q.qureg<3>
        // OR in custom assembly format
        %reg = q.slice %0[1,4] : !q.qureg<4> -> !q.qureg<3>
    }];

    let arguments = (ins
        Qureg_Type : $reg,
        Confined<APIntAttr, [IntNonNegative]> : $a,
        Confined<APIntAttr, [IntNonNegative]> : $b
    );

    let results = (outs
        Qureg_Type : $newreg
    );

    let verifier = [{
        llvm::APInt a = this->a();
        llvm::APInt b = this->b();
        unsigned regsize = this->reg().getType().cast<QuregType>().getNumQubits();
        unsigned newregsize = this->newreg().getType().cast<QuregType>().getNumQubits();

        if (a.uge(regsize) || b.ugt(regsize))
            return this->emitError() << "one or both indeces out of bounds, got "
                                     << "a=" << *(a.getRawData())
                                     << ", b=" << *(b.getRawData())
                                     << ", regsize=" << regsize;
        if ((b-a).slt(2))
            return this->emitError() << "slice must include at least 2 elements, got "
                                     << (long long) *((b-a).getRawData());
        if ((b-a) != newregsize)
            return this->emitError() << "output register has wrong size, got " << newregsize
                                     << ", requires " << *((b-a).getRawData());

        return success();
    }];

    let assemblyFormat = [{
        $reg `[` $a `,` $b `]` attr-dict `:` type(operands) `->` type(results)
    }];
}

def GenRegOp : Quantum_Op<"genreg", [NoSideEffect]> {
    let summary = "Generate a register from multiple qubits/quregs.";
    let description = [{
        This op allows to combine previously allocated qubits/quregs
        into a new register for the purpose of passing it to other operations.
        Note that there is no checks against qubit overlap, that is putting
        the same qubit into a register multiple times. This resposibility
        is left to the code generating the IR, and thus the op should be
        considered "internal".

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> !q.qubit
        %1 = "q.allocreg"() {size=2} : () -> !q.qureg<2>
        // Generate new register from other qubits/registers
        %reg = "q.genreg"(%0, %1) : (!q.qubit, !q.qureg<2>) -> !q.qureg<3>
        // OR in custom assembly format
        %reg = q.genreg %0, %1 : !q.qubit, !q.qureg<2> -> !q.qureg<3>
        // Cast Qlist to fixed-sized Qureg
        func @foo(%l : !q.qlist) {
            %reg = "q.genreg"(%l) : (!q.qlist) -> !q.qureg<7>
            // OR in custom assembly format
            %reg = q.genreg %l : !q.qlist -> !q.qureg<7>
        ```
    }];

    let arguments = (ins
        QDataX_Type : $input1,
        Variadic<QData_Type> : $extrainputs
    );

    let results = (outs
        Qureg_Type : $newreg
    );

    let verifier = [{
        if (this->input1().getType().isa<QlistType>()) {
            if (this->extrainputs().size())
                this->emitError() << "Usage on Qlist reserved for cast to Qureg, "
                                     "thus no additional inputs are allowed!";
            return success();
        }

        // obtain total number of qubits from input
        unsigned totalqbs = 0;

        auto qbtype = this->input1().getType();
        if (qbtype.isa<QubitType>())
            totalqbs++;
        else if (qbtype.isa<QuregType>())
            totalqbs += qbtype.cast<QuregType>().getNumQubits();

        auto xtypes = this->extrainputs().getTypes();
        for (auto xtype : xtypes) {
            if (xtype.isa<QubitType>())
                totalqbs++;
            else if (xtype.isa<QuregType>())
                totalqbs += xtype.cast<QuregType>().getNumQubits();
        }

        // a register needs at least 2 qubits
        if (totalqbs < 2)
            return this->emitError() << "A register is composed of at least 2 qubits!";

        // assert that this matches the output type
        unsigned declsize = this->newreg().getType().cast<QuregType>().getNumQubits();
        if (declsize != totalqbs)
            return this->emitError() << "Number of qubits in the output doesn't match inputs!";

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
        QData_Type : $ctrl,
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
        //if (this->qbs() && this->qbs() == this->ctrl())
        //    return this->emitError() << "Cannot use same qubit for control AND target!";
        if (this->ctrl().getType().isa<QuregType>() && (!this->crange() &&
                                                         this->static_crange().size() != 1))
            return this->emitError() << "The control operand can only be a singular qubit, thus if "
                                        "type is Qureg, must provide a single reg accessor index!";
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

    let printer = "return ::print(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
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
        // obtain total number of qubits from qbs
        unsigned totalqbs = 0;

        auto qbstypes = this->qbs().getTypes();
        for (auto type : qbstypes) {
            if (type.isa<QubitType>())
                totalqbs++;
            else if (type.isa<QuregType>())
                totalqbs += type.cast<QuregType>().getNumQubits();
        }

        if (this->n().ugt(totalqbs))
            this->emitError() << "Supplied insufficient qubits for size parameter n!";

        // make sure the function exists
        auto funptr = this->resolveCallable();
        if (!funptr)
            this->emitError() << "The given function could not be resolved!";

        // make sure function arguments match
        auto typeattr = funptr->getAttrDictionary().get("type").dyn_cast<TypeAttr>();
        auto funtype = typeattr.getValue().dyn_cast<FunctionType>();

        // compare number of arguments while ignoring parameter in function signature
        if (funtype.getNumInputs()-1 != this->qbs().size())
            this->emitError() << "Given number of function arguments does not match!"
                                 " Required: " << funtype.getNumInputs()-1 <<
                                 ". Got: " << this->getNumOperands() << ".";

        // compare each parameter type:
        //  - a !q.qlist must match either !q.qubit or !q.qureg
        //  - for all other types, they must match exactly
        unsigned i = 0;
        auto opargtypes = qbstypes.begin();
        for (auto arg : funtype.getInputs()) {
            if (i++ == 0) {
                // make sure the circuit size parameter is declared in the function
                if (!arg.isa<IndexType>())
                    this->emitError() << "First function argument must be index (circuit size)!";
                // for the rest, skip circuit parameter (first in arg list)
                continue;
            }

            auto giventype = *opargtypes++;
            if (arg.isa<QlistType>()) {
                if (!giventype.isa<QuregType>() && !giventype.isa<QubitType>())
                    this->emitError() << "Function expects '!q.qlist' argument, which must be "
                                         "instantiated with either a '!q.qubit' or '!q.qureg'. "
                                         "Got: " << giventype << "!";
            } else {
                if (arg != giventype)
                    this->emitError() << "Type " << giventype << " at index " << i-2 <<
                                         " did not match function signature type " << arg << "!";
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
            return this->emitError() << "This op can only be applied to circuit(-derived) types!";

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
        // make sure target and control aren't identical (basic test, not comprehensive)
        //if (this->qbs() && this->qbs() == this->ctrls())
        //    return this->emitError() << "Cannot use same qubits for control AND target!";

        auto heldOp = this->heldOp().getType();
        // make sure user is not trying to apply a circuit op by specifying a target
        if (heldOp.isa<CircType>() && this->qbs())
            return this->emitError() << "Circuit cannot be applied by specifying qubits!";

        // obtain total number of control qubits from input
        unsigned totalctrls = 0;

        if (heldOp.isa<COpType>())
            totalctrls += heldOp.cast<COpType>().getNumCtrls();

        auto ctrls = this->ctrls().getType();
        if (ctrls.isa<QubitType>())
            totalctrls++;
        else if (ctrls.isa<QuregType>())
            totalctrls += ctrls.cast<QuregType>().getNumQubits();

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
                return this->emitError() << "Number of output controls doesn't match inputs!";

            // assert proper base type
            if (heldOp.isa<COpType>()) {
                if (heldOp.cast<COpType>().getBaseType() != output.getBaseType())
                    return this->emitError() << "ControlOp must preserve base type when given COp!";
            } else {
                if (heldOp != output.getBaseType())
                    return this->emitError() << "Returned base type doesn't match given operation!";
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
            return this->emitError() << "Input op type must match output op type!";

        if (this->heldOp().getType().isa<CircType>() && this->qbs())
            return this->emitError() << "Circuit cannot be applied by specifying qubits!";
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
