#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AllocOp : Quantum_Op<"alloc"> {
    let summary = "This allocates a qubit.";
    let description = [{
        The allocation command registers a new qubit with the system, which is guaranteed to
        be in the |0> state.

        This operation takes no inputs and returns a value of type qubit.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        ```
    }];

    let results = (outs Quantum_QubitType:$qb);
}

def AllocRegOp : Quantum_Op<"allocreg"> {
    let summary = "This allocates a qubit register.";
    let description = [{
        The allocation command registers a new qubit register with the system, which is guaranteed
        to be in the |0..> state.

        This operation takes an integer attribute as input specifying the size,
        and returns a value of type qureg<size>.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        ```
    }];

    let arguments = (ins Confined<APIntAttr, [IntPositive]>:$size);
    let results = (outs Quantum_QuregType:$reg);
}

def ExtrRegOp : Quantum_Op<"extract", [NoSideEffect]> {
    let summary = "Extract single qubit from register.";
    let description = [{
        This op allows to get out individual qubits from a register.
        The register is passed as an argument, and the index as an attribute.

        Supports custom assembly.
        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        %1 = "q.extract"(%0) {idx=2} : (!q.qureg<4>) -> (!q.qubit)
        // OR in custom assembly format
        %2 = q.extract %0[2] : !q.qureg<4> -> !q.qubit
    }];

    let arguments = (ins Quantum_QuregType:$reg,
                         Confined<APIntAttr, [IntNonNegative]>:$idx);
    let results = (outs Quantum_QubitType:$qb);

    let verifier = [{
        llvm::APInt idx = this->idx();
        unsigned regsize = this->reg().getType().cast<QuregType>().getNumQubits();

        if (idx.uge(regsize))
            return this->emitError() << "index out of bounds, got "
                                     << "idx=" << *(idx.getRawData())
                                     << ", regsize=" << regsize;

        return success();
    }];

    let assemblyFormat = [{
        $reg `[` $idx `]` attr-dict `:` type($reg) `->` type($qb)
    }];
}

def SliceRegOp : Quantum_Op<"slice", [NoSideEffect]> {
    let summary = "Extract range of qubits from register.";
    let description = [{
        This op allows to extract a continous range of qubits from a register.
        The register is passed as an argument, and the indeces as attributes.
        Endpoint is included for left index but NOT the right index.

        Supports custom assembly.
        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        %1 = "q.extract"(%0) {a=1, b=4} : (!q.qureg<4>) -> (!q.qureg<3>)
        // OR in custom assembly format
        %2 = q.extract %0[1,4] : !q.qureg<4> -> !q.qureg<3>
    }];

    let arguments = (ins Quantum_QuregType:$reg,
                         Confined<APIntAttr, [IntNonNegative]>:$a,
                         Confined<APIntAttr, [IntNonNegative]>:$b);
    let results = (outs Quantum_QuregType:$newreg);

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
        $reg `[` $a `,` $b `]` attr-dict `:` type($reg) `->` type($newreg)
    }];
}

def GenRegOp : Quantum_Op<"genreg", [NoSideEffect]> {
    let summary = "Generate a register from multiple qubits/quregs.";
    let description = [{
        This op allows to combine previously allocated qubits/quregs
        into a new register for the purpose of passing it to other operations.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        %1 = "q.allocreg"() {size=2} : () -> (!q.qureg<2>)
        %2 = "q.genreg"(%0, %1) : (!q.qubit, !q.qureg<2>) -> (!q.qureg<3>)
        ```
    }];

    let arguments = (ins Variadic<Quantum_Type>:$inputs);
    let results = (outs Quantum_QuregType:$reg);
}

def HOp : Quantum_Op<"H"> {
    let summary = "This is the Hadamard gate.";
    let description = [{
        The Hadamard gate is a single Qubit unitary.
        Due to its hermitian property it is its own inverse. 
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,  1],
             [1, -1]] / sqrt(2)

        This operation takes a qubit as input and returns nothing.
        Thus, it only acts via side-effects by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        %1 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        // Apply a Hadamard gate to qubit %0
        "q.H"(%0) : (!q.qubit) -> ()
        // Apply a Hadamard gate to each qubit in register %1
        "q.H"(%1) : (!q.qureg<4>) -> ()
        // Apply a Hadamard gate to each qubit in register %1 controlled upon the state of qubit %0
        "q.H"(%1) {ctrls=%0} : (!q.qureg<4>) -> ()
        ```
    }];

    let arguments = (ins Quantum_Type:$qbs,
                         Optional<Quantum_Type>:$ctrls);
}

def Barrier : Quantum_Op<"bar", [Terminator]> {
    let summary = "This marks the end of a circuit.";
    let description = [{
        As a circuit is described by a block,
        it needs to be terminated by an appropriate op.
    }];
}

def Circuit : Quantum_Op<"circ"> {
    let summary = "This is a basic quantum circuit.";
    let description = [{
        A basic quantum circuit is a simple block regrouping multiple gates.
        These operations can then be repeated or controlled as as single unit.

        The circuit input consists of an optional argument to specify a qubit
        or register upon which the entire circuit should be controlled.
        Additionally, an optional attribute specifies how many times the
        circuit should be repeated (>= 2).
        Note that the gates are grouped within a block, which must be terminated
        with the special Barrier operation.

        Example:

        ```mlir
        %ctrls = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        "q.circ"(%ctrls) ({
        ^bb0:
            %qbs = "q.allocreg"() {size=3} : () -> (!q.qureg<3>)
            "q.H"(%qbs) : (!q.qureg<3>) -> ()
            "q.bar"() : () -> ()
        }) {repeat=6} : (!q.qureg<4>) -> ()
        ```
    }];

    // todo: multiple optional arguments?
    let arguments = (ins Optional<Quantum_Type>:$ctrls,
                         OptionalAttr<Confined<APIntAttr, [IntMinValue<2>]>>:$repeat);
    let regions = (region SizedRegion<1>:$gates);
}

#endif // QUANTUM_OPS
