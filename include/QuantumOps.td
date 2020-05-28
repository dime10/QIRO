#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"

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

    let arguments = (ins APIntAttr:$size);
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
        ```
    }];

    let arguments = (ins Quantum_Type:$qbs);
}

#endif // QUANTUM_OPS
