#ifndef QUANT_OPS
#define QUANT_OPS

include "QuantDialect.td"

// example traits
include "mlir/Interfaces/SideEffects.td"

def AllocOp : Quant_Op<"allocate", [NoSideEffect]> {
    let summary = "This allocates a qubit.";
    let description = [{
        The allocation command register a new qubit with the system, which is guaranteed to
        be in the |0> state.

        The NoSideEffect trait is given as example on how to add properties to an operation.

        This operation takes no inputs and returns a value of type qubit.

        Example:

        ```mlir
        %0 = "q.allocate"() : qubit
        ```
    }];

    let results = (outs Quant_QubitType:$qb);
}

def HOp : Quant_Op<"H"> {
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
        %0 = "q.allocate"() : qubit
        // Apply the H operation to %0
        "q.H"(%0) : none
        ```
    }];

    let arguments = (ins Quant_QubitType:$qb);

    // custom MLIR assembly format if desired
    /*
    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
    */
}

#endif // QUANT_OPS
