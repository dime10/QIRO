## Quantum Program Transformations

The compiler core makes use of various elements of the MLIR transformation infrastructure to express quantum program transformations.

The general pass infrastructure supports passes that run on a particular type of operation and allows arbitrary mutation of all IR nested within this operation.
Typically this sort of pass is run on either the `func` or `module` operation to transform entire functions or compilation units.
The following passes are implemented in this way:

- `MemToValPass` : This pass converts all quantum code within a module from memory semantics to value semantics. It is used as one of the first passes to run on the input to convert the *input dialect* (`Quantum`) to the *optimization dialect* (`QuantumSSA`).

- `QuantumGateOptimizationPass` : This pass uses the greedy pattern rewrite driver to apply the following rewrite patterns to an entire module:
    - `HermitianCancel` : Cancel an identical successive pair of any operation with the `Hermitian` trait.
    - `AdjointCancelBw` : Cancel an applied `adjoint` operation which is preceded by its base operation.
    - `AdjointCancelFw` : Cancel an applied `adjoint` operation which is succeeded by its base operation.
    - `CircuitCancelBw` : Cancel a circuit invocation which is preceded by its inverse.
    - `CircuitCancelFw` : Cancel a circuit invocation which is succeeded by its inverse.
    - `FoldRotation<RzOp>` : Combine two successive `RZ` operations into a single rotation by the sum of the angles.
    - `FoldRotation<ROp>` : Combine two successive `R` operations into a single rotation by the sum of the angles.
    - `FoldControlledRotations` : Combine two successive controlled rotation (`RZ` or `R`) operations into a single controlled rotation by the sum of the angles.

- `StripUnusedCircuitPass` : Remove circuit (i.e. quantum function) definitions which are not invoked in the current module.

- `LowerControlledCircuitsPass` : This is a partial lowering pass for the `control` meta-operation that recursively pushes control modifiers on circuits down into the circuit body. While this does not eliminate control modifiers, it is primarily intended to be used before the resource estimation pass so that resource counts can be deduced.

- `ResourceCounterPass` : This pass is a pseudo lowering pass for the quantum dialect as it enables the "execution" of quantum MLIR programs as a resource counting program instead. It so by removing all quantum operations from the program and replacing them with counter increments instead, leaving the classical program and control flow structure intact. Currently only rotation gates and T gates are counted as the dominant resources for fault-tolerant computing, either directly or via cost formulas for a subset of gates.

The `CircuitInlinerPass` is a slight modification of the built-in MLIR inliner pass adapted to *circuit* operations (i.e. quantum functions).
Inlining quantum functions greatly increases the number of optimization opportunities available to other passes.

The canonicalization infrastructure enables any operation to register custom optimization or "normalization" patterns that will automatically be run during the `-canonicalize` pass and other situations without having to worry about managing passes.
The following quantum operations define optimizations via canonicalization patterns:

- `AdjointOp`
    - `FoldDanglingAdjoint` : Remove instances of the `adjoint` meta-operation whose result is not used.

- `ControlOp`
    - `FoldDanglingControl` : Remove instances of the `control` meta-operation whose result is not used.

- `ApplyCircOp`
    - `FoldApply` : Convert indirect quantum function calls via SSA values (i.e. instances of the `apply` operation) to direct function calls via symbols.

- `CircuitValueOp`
    - `FoldCircVal` : Remove instances of the `getval` operation (generating function pointers for circuits) whose result is not used.

- `ExtractOp`
    - `ExtractExtractPatt` : Combine successive qubit extractions from the same register into a single `extract` operation.
    - `CombineExtractPatt` : Cancel successive qubit insertions and re-extractions from the same position of a register.

- `CombineStatOp`
    - `CombineCombinePatt` : Combine successive qubit insertions into the same register into a single `combine` operation.
    - `CombineExtractCombinePatt` : Delay qubit insertions into a register until the next insertion, provided a matched intermediate extraction from that register has no qubit overlap with the insertion. This pattern is generally matched first, enabling the other patterns for `extract`/`combine` operations.
    - `ExtractCombinePatt` : Cancel successive qubit insertions and re-extractions from the same position of a register.

The purpose of the last two set of patterns is to expand regions of locally available qubit dataflow, following which use-def chains on qubit state values can be used by quantum optimization patters.

The operation folding infrastructure is not directly used.
