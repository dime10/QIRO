## Execution

This build target enables the execution of MLIR programs.

While the *quantum-opt* tool can be used to construct and test arbitrary pass pipelines, it is restricted to the MLIR representation in its input and output.
The *run-jit* command on the other hand provides a full execution pipeline for MLIR programs via the LLVM JIT engine.
Any stage of the pipeline can be chosen as the output via `run-jit -emit=<stage>`, which include (in order):

- `mlir-quant` : output the MLIR dump after lowering to QuantumSSA (the optimization dialect)
- `mlir-scf` : output the MLIR dump after lowering to SCF (removes all quantum code via resource estimation)
- `mlir-std` : output the MLIR dump after lowering to STD
- `mlir-llvm` : output the MLIR dump after lowering to LLVM
- `llvm` : output the LLVM IR dump
- `jit` : JIT the code and run it by invoking `main` from the input program

Additionally, optimization passes can be toggled via the following flags:

- `-opt` : enable level 3 optimizations within the JIT engine
- `-lower` : propagate control modifiers on circuits into the function body
- `-inline` : enable quantum circuit inlining for higher optimization impact
- `-strip` : remove unused circuit definitions
- `-qopt` : enable quantum optimizations

### Printing

A small print library is included under [lib](./lib/) to enable printing from within MLIR programs via the `vector.print` operation.
