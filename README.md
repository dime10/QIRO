This is the quantum optimizing compiler project in MLIR.
The project can be built using the CMake file in this directory, run:
* `cmake -Bbuild -H.`
* `cmake --build build --target quantum-opt`

To manually generate .h.inc and .cpp.inc via TableGen use:
* `mlir-tblgen -gen-op-decls QuantumOps.td -I ../../llvm-project/mlir/include -o QuantumOps.h.inc`
* `mlir-tblgen -gen-op-defs QuantumOps.td -I ../../llvm-project/mlir/include -o QuantumOps.cpp.inc`
