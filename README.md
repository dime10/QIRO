This is the quantum optimizing compiler project in MLIR.
The project can be built using the CMake file in this directory:
  run `cmake -Bbuild -H.`

To manually generate .h.inc and .cpp.inc via TableGen use:
mlir-tblgen -gen-op-decls QuantOps.td -I ../../llvm-project/mlir/include -o QuantOps.h.inc
mlir-tblgen -gen-op-defs QuantOps.td -I ../../llvm-project/mlir/include -o QuantOps.cpp.inc
