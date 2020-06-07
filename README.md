# This is the quantum optimizing compiler project in MLIR.
The project can be built using CMake.
First, make sure to download and build llvm (at least the mlir part).
To do so on Windows, follow the steps below:
* Get the Build Tools for Visual Studio 2019 (https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16). Select the C++ build tools, Windows 10 SDK, and CMake tools.
* Get the llvm requirements: Python (>=2.7), WinGnu32 (http://gnuwin32.sourceforge.net/). Make sure they are added to your PATH variable.
* Download the llvm source: `git clone https://github.com/llvm/llvm-project.git`
* Execute the below commands in a Developer Prompt for VS 2019 or execute `vcvarsal.bat x64` beforehand:
* `cd llvm-project`
* `cmake -Bbuild -Hllvm -G "Visual Studio 16 2019" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Debug -Thost=x64 -DLLVM_ENABLE_ASSERTIONS=ON`
* `cmake --build build --target check-mlir`

Then build the project by running:
* `cd QCompile`
* `cmake -Bbuild -H.`
* `cmake --build build --target quantum-opt`

To manually generate .h.inc and .cpp.inc files via TableGen use:
* `mlir-tblgen -gen-dialect-decls QuantumOps.td -I ../../llvm-project/mlir/include -o QuantumOpsDialect.h.inc`
* `mlir-tblgen -gen-op-decls QuantumOps.td -I ../../llvm-project/mlir/include -o QuantumOps.h.inc`
* `mlir-tblgen -gen-op-defs QuantumOps.td -I ../../llvm-project/mlir/include -o QuantumOps.cpp.inc`

NOTE: The project assumes the following directory structure, be sure to adjust the mlir build path in the top level CMakeLists.txt file, as well as for the above commands, accordingly.
```
ProjectFolder
|-> QCompile
|   |-> include
|   |-> lib
|   |   |-> IR
|   |-> quantum-opt
|   |-> test
|-> llvm-project
|   |-> ...
|   |-> mlir
|   |-> ...
```

&nbsp;

## Have a look at the high-level design decisions here:
[DesignDoc](DesignDoc.md)

&nbsp;

## Technical overview of the different modules used in this project:
![](Hierarchy.png)
