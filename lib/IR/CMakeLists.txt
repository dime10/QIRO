# add library dependencies of the dialect
add_mlir_dialect_library(MLIRQuantum  # library name
    QuantumDialect.cpp                # file for operations directly implemented in C++
    QuantumSSADialect.cpp

    ADDITIONAL_HEADER_DIRS            # specify where to find header files
    ${PROJECT_SOURCE_DIR}/include

    DEPENDS                           # keyword for libraries belonging to this dialect
    MLIRQuantumOpsIncGen              # (typically in the form of IncGen files)
    MLIRQuantumSSAOpsIncGen

    LINK_LIBS PUBLIC                  # libs from other dialects are declared using PUBLIC keyword
    MLIRIR                            # LINK_LIBS is used to link to MLIR libraries

    #LINK_COMPONENTS                  # LINK_COMPONENTS is used to link to llvm libraries
)
