# add library dependencies of the dialect
add_mlir_dialect_library(MLIRQuantumTransforms
    QuantumTransforms.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

    DEPENDS
    MLIRQuantumOpsIncGen
    MLIRQuantumSSAOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
