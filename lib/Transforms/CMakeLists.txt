# add library dependencies of the dialect
add_mlir_dialect_library(MLIRQuantumTransforms
    QuantumTransforms.cpp

    ADDITIONAL_HEADER_DIRS

    DEPENDS

    LINK_LIBS PUBLIC
)
