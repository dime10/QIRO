# add library dependencies of the dialect
add_mlir_dialect_library(MLIRQuantum  # library name
    QuantumDialect.cpp                # file for operations directly implemented in C++
    QuantumSSADialect.cpp

    ADDITIONAL_HEADER_DIRS            # specify where to find header filess

    DEPENDS                           # keyword for libraries belonging to this dialect
                                      # (typically in the form of IncGen files)

    LINK_LIBS PUBLIC                  # libs from other dialects are declared using PUBLIC keyword
                                      # LINK_LIBS is used to link to MLIR libraries

    LINK_COMPONENTS                   # LINK_COMPONENTS is used to link to llvm libraries
)
