# register a dialect and its operations (in ODS format), generate documentation
    # args: <Dialect>(must match <Dialect>.td) <dialect name>(must match name attribute in tblgen)
add_mlir_dialect(QuantumOps q)
    # args: <Dialect> <option> <outputfile>(.md) <outputdirectory>
add_mlir_doc(QuantumOps -gen-dialect-doc QuantumDialect Quantum/)
add_mlir_doc(QuantumOps -gen-op-doc QuantumOps Quantum/)

add_subdirectory(Standalone)
