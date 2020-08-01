# register a dialect and its operations (in ODS format), generate documentation
    # args: <Dialect>(must match <Dialect>.td) <dialect name>(must match name attribute in tblgen)
    # args: <Dialect> <option> <outputfile>(.md) <outputdirectory>

add_mlir_dialect(QuantumSSAOps q)
add_mlir_doc(QuantumSSAOps -gen-dialect-doc QuantumDialect Quantum/)
add_mlir_doc(QuantumSSAOps -gen-op-doc QuantumSSAOps Quantum/)
