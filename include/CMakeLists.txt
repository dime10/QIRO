# register a dialect and its operations (in ODS format), generate documentation
    # args: <Dialect>(must match <Dialect>.td) <namespace>(mlir namespace, usage: <namespace>.<op>)
add_mlir_dialect(QuantumOps q)
    # args: <Dialect> <option> <outputfile>(.md) <outputdirectory>
add_mlir_doc(QuantumOps -gen-dialect-doc QuantumDialect Quantum/)
add_mlir_doc(QuantumOps -gen-ops-doc QuantumOps Quantum/)
