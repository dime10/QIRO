# register a dialect and its operations (in ODS format), generate documentation
    # args: <Dialect>(must match <Dialect>.td) <namespace>(mlir namespace, usage: <namespace>.<op>)
add_mlir_dialect(QuantOps q)
    # args: <Dialect> <option> <outputfile>(.md) <outputdirectory>
add_mlir_doc(QuantOps -gen-dialect-doc QuantDialect Quant/)
add_mlir_doc(QuantOps -gen-ops-doc QuantOps Quant/)
