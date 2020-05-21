get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRQuant
        )
add_llvm_executable(quant-opt quant-opt.cpp)

llvm_update_compile_flags(quant-opt)
target_link_libraries(quant-opt PRIVATE ${LIBS})
