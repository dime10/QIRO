get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRExecutionEngine
    MLIRQuantum
    MLIRQuantumTransforms
)
add_llvm_executable(run-jit run-jit.cpp)

llvm_update_compile_flags(run-jit)
target_link_libraries(run-jit PRIVATE ${LIBS})
