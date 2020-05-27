get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRQuantum
        )
add_llvm_executable(quantum-opt quantum-opt.cpp)

llvm_update_compile_flags(quantum-opt)
target_link_libraries(quantum-opt PRIVATE ${LIBS})

add_custom_command(TARGET quantum-opt POST_BUILD
    COMMAND quantum-opt ../../test/test.mlir
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin
    COMMENT "Running quantum circuit check..."
)
