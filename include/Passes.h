#ifndef MLIR_QUANTUM_PASSES_H
#define MLIR_QUANTUM_PASSES_H

namespace mlir {
namespace quantum {

std::unique_ptr<Pass> createMemToValPass();

} // end namespace quantum
} // end namespace mlir

#endif // MLIR_QUANTUM_PASSES_H
