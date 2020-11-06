#ifndef MLIR_QUANTUM_PASSES_H
#define MLIR_QUANTUM_PASSES_H

namespace mlir {
namespace quantum {

std::unique_ptr<Pass> createMemToValPass();
std::unique_ptr<Pass> createQuantumGateOptimizationPass();
std::unique_ptr<Pass> createCircuitInlinerPass();

} // end namespace quantum
} // end namespace mlir

#endif // MLIR_QUANTUM_PASSES_H
