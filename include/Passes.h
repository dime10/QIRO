#ifndef MLIR_QUANTUM_PASSES_H
#define MLIR_QUANTUM_PASSES_H

namespace mlir {
namespace quantum {

std::unique_ptr<Pass> createMemToValPass();
std::unique_ptr<Pass> createQuantumGateOptimizationPass();
std::unique_ptr<Pass> createCircuitInlinerPass();
std::unique_ptr<Pass> createResourceCounterPass();
std::unique_ptr<Pass> createStripUnusedCircuitPass();
std::unique_ptr<Pass> createLowerControlledCircuitsPass();

} // end namespace quantum
} // end namespace mlir

#endif // MLIR_QUANTUM_PASSES_H
