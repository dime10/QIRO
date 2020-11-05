/* Create a quantum-opt program to roundtrip IR examples using the quant dialect */

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "QuantumDialect.h"
#include "QuantumSSADialect.h"
#include "Passes.h"

int main(int argc, char **argv) {
    mlir::registerAllDialects();
    mlir::registerAllPasses();

    // Register quantum passes here.
    mlir::registerPass("convert-mem-to-val",
                       "Changes op mode from memory to value semantics, by module.",
                       mlir::quantum::createMemToValPass);
    mlir::registerPass("quantum-gate-opt",
                       "Run the greddy driver on a variety of patterns to optimize quantum gates.",
                       mlir::quantum::createQuantumGateOptimizationPass);

    mlir::DialectRegistry registry;
    // Below we selectively register all dialects that might show up in the input file.
    // If blanket registration of all dialects is prefered, use this statement instead:
    // `registerAllDialects(registry);`
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::AffineDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::quantum::QuantumDialect>();
    registry.insert<mlir::quantumssa::QuantumSSADialect>();

    return failed(mlir::MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
