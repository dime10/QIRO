/* Create a quantum-opt program to roundtrip IR examples using the quant dialect */

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#include "QuantumDialect.h"
#include "QuantumSSADialect.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                                                      llvm::cl::value_desc("filename"),
                                                      llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile("split-input-file",
                                          llvm::cl::desc("Split the input file into pieces and "
                                                         "process each chunk independently"),
                                          llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics("verify-diagnostics",
                                             llvm::cl::desc("Check that diagnostics match expected-"
                                                            "lines on the corresponding line"),
                                             llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses("verify-each",
                                        llvm::cl::desc("Run verifier after each transf. pass"),
                                        llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects("allow-unregistered-dialect",
                                                     llvm::cl::desc("Allow operation with no"
                                                                    "registered dialects"),
                                                     llvm::cl::init(false));

static llvm::cl::opt<bool> showDialects("show-dialects",
                                        llvm::cl::desc("Print the list of registered dialects"),
                                        llvm::cl::init(false));

int main(int argc, char **argv) {
    mlir::registerAllDialects();
    mlir::registerAllPasses();

    mlir::registerDialect<mlir::quantum::QuantumDialect>();
    mlir::registerDialect<mlir::quantumssa::QuantumSSADialect>();
    // TODO: Register quantum passes here.

    llvm::InitLLVM y(argc, argv);

    // Register any pass manager command line options.
    mlir::registerPassManagerCLOptions();
    mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

    // Parse pass names in main to ensure static initialization completed.
    llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR modular optimizer driver\n");

    if (showDialects) {
        mlir::MLIRContext context;
        llvm::outs() << "Registered Dialects:\n";
        for (mlir::Dialect *dialect : context.getRegisteredDialects()) {
            llvm::outs() << dialect->getNamespace() << "\n";
        }
        return 0;
    }

    // Set up the input file.
    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }

    auto output = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!output) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
    }

    if (failed(MlirOptMain(output->os(), std::move(file), passPipeline, splitInputFile,
                           verifyDiagnostics, verifyPasses, allowUnregisteredDialects))) {
        return 1;
    }

    // Keep the output file if the invocation of MlirOptMain was successful.
    output->keep();
    return 0;
}
