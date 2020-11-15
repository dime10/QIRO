/* Create a compiler program to run MLIR in a JIT */
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "QuantumDialect.h"
#include "QuantumSSADialect.h"
#include "Passes.h"

namespace {
enum Action {
  None,
  DumpMLIRQuant,
  DumpMLIRSCF,
  DumpMLIRSTD,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
}

static llvm::cl::opt<std::string> mlirSource(llvm::cl::Positional,
                                             llvm::cl::desc("<input file>"),
                                             llvm::cl::init("-"),
                                             llvm::cl::value_desc("filename"));

static llvm::cl::opt<enum Action> emitAction("emit",
    llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(DumpMLIRQuant, "mlir-quant", "output the MLIR dump after lowering QuantumSSA")),
    llvm::cl::values(clEnumValN(DumpMLIRSCF, "mlir-scf", "output the MLIR dump after lowering to SCF")),
    llvm::cl::values(clEnumValN(DumpMLIRSTD, "mlir-std", "output the MLIR dump after lowering to std")),
    llvm::cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump after lowering to LLVM")),
    llvm::cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    llvm::cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoking the main function")));

static llvm::cl::opt<bool> enableOpt("opt", llvm::cl::desc("Enable optimizations"));
static llvm::cl::opt<bool> lowerControls("lower", llvm::cl::desc("Lower controlled circuit calls"));
static llvm::cl::opt<bool> enableInline("inline", llvm::cl::desc("Enable quantum circuit inlining"));
static llvm::cl::opt<bool> stripCircuit("strip", llvm::cl::desc("Remove unused circuit definitions"));
static llvm::cl::opt<bool> enableQOpt("qopt", llvm::cl::desc("Enable quantum optimizations"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(mlirSource);
    if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << mlirSource << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    if (int error = loadMLIR(context, module))
        return error;

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    if (emitAction >= Action::DumpMLIRQuant)
        pm.addPass(mlir::quantum::createMemToValPass());
    if (lowerControls)
        pm.addPass(mlir::quantum::createLowerControlledCircuitsPass());
    if (stripCircuit) {
        pm.addPass(mlir::quantum::createStripUnusedCircuitPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::quantum::createStripUnusedCircuitPass());
    }
    if (enableInline)
        pm.addPass(mlir::quantum::createCircuitInlinerPass());
    if (stripCircuit) {
        pm.addPass(mlir::quantum::createStripUnusedCircuitPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::quantum::createStripUnusedCircuitPass());
    } else {
        pm.addPass(mlir::createCanonicalizerPass());
    }
    if (enableQOpt) {
        pm.addPass(mlir::quantum::createQuantumGateOptimizationPass());
        pm.addPass(mlir::createCanonicalizerPass());
    }
    if (emitAction >= Action::DumpMLIRSCF)
        pm.addPass(mlir::quantum::createResourceCounterPass());
    if (emitAction >= Action::DumpMLIRSTD)
        pm.addPass(mlir::createLowerToCFGPass());
    if (emitAction >= Action::DumpMLIRLLVM) {
        pm.addPass(mlir::createConvertVectorToLLVMPass());
        pm.addPass(mlir::createLowerToLLVMPass());
    }

    if (mlir::failed(pm.run(*module)))
        return 4;
    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    // Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }
    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int runJit(mlir::ModuleOp module) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles the module.
    auto maybeEngine = mlir::ExecutionEngine::create(
        module, optPipeline, llvm::None, {"/home/dave/QCompile/run-jit/lib/printlib.so"});
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invoke("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}


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
    mlir::registerPass("circuit-inline",
                       "Inline circuit calls",
                       mlir::quantum::createCircuitInlinerPass);
    mlir::registerPass("count-resources",
                       "Count the quantum resources used in this program.",
                       mlir::quantum::createResourceCounterPass);

    // Below we selectively register all dialects that might show up in the input file.
    // If blanket registration of all dialects is prefered, use this statement instead:
    // `registerAllDialects(registry);`
    mlir::MLIRContext context(/*loadAllDialects=*/false);
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    context.getOrLoadDialect<mlir::quantum::QuantumDialect>();
    context.getOrLoadDialect<mlir::quantumssa::QuantumSSADialect>();

    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    llvm::cl::ParseCommandLineOptions(argc, argv, "Quantum compiler\n");

    mlir::OwningModuleRef module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;

    // If we aren't exporting to non-mlir, then we are done.
    if (emitAction <= Action::DumpMLIRLLVM) {
        module->dump();
        return 0;
    }

    // Check to see if we are compiling to LLVM IR.
    if (emitAction == Action::DumpLLVMIR)
        return dumpLLVMIR(*module);

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
        return runJit(*module);

    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
