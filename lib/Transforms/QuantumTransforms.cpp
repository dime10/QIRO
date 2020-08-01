#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "QuantumSSADialect.h"
#include "Passes.h"
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <string>

using namespace mlir;
using namespace quantum;


//===------------------------------------------------------------------------------------------===//
// Memory to Value semantics pass
//===------------------------------------------------------------------------------------------===//

// register hash function of mlir:Value in std for use in std::map
namespace std {
    template<> struct hash<mlir::Value> {
        std::size_t operator()(mlir::Value const& val) const noexcept {
            return mlir::hash_value(val);
        }
    };
} // end namespace std

struct MemToValPass : public OperationPass<ModuleOp> {
    MemToValPass() : OperationPass<ModuleOp>(TypeID::get<MemToValPass>()), numCirc(0) {}
    MemToValPass(const MemToValPass &) : OperationPass<ModuleOp>(TypeID::get<MemToValPass>()),
                                         numCirc(0) {}

    StringRef getName() const {
        return "MemToValPass";
    }

    std::unique_ptr<Pass> clonePass() const {
        return std::make_unique<MemToValPass>(*this);
    }

private:
    using value_map = std::unordered_map<mlir::Value, mlir::Value>;
    // keep a record of the latest qubit state to replace qubit values with
    value_map globalStateMap;
    value_map localStateMap;
    // temporary storage for newly created functions
    Operation *newfn;
    // storage for operands of parcirc ops since these need to be move to the call site (applyfc)
    std::unordered_map<mlir::Value, llvm::SmallVector<mlir::Value, 4>> circArgMap;
    // circuit counter to generate unique names
    unsigned numCirc;

    void walk(Operation *op, function_ref<void(Operation*, value_map&)> callback,
                             function_ref<void(Operation*)> cleanup) {
        // function ops need to be called before nested ops are recursed into
        if (isa<FuncOp>(op) || isa<CircuitOp>(op))
            callback(op, localStateMap);

        // walk nested operations
        for (auto &region : op->getRegions())
            for (auto &block : region)
                for (auto &nestedOp : llvm::make_early_inc_range(block))
                    walk(&nestedOp, callback, cleanup);

        // perform postorder call -> most ops as they can only be deleted after recursing
        if (isa<FuncOp>(op) || isa<CircuitOp>(op))
            cleanup(op);
        else if (op->getParentOp() && (isa<FuncOp>(op->getParentOp()) ||
                                       isa<CircuitOp>(op->getParentOp())))
            callback(op, localStateMap);
        else
            callback(op, globalStateMap);
    }

public:
    void runOnOperation() override {
        Operation *module = getOperation().getOperation();
        // the op builder can create new ops and types for us
        OpBuilder opBuilder(module->getContext());

        walk(module,
        [this, &opBuilder] (Operation *op, value_map &qbmap) { // callback
            // nothing to do for modules
            if (isa<ModuleOp>(op) || isa<AllocRegOp>(op))
                return;

            // set up common resources
            op->print(llvm::errs());
            opBuilder.setInsertionPoint(op);
            QstateType qstateType = opBuilder.getType<QstateType>();
            OperationState opState(op->getLoc(), op->getName());
            Operation *newOp = nullptr;

            if (isa<FuncOp>(op)) {
                FuncOp fn = dyn_cast<FuncOp>(op);

                // generate new function argument and return type, replace qubit -> state
                std::vector<Type> inputTypes = fn.getType().getInputs().vec();
                unsigned numQubits = 0;
                for (unsigned i = 0; i < fn.getNumArguments(); i++) {
                    if (inputTypes[i].isa<QubitType>()) {
                        inputTypes[i] = qstateType;
                        numQubits++;
                    }
                }
                SmallVector<Type, 4> resultTypes(numQubits, qstateType);
                FunctionType newType = opBuilder.getFunctionType(inputTypes, resultTypes);

                // create new function object with empty entry block (to be populated later)
                FuncOp::build(opBuilder, opState, fn.getName(), newType, {});
                newOp = newfn = opBuilder.createOperation(opState);
                dyn_cast<FuncOp>(newOp).addEntryBlock();

                // add new block argument values to the local storage
                for (unsigned i = 0; i < dyn_cast<FuncOp>(op).getNumArguments(); i++) {
                    if (dyn_cast<FuncOp>(op).getArgument(i).getType().isa<QubitType>())
                        qbmap[dyn_cast<FuncOp>(op).getArgument(i)] = dyn_cast<FuncOp>(newOp).getArgument(i);
                }
            } else if (isa<TerminatorOp>(op)) {
                SmallVector<Value, 4> states;
                states.reserve(qbmap.size());
                for (auto &pair : qbmap)
                    states.push_back(pair.second);

                opState = OperationState(op->getLoc(), quantum::ReturnOp::getOperationName());
                quantum::ReturnOp::build(opBuilder, opState, ValueRange(states));
                newOp = opBuilder.createOperation(opState);

                op->erase();
            } else if (isa<mlir::CallOp>(op)) {
                CallOp call = dyn_cast<CallOp>(op);
                FlatSymbolRefAttr callee = call.calleeAttr();
                FunctionType calleeType = call.getCalleeType();
                SmallVector<Value, 4> operands;
                SmallVector<Type, 4> resultTypes;
                for (auto &arg : call.getOperands()) {
                    if (arg.getType().isa<QubitType>()) {
                        operands.push_back(qbmap[arg]);
                        resultTypes.push_back(qstateType);
                    } else {
                        operands.push_back(arg);
                    }
                }

                CallOp::build(opBuilder, opState, callee, resultTypes, operands);
                newOp = opBuilder.createOperation(opState);

                for (unsigned i = 0, j = 0; i < call.getNumOperands(); i++) {
                    if (op->getOperand(i).getType().isa<QubitType>())
                        qbmap[op->getOperand(i)] = newOp->getResult(j++);
                }
                op->erase();
            } else if (isa<CircuitOp>(op)) {
                CircuitOp circOp = cast<CircuitOp>(op);
                // collect all external ssa values to generate equivalent function in second step
                std::unordered_set<Value> uniqueOperands;
                for (auto &childOp : circOp.getOps()) {
                    for (auto &arg : childOp.getOperands()) {
                        if (arg.getDefiningOp()->getParentOp() != op)
                            uniqueOperands.insert(arg);
                    }
                }
                SmallVector<Value, 4> externalOperands(uniqueOperands.begin(), uniqueOperands.end());

                // generate new function argument and return types, replace qubit -> state
                SmallVector<Type, 4> inputTypes; inputTypes.reserve(externalOperands.size());
                unsigned numQubits = 0;
                for (auto &arg : externalOperands) {
                    if (arg.getType().isa<QubitType>()) {
                        inputTypes.push_back(qstateType);
                        numQubits++;
                    } else {
                        inputTypes.push_back(arg.getType());
                    }
                }
                SmallVector<Type, 4> resultTypes(numQubits, qstateType);
                FunctionType newType = opBuilder.getFunctionType(inputTypes, resultTypes);

                // create new function object with empty entry block (to be populated later)
                StringRef name = circOp.name() ? circOp.name().getValue()
                                               : *(new std::string("circ" + std::to_string(numCirc++))); // TODO: fix this
                opState = OperationState(op->getLoc(), FuncOp::getOperationName());
                FuncOp::build(opBuilder, opState, name, newType, {});
                newfn = opBuilder.createOperation(opState);
                cast<FuncOp>(newfn).addEntryBlock();

                // add new block argument values to the local storage
                for (unsigned i = 0; i < externalOperands.size(); i++) {
                    if (externalOperands[i].getType().isa<QubitType>())
                        qbmap[externalOperands[i]] = cast<FuncOp>(newfn).getArgument(i);
                }

                // create the function circuit op
                FunCircType retType = opBuilder.getType<FunCircType>(cast<FuncOp>(newfn).getType());
                opState = OperationState(op->getLoc(), FunCircOp::getOperationName());
                FunCircOp::build(opBuilder, opState, retType, name, nullptr);
                newOp = opBuilder.createOperation(opState);

                // add the external operands to the circuit argument cache
                circArgMap[newOp->getResult(0)] = externalOperands;
                op->replaceAllUsesWith(newOp);
            } else if (isa<ParametricCircuitOp>(op)) {
                ParametricCircuitOp parCircOp = cast<ParametricCircuitOp>(op);
                FuncOp fun = dyn_cast<FuncOp>(parCircOp.resolveCallable());
                FunCircType retType = opBuilder.getType<FunCircType>(fun.getType());

                opState = OperationState(op->getLoc(), FunCircOp::getOperationName());
                FunCircOp::build(opBuilder, opState, retType, fun.getName(), parCircOp.nAttr());
                newOp = opBuilder.createOperation(opState);

                circArgMap[newOp->getResult(0)] = op->getOperands();
                op->replaceAllUsesWith(newOp);
                op->erase();
            } else if (isa<ApplyCircOp>(op)) {
                ApplyCircOp applyOp = cast<ApplyCircOp>(op);
                Operation *circDef = applyOp.circ().getDefiningOp();
                while (!isa<FunCircOp>(circDef))
                    circDef = circDef->getOperand(0).getDefiningOp(); // TODO: contractualize heldOp at 0
                FunCircOp fcirc = cast<FunCircOp>(circDef);
                SmallVector<Value, 4> oldOperands = circArgMap[fcirc.funcirc()];
                SmallVector<Value, 4> newOperands;

                if (fcirc.n()) {
                    opState = OperationState(op->getLoc(), ConstantOp::getOperationName());
                    auto n = IntegerAttr::get(opBuilder.getIndexType(), fcirc.nAttr().getValue());
                    ConstantOp::build(opBuilder, opState, n);
                    Operation *constOp = opBuilder.createOperation(opState);
                    newOperands.push_back(constOp->getResult(0));
                }

                for (auto &arg : oldOperands) {
                    if (arg.getType().isa<QubitType>())
                        newOperands.push_back(qbmap[arg]);
                    else
                        newOperands.push_back(arg);
                }

                ArrayRef<Type> retTypes = fcirc.getType().dyn_cast<FunCircType>().getFunType().getResults();
                opState = OperationState(op->getLoc(), ApplyFunCircOp::getOperationName());
                ApplyFunCircOp::build(opBuilder, opState, retTypes, applyOp.circ(), newOperands);
                newOp = opBuilder.createOperation(opState);

                for (unsigned i = 0, j = 0; i < oldOperands.size(); i++) {
                    if (oldOperands[i].getType().isa<QubitType>())
                        qbmap[oldOperands[i]] = newOp->getResult(j++);
                }
                op->erase();
            } else if (isa<quantum::AllocOp>(op)) {
                OpResult qb = op->getOpResult(0);
                if (qb.getType().isa<QubitType>()) {
                    quantum::AllocOp::build(opBuilder, opState, qstateType);
                    newOp = opBuilder.createOperation(opState);

                    qbmap[op->getOpResult(0)] = newOp->getOpResult(0);
                    // can't remove the op yet as it's return value is still needed for qubit map
                }
            } else if ((isa<HOp>(op) || isa<XOp>(op) || isa<RzOp>(op)) && op->getNumOperands()) {
                Value qb = op->getOpOperand(0).get();
                if (qb.getType().isa<QubitType>()) {
                    Value arg = qbmap[qb];

                    if (isa<HOp>(op))
                        HOp::build(opBuilder, opState, qstateType, arg);
                    else if (isa<XOp>(op))
                        XOp::build(opBuilder, opState, qstateType, arg);
                    else {
                        FloatAttr phi = op->getAttrOfType<FloatAttr>("phi");
                        RzOp::build(opBuilder, opState, qstateType, arg, phi);
                    }
                    newOp = opBuilder.createOperation(opState);

                    qbmap[qb] = newOp->getOpResult(0);
                    op->erase();
                }
            } else if (isa<CNotOp>(op) || isa<ControlOp>(op)) {
                Value ctrl = isa<CNotOp>(op) ? dyn_cast<CNotOp>(op).ctrl()
                                             : dyn_cast<ControlOp>(op).ctrls();
                Value qbs = isa<CNotOp>(op) ? dyn_cast<CNotOp>(op).qbs()
                                            : dyn_cast<ControlOp>(op).qbs();
                if (ctrl.getType().isa<QubitType>() && (!qbs || qbs.getType().isa<QubitType>())) {
                    Value arg = qbmap[ctrl];
                    Value arg2 = nullptr;
                    Type retType = qstateType;

                    if (qbs) {
                        arg2 = qbmap[qbs];
                    }else
                        retType = op->getResultTypes().front().dyn_cast<COpType>();

                    if (isa<CNotOp>(op))
                        CNotOp::build(opBuilder, opState, retType, arg2, arg);
                    else {
                        Value heldOp = dyn_cast<ControlOp>(op).heldOp();
                        ControlOp::build(opBuilder, opState, retType, heldOp, arg2, arg);
                    }
                    newOp = opBuilder.createOperation(opState);

                    if (qbs)
                        qbmap[qbs] = newOp->getOpResult(0);
                    else // if return value is an op, do replace all uses of its SSA value
                        op->replaceAllUsesWith(newOp);
                    op->erase();
                }
            } else if (isa<AdjointOp>(op)) {
                Value qbs = dyn_cast<AdjointOp>(op).qbs();
                Value heldOp = dyn_cast<AdjointOp>(op).heldOp();
                if (qbs && qbs.getType().isa<QubitType>()) {
                    Value arg = qbmap[qbs];

                    AdjointOp::build(opBuilder, opState, qstateType, heldOp, arg);
                    newOp = opBuilder.createOperation(opState);

                    qbmap[qbs] = newOp->getOpResult(0);
                    op->erase();
                }
            }
            if (newOp) {
                printf(" -- Building new op...\n  ");
                newOp->dump();
            } else
                printf("\n");
        },
        [this, &opBuilder] (Operation *op) { // cleanup
            if (isa<FuncOp>(op)) {
                // finish up the new function by transfering all operations
                FuncOp fn = cast<FuncOp>(op);
                BlockAndValueMapping valmap;
                valmap.map<Block::BlockArgListType, Block::BlockArgListType>(
                    fn.getArguments(), cast<FuncOp>(newfn).getArguments());

                // the first block needs to be cloned into our existing entry block
                for (auto &op : fn.front()) {
                    cast<FuncOp>(newfn).front().push_back(op.clone(valmap));
                }

                // deal with any remaining blocks
                for (auto it = ++fn.getBlocks().begin(); it != fn.getBlocks().end(); it++) {
                    Block &block = *it;
                    Block *newBlock = new Block();
                    valmap.map(&block, newBlock);
                    for (auto &op : block) {
                        newBlock->push_back(op.clone(valmap));
                    }
                    newfn->getRegion(0).getBlocks().insert(newfn->getRegion(0).end(), newBlock);
                }

                printf("\n -- finalized function:\n");
                newfn->dump();
                op->erase();
                this->localStateMap.clear();
            } else if (isa<CircuitOp>(op)) {
                // finish up the new function by transfering all operations
                CircuitOp circ = cast<CircuitOp>(op);
                BlockAndValueMapping valmap;

                // circuits only have one block, clone it into the entry block of the new function
                for (auto &op : circ.getOps()) {
                    cast<FuncOp>(newfn).front().push_back(op.clone(valmap));
                }

                printf("\n -- finalized function:\n");
                newfn->dump();
                op->erase();
                this->localStateMap.clear();
            }
        });

        // now remove all the alloc ops that have been replaced
        module->walk([](quantum::AllocOp alloc) {
            if (alloc.qb().getType().isa<QubitType>())
                alloc.erase();
        });
    }
};

// Register this pass to make it accessible to utilities like quantum-opt.
static PassRegistration<MemToValPass> pass(
    "convert-mem-to-val",
    "Changes op mode from memory to value semantics, by module."
);

// Create a pass to convert from memory to value semantics
std::unique_ptr<mlir::Pass> mlir::quantum::createMemToValPass() {
  return std::make_unique<MemToValPass>();
}
