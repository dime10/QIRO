#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "QuantumDialect.h"
#include "QuantumSSADialect.h"
#include "Passes.h"

#include <unordered_map>
#include <unordered_set>
#include <string>

using namespace mlir;
using namespace mlir::quantumssa;


//===------------------------------------------------------------------------------------------===//
// Utility functions
//===------------------------------------------------------------------------------------------===//

bool isQData(Type ty) {
    return ty.isa<quantum::QubitType>() || ty.isa<quantum::QuregType>();
}


//===------------------------------------------------------------------------------------------===//
// Memory to Value semantics pass
//===------------------------------------------------------------------------------------------===//

// register hash function of mlir:Value in std for use with maps and sets
namespace std {
    template<> struct hash<Value> {
        size_t operator()(Value const& val) const noexcept {
            return hash_value(val);
        }
    };
} // end namespace std

namespace {
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
    using value_map = std::unordered_map<Value, Value>;
    // keep a record of the latest qubit state to replace qubit values with
    value_map globalStateMap;
    value_map localStateMap;
    // temporary storage for newly created functions
    Operation *newfn;
    // storage for operands of parcirc ops since these need to be move to the call site (applyfc)
    std::unordered_map<Value, llvm::SmallVector<Value, 4>> circArgMap;
    // circuit counter to generate unique names
    unsigned numCirc;

    void walk(Operation *op, function_ref<void(Operation*, value_map&)> callback,
                             function_ref<void(Operation*)> cleanup) {
        // function ops need to be called before nested ops are recursed into
        if (isa<FuncOp>(op) || isa<quantum::CircuitOp>(op))
            callback(op, localStateMap);

        // walk nested operations
        for (auto &region : op->getRegions())
            for (auto &block : region)
                for (auto &nestedOp : llvm::make_early_inc_range(block))
                    walk(&nestedOp, callback, cleanup);

        // perform postorder call -> most ops as they can only be deleted after recursing
        if (isa<FuncOp>(op) || isa<quantum::CircuitOp>(op))
            cleanup(op);
        else if (op->getParentOp() && (isa<FuncOp>(op->getParentOp()) ||
                                       isa<quantum::CircuitOp>(op->getParentOp())))
            callback(op, localStateMap);
        else
            callback(op, globalStateMap);
    }

    static Type convDialectType(Type inType, Builder &builder, const Value &operand = nullptr) {
        Type outType;
        if (inType.isa<quantum::U1Type>()) {
            outType = builder.getType<U1Type>();
        } else if (inType.isa<quantum::U2Type>()) {
            outType = builder.getType<U2Type>();
        } else if (auto copType = inType.dyn_cast<quantum::COpType>()) {
            llvm::Optional<int> n = copType.getNumCtrls();
            Type baseType = copType.getBaseType();
            baseType = baseType ? convDialectType(baseType, builder, operand) : baseType;
            outType = builder.getType<COpType>(n, baseType);
        } else if (inType.isa<quantum::CircType>()) {
            assert(operand && "Invalid operand during type conversion of 'CircType'!");
            outType = operand.getType();
        } else if (inType.isa<quantum::QubitType>()) {
            outType = builder.getType<QstateType>();
        } else if (auto regType = inType.dyn_cast<quantum::QuregType>()) {
            outType = builder.getType<RstateType>(regType.getNumQubits());
        } else {
            assert(false && "Unrecognized dialect type encountered during conversion!");
        }
        return outType;
    }

    static void parseBuildParams(Operation *op, value_map &qbmap, OpBuilder &builder, Value &phi,
                                 Value &ctrl, Value &ctrlState, Value &trgt, Value &trgtState,
                                 Value &heldOp, Type &retType, OperationState &opState) {
        phi = heldOp = ctrl = ctrlState = trgt = trgtState = nullptr;
        if (isa<quantum::HOp>(op)) {
            trgt = cast<quantum::HOp>(op).qbs();
            opState = OperationState(op->getLoc(), HOp::getOperationName());
        } else if (isa<quantum::XOp>(op)) {
            trgt = cast<quantum::XOp>(op).qbs();
            opState = OperationState(op->getLoc(), XOp::getOperationName());
        } else if (isa<quantum::RzOp>(op)) {
            if (auto phiattr = cast<quantum::RzOp>(op).static_phiAttr()) {
                OperationState auxState(op->getLoc(), ConstantOp::getOperationName());
                ConstantOp::build(builder, auxState, builder.getF64Type(), phiattr);
                phi = builder.createOperation(auxState)->getResult(0);
            } else {
                phi = cast<quantum::RzOp>(op).phi();
            }
            trgt = cast<quantum::RzOp>(op).qbs();
            opState = OperationState(op->getLoc(), RzOp::getOperationName());
        } else if (isa<quantum::ROp>(op)) {
            if (auto phiattr = cast<quantum::ROp>(op).static_phiAttr()) {
                OperationState auxState(op->getLoc(), ConstantOp::getOperationName());
                ConstantOp::build(builder, auxState, builder.getF64Type(), phiattr);
                phi = builder.createOperation(auxState)->getResult(0);
            } else {
                phi = cast<quantum::ROp>(op).phi();
            }
            trgt = cast<quantum::ROp>(op).qbs();
            opState = OperationState(op->getLoc(), ROp::getOperationName());
        } else if (isa<quantum::CNotOp>(op)) {
            ctrl = cast<quantum::CNotOp>(op).ctrl();
            trgt = cast<quantum::CNotOp>(op).qbs();
            opState = OperationState(op->getLoc(), CNotOp::getOperationName());
        } else if (isa<quantum::ControlOp>(op)) {
            heldOp = cast<quantum::ControlOp>(op).heldOp();
            ctrl = cast<quantum::ControlOp>(op).ctrls();
            trgt = cast<quantum::ControlOp>(op).qbs();
            opState = OperationState(op->getLoc(), ControlOp::getOperationName());
        } else if (isa<quantum::AdjointOp>(op)) {
            heldOp = cast<quantum::AdjointOp>(op).heldOp();
            trgt = cast<quantum::AdjointOp>(op).qbs();
            opState = OperationState(op->getLoc(), AdjointOp::getOperationName());
        } else {
            return;
        }
        if (trgt) trgtState = qbmap[trgt];
        if (ctrl) ctrlState = qbmap[ctrl];
        if (trgt) retType = trgtState.getType();
        else      retType = convDialectType(op->getResult(0).getType(), builder, heldOp);
    }

public:
    void runOnOperation() override {
        Operation *module = getOperation().getOperation();
        // the op builder can create new ops and types for us
        OpBuilder opBuilder(module->getContext());

        walk(module,
        [this, &opBuilder] (Operation *op, value_map &qbmap) { // callback
            // nothing to do for modules
            if (isa<ModuleOp>(op))
                return;

            // set up common resources
            op->print(llvm::errs());
            opBuilder.setInsertionPoint(op);
            OperationState opState(op->getLoc(), op->getName());
            Value phi, heldOp, ctrl, ctrlState, trgt, trgtState; Type retType;
            Operation *newOp = nullptr;

            if (isa<quantum::AllocOp>(op) || isa<quantum::AllocRegOp>(op)) {
                if (isa<quantum::AllocOp>(op)) {
                    opState = OperationState(op->getLoc(), AllocQbOp::getOperationName());
                    AllocQbOp::build(opBuilder, opState, opBuilder.getType<QstateType>());
                } else {
                    quantum::AllocRegOp regOp = cast<quantum::AllocRegOp>(op);
                    IntegerAttr staticSize = regOp.static_sizeAttr();
                    Type regType = convDialectType(regOp.reg().getType(), opBuilder);
                    opState = OperationState(op->getLoc(), AllocRegOp::getOperationName());
                    AllocRegOp::build(opBuilder, opState, regType, regOp.size(), staticSize);
                }
                newOp = opBuilder.createOperation(opState);

                qbmap[op->getOpResult(0)] = newOp->getOpResult(0);
                // can't remove the op yet as it's return value is still needed for the qubit map

            } else if (isa<quantum::HOp>(op) || isa<quantum::XOp>(op) || isa<quantum::RzOp>(op) ||
                       isa<quantum::ROp>(op) || isa<quantum::CNotOp>(op) ||
                       isa<quantum::ControlOp>(op) || isa<quantum::AdjointOp>(op)) {
                // Regroup all quantum "gate" operations here as they have a very similar pattern
                parseBuildParams(op, qbmap, opBuilder,
                                 phi, ctrl, ctrlState, trgt, trgtState, heldOp, retType, opState);

                if (isa<quantum::HOp>(op))
                    HOp::build(opBuilder, opState, retType, trgtState);
                else if (isa<quantum::XOp>(op))
                    XOp::build(opBuilder, opState, retType, trgtState);
                else if (isa<quantum::RzOp>(op))
                    RzOp::build(opBuilder, opState, retType, phi, trgtState);
                else if (isa<quantum::ROp>(op))
                    ROp::build(opBuilder, opState, retType, phi, trgtState);
                else if (isa<quantum::CNotOp>(op))
                    CNotOp::build(opBuilder, opState, retType, ctrlState, trgtState);
                else if (isa<quantum::ControlOp>(op))
                    ControlOp::build(opBuilder, opState, retType, heldOp, ctrlState, trgtState);
                else if (isa<quantum::AdjointOp>(op))
                    AdjointOp::build(opBuilder, opState, retType, heldOp, trgtState);
                newOp = opBuilder.createOperation(opState);

                // cleanup
                if (trgt)
                    qbmap[trgt] = newOp->getOpResult(0);
                else // if return value is an op, do replace all uses of its SSA value
                    op->replaceAllUsesWith(newOp);
                op->erase();

            } else if (isa<FuncOp>(op)) {
                FuncOp fn = cast<FuncOp>(op);

                // generate new function argument and return type, replace qubit -> state
                std::vector<Type> inputTypes = fn.getType().getInputs().vec();
                SmallVector<Type, 4> resultTypes;
                for (auto &argType : inputTypes) {
                    if (isQData(argType)) {
                        auto newType = convDialectType(argType, opBuilder);
                        argType = newType;
                        resultTypes.push_back(newType);
                    }
                }
                FunctionType newType = opBuilder.getFunctionType(inputTypes, resultTypes);

                // create new function object with empty entry block (to be populated later)
                FuncOp::build(opBuilder, opState, fn.getName(), newType, {});
                newOp = newfn = opBuilder.createOperation(opState);
                cast<FuncOp>(newOp).addEntryBlock();

                // add new block argument values to the local storage
                for (unsigned i = 0; i < cast<FuncOp>(op).getNumArguments(); i++) {
                    if (isQData(fn.getArgument(i).getType()))
                        qbmap[fn.getArgument(i)] = cast<FuncOp>(newOp).getArgument(i);
                }

            } else if (isa<quantum::TerminatorOp>(op)) {
                SmallVector<Value, 4> states;
                states.reserve(qbmap.size());
                for (auto &pair : qbmap)
                    states.push_back(pair.second);

                opState = OperationState(op->getLoc(), ReturnStateOp::getOperationName());
                ReturnStateOp::build(opBuilder, opState, ValueRange(states));
                newOp = opBuilder.createOperation(opState);

                op->erase();

            } else if (isa<CallOp>(op)) {
                CallOp call = cast<CallOp>(op);
                FlatSymbolRefAttr callee = call.calleeAttr();
                FunctionType calleeType = call.getCalleeType();
                SmallVector<Value, 4> operands;
                SmallVector<Type, 4> resultTypes;
                for (auto arg : call.getOperands()) {
                    if (isQData(arg.getType())) {
                        operands.push_back(qbmap[arg]);
                        resultTypes.push_back(convDialectType(arg.getType(), opBuilder));
                    } else {
                        operands.push_back(arg);
                    }
                }

                CallOp::build(opBuilder, opState, callee, resultTypes, operands);
                newOp = opBuilder.createOperation(opState);

                for (unsigned i = 0, j = 0; i < call.getNumOperands(); i++) {
                    if (isQData(op->getOperand(i).getType()))
                        qbmap[op->getOperand(i)] = newOp->getResult(j++);
                }
                op->erase();

            } else if (isa<quantum::CircuitOp>(op)) {
                quantum::CircuitOp circOp = cast<quantum::CircuitOp>(op);
                // collect all external ssa values to generate equivalent function in second step
                std::unordered_set<Value> uniqueValues;
                for (auto &childOp : circOp.getOps()) {
                    for (auto arg : childOp.getOperands()) {
                        if (arg.getDefiningOp()->getParentOp() != op)
                            uniqueValues.insert(arg);
                    }
                }
                SmallVector<Value, 4> externalOperands(uniqueValues.begin(), uniqueValues.end());

                // generate new function argument and return types, replace qubit -> state
                SmallVector<Type, 4> inputTypes; inputTypes.reserve(externalOperands.size());
                SmallVector<Type, 4> resultTypes;
                for (auto &arg : externalOperands) {
                    if (isQData(arg.getType())) {
                        auto newType = convDialectType(arg.getType(), opBuilder);
                        inputTypes.push_back(newType);
                        resultTypes.push_back(newType);
                    } else {
                        inputTypes.push_back(arg.getType());
                    }
                }
                FunctionType newType = opBuilder.getFunctionType(inputTypes, resultTypes);

                // create new function object with empty entry block (to be populated later)
                StringRef name = circOp.name() ?
                                 circOp.name().getValue() :
                                 *(new std::string("circ" + std::to_string(numCirc++))); // ok?
                opState = OperationState(op->getLoc(), FuncOp::getOperationName());
                FuncOp::build(opBuilder, opState, name, newType, {});
                newfn = opBuilder.createOperation(opState);
                cast<FuncOp>(newfn).addEntryBlock();

                // add new block argument values to the local storage
                for (int i = 0; i < externalOperands.size(); i++) {
                    if (isQData(externalOperands[i].getType()))
                        qbmap[externalOperands[i]] = cast<FuncOp>(newfn).getArgument(i);
                }

                // create the function circuit op
                FunCircType retType = opBuilder.getType<FunCircType>();
                opState = OperationState(op->getLoc(), FunCircOp::getOperationName());
                FunCircOp::build(opBuilder, opState, retType, name, nullptr);
                newOp = opBuilder.createOperation(opState);

                // add the external operands to the circuit argument cache
                circArgMap[newOp->getResult(0)] = externalOperands;
                op->replaceAllUsesWith(newOp);

            } else if (isa<quantum::ParametricCircuitOp>(op)) {
                quantum::ParametricCircuitOp parCircOp = cast<quantum::ParametricCircuitOp>(op);
                FuncOp fun = dyn_cast<FuncOp>(parCircOp.resolveCallable());
                FunCircType retType = opBuilder.getType<FunCircType>();

                opState = OperationState(op->getLoc(), FunCircOp::getOperationName());
                FunCircOp::build(opBuilder, opState, retType, fun.getName(), parCircOp.nAttr());
                newOp = opBuilder.createOperation(opState);

                circArgMap[newOp->getResult(0)] = op->getOperands();
                op->replaceAllUsesWith(newOp);
                op->erase();

            } else if (isa<quantum::ApplyCircOp>(op)) {
                quantum::ApplyCircOp applyOp = cast<quantum::ApplyCircOp>(op);
                Operation *circDef = applyOp.circ().getDefiningOp();
                while (!isa<FunCircOp>(circDef))
                    circDef = circDef->getOperand(0).getDefiningOp(); // contractualize heldOp at 0
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
                    if (isQData(arg.getType()))
                        newOperands.push_back(qbmap[arg]);
                    else
                        newOperands.push_back(arg);
                }

                TypeAttr fcircTypeAttr = fcirc.resolveCallable()->getAttrOfType<TypeAttr>("type");
                ArrayRef<Type> retTypes = fcircTypeAttr.getValue().dyn_cast<FunctionType>();
                opState = OperationState(op->getLoc(), ApplyFunCircOp::getOperationName());
                ApplyFunCircOp::build(opBuilder, opState, retTypes, applyOp.circ(), newOperands);
                newOp = opBuilder.createOperation(opState);

                for (int i = 0, j = 0; i < oldOperands.size(); i++) {
                    if (isQData(oldOperands[i].getType()))
                        qbmap[oldOperands[i]] = newOp->getResult(j++);
                }
                op->erase();
            }

            if (newOp) {
                printf(" -- Building new op...\n  ");
                newOp->dump();
            } else {
                printf("\n");
            }
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

                printf("\n -- finalized function:\n");
                newfn->dump();
                op->erase();
                this->localStateMap.clear();
            } else if (isa<quantum::CircuitOp>(op)) {
                // finish up the new function by transfering all operations
                quantum::CircuitOp circ = cast<quantum::CircuitOp>(op);
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

        // at the very end, remove all allocation ops as their values are no longer needed
        module->walk([](quantum::AllocOp alloc) {
            alloc.erase();
        });
        module->walk([](quantum::AllocRegOp allocreg) {
            allocreg.erase();
        });
    }
};
} // end namespace

// Register passes defined in this file to make them accessible to utilities like quantum-opt.
static PassRegistration<MemToValPass> memValPass(
    "convert-mem-to-val",
    "Changes op mode from memory to value semantics, by module."
);

// Pass creation functions declared in Passes.h
std::unique_ptr<Pass> quantum::createMemToValPass() {
    return std::make_unique<MemToValPass>();
}
