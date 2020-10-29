#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
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
    // storage for operands of circuit ops, since these need to be moved to the call site (apply...)
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
                    opState = OperationState(op->getLoc(), quantumssa::AllocOp::getOperationName());
                    quantumssa::AllocOp::build(opBuilder, opState, opBuilder.getType<QstateType>());
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
                    ControlOp::build(opBuilder, opState, retType, heldOp, ctrlState, trgtState, trgtState); //fix
                else if (isa<quantum::AdjointOp>(op))
                    AdjointOp::build(opBuilder, opState, retType, heldOp, trgtState, trgtState); //fix
                newOp = opBuilder.createOperation(opState);

                // cleanup
                if (trgt)
                    qbmap[trgt] = newOp->getOpResult(0);
                else // if return value is an op, do replace all uses of its SSA value
                    op->replaceAllUsesWith(newOp);
                op->erase();

            } else if (isa<FuncOp>(op)) {
                /*FuncOp fn = cast<FuncOp>(op);

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
                }*/

            } else if (isa<quantum::TerminatorOp>(op)) {
                /*SmallVector<Value, 4> states;
                states.reserve(qbmap.size());
                for (auto &pair : qbmap)
                    states.push_back(pair.second);

                opState = OperationState(op->getLoc(), ReturnStateOp::getOperationName());
                ReturnStateOp::build(opBuilder, opState, ValueRange(states));
                newOp = opBuilder.createOperation(opState);

                op->erase();*/

            } else if (isa<CallOp>(op)) {
               /* CallOp call = cast<CallOp>(op);
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
                op->erase();*/

            } else if (isa<quantum::CircuitOp>(op)) {
                /*quantum::CircuitOp circOp = cast<quantum::CircuitOp>(op);
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
                op->replaceAllUsesWith(newOp);*/

            } else if (false/*parametric circuit op*/) {
                /*quantum::ParametricCircuitOp parCircOp = cast<quantum::ParametricCircuitOp>(op);
                FuncOp fun = dyn_cast<FuncOp>(parCircOp.resolveCallable());
                FunCircType retType = opBuilder.getType<FunCircType>();

                opState = OperationState(op->getLoc(), CircuitOp::getOperationName());
                CircuitOp::build(opBuilder, opState, retType, fun.getName(), parCircOp.nAttr());
                newOp = opBuilder.createOperation(opState);

                circArgMap[newOp->getResult(0)] = op->getOperands();
                op->replaceAllUsesWith(newOp);
                op->erase();*/

            } else if (isa<quantum::ApplyCircOp>(op)) {
                /*quantum::ApplyCircOp applyOp = cast<quantum::ApplyCircOp>(op);
                Operation *circDef = applyOp.circval().getDefiningOp();
                while (!isa<CircuitOp>(circDef))
                    circDef = circDef->getOperand(0).getDefiningOp(); // contractualize heldOp at 0
                CircuitOp fcirc = cast<CircuitOp>(circDef);
                SmallVector<Value, 4> oldOperands = circArgMap[fcirc.circval()];
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
                op->erase();*/
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


//===------------------------------------------------------------------------------------------===//
// Rewrite patterns for the consolidation of extract/combine blocks
//===------------------------------------------------------------------------------------------===//

class ExtractExtractPatt : public RewritePattern {
public:
    ExtractExtractPatt(PatternBenefit benefit, MLIRContext *context) :
        RewritePattern(ExtractOp::getOperationName(), benefit, context) {}

    // Match (extract - extract) pattern within the same region
    LogicalResult match(Operation *op) const override {
        ExtractOp extr = dyn_cast<ExtractOp>(op);
        Operation *parent = extr.reg().getDefiningOp();
        if (!isa_and_nonnull<ExtractOp>(parent) ||
                parent->getParentRegion() != op->getParentRegion())
            return failure();
        // for now only match const index operations
        if (extr.dyn_idx().size() || cast<ExtractOp>(parent).dyn_idx().size())
            return failure();
        return success();
    }

    void rewrite(Operation *op2, PatternRewriter &rewriter) const override {
        ExtractOp extr2 = dyn_cast<ExtractOp>(op2);
        Operation *op1 = extr2.reg().getDefiningOp();
        ExtractOp extr1 = dyn_cast<ExtractOp>(op1);

        // index recalculation & merging (indices of later extract op get added to end of earlier)
        ArrayAttr idxArr2 = extr2.const_idxAttr();
        ArrayAttr idxArr1 = extr1.const_idxAttr();
        int totalNumIdx = idxArr2.size() + idxArr1.size();

        llvm::SmallVector<int64_t, 8> mergedIdxArr; mergedIdxArr.reserve(totalNumIdx);
        llvm::SmallVector<int64_t, 4> sortedIdx1; sortedIdx1.reserve(idxArr1.size());
        for (auto idx1 : idxArr1) {
            mergedIdxArr.push_back(idx1.dyn_cast<IntegerAttr>().getInt());
            sortedIdx1.push_back(idx1.dyn_cast<IntegerAttr>().getInt());
        }
        std::sort(sortedIdx1.begin(), sortedIdx1.end());

        for (auto idx2 : idxArr2) {
            int64_t idx2val = idx2.dyn_cast<IntegerAttr>().getInt();
            for (int64_t idx1val : sortedIdx1) {
                if (idx1val <= idx2val)
                    idx2val++;
            }
            mergedIdxArr.push_back(idx2val);
        }

        // build new op to replace the first extract op in order to add additional result values
        // (seemingly no way currently to do it in place)
        llvm::SmallVector<Type, 9> mergedResTypes(totalNumIdx + 1, rewriter.getType<QstateType>());
        Type newRemType;
        if (auto numQubits = extr1.reg().getType().cast<RstateType>().getNumQubits()) {
            int qbsLeft = *numQubits - totalNumIdx;
            assert(qbsLeft >= 0 && "Error during ExtractOp merge, new register has negative size!");
            newRemType = rewriter.getType<RstateType>(qbsLeft);
        } else {
            newRemType = extr1.rem().getType();
        }
        mergedResTypes[totalNumIdx] = newRemType;
        OperationState state(op1->getLoc(), op1->getName());
        ExtractOp::build(rewriter, state, mergedResTypes, op1->getOperands(), op1->getAttrs());

        rewriter.setInsertionPoint(op1);
        Operation *newOp1 = rewriter.createOperation(state);
        newOp1->setAttr("const_idx", rewriter.getI64ArrayAttr(mergedIdxArr));
        ExtractOp newExtr1 = cast<ExtractOp>(newOp1);

        // extr2 will be removed entirely, return values replaced with new ones from newExtr1
        // extr1 will be replaced with the merged newExtr1, return values similarly replaced
        llvm::SmallVector<Value, 5> resultsOp1; resultsOp1.reserve(idxArr1.size() + 1);
        llvm::SmallVector<Value, 5> resultsOp2; resultsOp2.reserve(idxArr2.size() + 1);
        for (int i = 0; i < idxArr1.size(); i++)
            resultsOp1.push_back(newOp1->getResult(i));
        resultsOp1.push_back(newExtr1.rem());
        for (int i = idxArr1.size(); i < totalNumIdx; i++)
            resultsOp2.push_back(newOp1->getResult(i));
        resultsOp2.push_back(newExtr1.rem());

        // this instruciton erases the given op and replaces all its uses with the provided values
        rewriter.replaceOp(op2, resultsOp2);
        rewriter.replaceOp(op1, resultsOp1);

        if (failed(newExtr1.verify()))
            throw;
    }
};

class CombineCombinePatt : public RewritePattern {
public:
    CombineCombinePatt(PatternBenefit benefit, MLIRContext *context) :
        RewritePattern(CombineStatOp::getOperationName(), benefit, context) {}

    // Match (combine - combine) pattern within the same region
    LogicalResult match(Operation *op) const override {
        CombineStatOp comb = dyn_cast<CombineStatOp>(op);
        Operation *parent = comb.reg().getDefiningOp();
        if (!isa_and_nonnull<CombineStatOp>(parent) ||
                parent->getParentRegion() != op->getParentRegion())
            return failure();
        return success();
    }

    void rewrite(Operation *op2, PatternRewriter &rewriter) const override {
        CombineStatOp comb2 = dyn_cast<CombineStatOp>(op2);
        Operation *op1 = comb2.reg().getDefiningOp();
        CombineStatOp comb1 = dyn_cast<CombineStatOp>(op1);

        rewriter.startRootUpdate(op2);

        // index recalculation & merging (indices of later extract op get added to end of earlier)
        ArrayAttr idxArr1 = comb1.const_idxAttr();
        ArrayAttr idxArr2 = comb2.const_idxAttr();
        unsigned totalNumIdx = idxArr1.size() + idxArr2.size();

        llvm::SmallVector<int64_t, 8> mergedIdxArr; mergedIdxArr.reserve(totalNumIdx);
        llvm::SmallVector<int64_t, 4> sortedIdx2; sortedIdx2.reserve(idxArr2.size());
        for (auto idx2 : idxArr2) {
            mergedIdxArr.push_back(idx2.dyn_cast<IntegerAttr>().getInt());
            sortedIdx2.push_back(idx2.dyn_cast<IntegerAttr>().getInt());
        }
        std::sort(sortedIdx2.begin(), sortedIdx2.end());

        for (auto idx1 : idxArr1) {
            int64_t idx1val = idx1.dyn_cast<IntegerAttr>().getInt();
            for (int64_t idx2val : sortedIdx2) {
                if (idx2val <= idx1val)
                    idx1val++;
            }
            mergedIdxArr.push_back(idx1val);
        }

        op2->setAttr("const_idx", rewriter.getI64ArrayAttr(mergedIdxArr));
        // append all qubit operands of first combine to second combine
        op2->insertOperands(op2->getNumOperands(), comb1.qbs());

        rewriter.finalizeRootUpdate(op2);

        // this instruction erases the given op and replaces all its uses with the provided values
        rewriter.replaceOp(op1, comb1.reg());

        if (failed(cast<CombineStatOp>(op2).verify()))
            throw;
    }
};

class CombineExtractPatt : public RewritePattern {
public:
    CombineExtractPatt(PatternBenefit benefit, MLIRContext *context) :
        RewritePattern(ExtractOp::getOperationName(), benefit, context) {}

    // Match (combine - extract) pattern within the same region
    LogicalResult match(Operation *op2) const override {
        ExtractOp extr2 = dyn_cast<ExtractOp>(op2);
        if (!extr2 || extr2.dyn_idx().size())
            return failure();
        Operation *op1 = extr2.reg().getDefiningOp();
        if (!isa_and_nonnull<CombineStatOp>(op1) ||
                op1->getParentRegion() != op2->getParentRegion())
            return failure();

        // only match if there is some overlapp in the index sets
        CombineStatOp comb1 = cast<CombineStatOp>(op1);
        for (auto idx1 : comb1.const_idxAttr()) {
            int64_t idx1val = idx1.dyn_cast<IntegerAttr>().getInt();
            for (auto idx2 : extr2.const_idxAttr()) {
                int64_t idx2val = idx2.dyn_cast<IntegerAttr>().getInt();
                if (idx1val == idx2val)
                    return success();
            }
        }

        return failure();
    }

    void rewrite(Operation *op2, PatternRewriter &rewriter) const override {
        ExtractOp extr2 = dyn_cast<ExtractOp>(op2);
        Operation *op1 = extr2.reg().getDefiningOp();
        CombineStatOp comb1 = dyn_cast<CombineStatOp>(op1);

        rewriter.startRootUpdate(op1);
        rewriter.startRootUpdate(op2);

        // identify commmon indices
        ArrayAttr idxArr1 = comb1.const_idxAttr();
        ArrayAttr idxArr2 = extr2.const_idxAttr();
        llvm::SmallVector<int64_t, 4> newIdxArr1; newIdxArr1.reserve(idxArr1.size());
        llvm::SmallVector<std::tuple<int, int, int64_t>, 4> commonIndices;
        for (int i = 0; i < idxArr1.size(); i++) {
            int64_t idx1val = idxArr1[i].dyn_cast<IntegerAttr>().getInt();
            bool match = false;
            for (int j = 0; j < idxArr2.size(); j++) {
                int64_t idx2val = idxArr2[j].dyn_cast<IntegerAttr>().getInt();
                if (idx1val == idx2val) {
                    commonIndices.push_back({i, j, idx1val});
                    match = true;
                    break;
                }
            }
            if (!match)
                newIdxArr1.push_back(idx1val);
        }
        std::vector<Attribute> newIdxArr2 = idxArr2.getValue().vec();
        for (auto idxTriple : commonIndices)
            newIdxArr2[std::get<1>(idxTriple)] = nullptr;
        for (auto it = newIdxArr2.begin(); it != newIdxArr2.end();) {
            if (!(*it))
                it = newIdxArr2.erase(it);
            else
                it++;
        }
        // adjust indices
        for (int64_t &idx : newIdxArr1) {
            int offset = 0;
            for (auto idxTriple : commonIndices) {
                if (std::get<2>(idxTriple) < idx)
                    offset++;
            }
            idx -= offset;
        }
        for (auto &idxAttr : newIdxArr2) {
            int offset = 0;
            int64_t idxVal = idxAttr.dyn_cast<IntegerAttr>().getInt();
            for (auto idxTriple : commonIndices) {
                if (std::get<2>(idxTriple) < idxVal)
                    offset++;
            }
            idxAttr = rewriter.getIntegerAttr(rewriter.getI64Type(), idxVal - offset);
        }
        // remove indices from combine
        op1->setAttr("const_idx", rewriter.getI64ArrayAttr(newIdxArr1));
        // remove indices from extract
        op2->setAttr("const_idx", rewriter.getArrayAttr(newIdxArr2));

        rewriter.finalizeRootUpdate(op2);

        // replace extract return values with combine qb args
        if (!newIdxArr2.empty()) {
            // build new extract op, unless empty
            llvm::SmallVector<Type, 4> newExtrResTypes(newIdxArr2.size() + 1,
                                                       rewriter.getType<QstateType>());
            Type newRemType;
            if (auto numQubits = extr2.reg().getType().cast<RstateType>().getNumQubits()) {
                int qbsLeft = *numQubits - commonIndices.size() - newIdxArr2.size();
                assert(qbsLeft >= 0 && "Error during Combine-Extract common index cancelation, "
                                       "new register has negative size!");
                newRemType = rewriter.getType<RstateType>(qbsLeft);
            } else {
                newRemType = extr2.rem().getType();
            }
            newExtrResTypes[newIdxArr2.size()] = newRemType;
            OperationState state(op2->getLoc(), op2->getName());
            ExtractOp::build(rewriter, state, newExtrResTypes, op2->getOperands(), op2->getAttrs());

            rewriter.setInsertionPoint(op2);
            Operation *newOp2 = rewriter.createOperation(state);
            ExtractOp newExtr2 = cast<ExtractOp>(newOp2);
            // build updated value list
            llvm::SmallVector<Value, 4> updatedValues; updatedValues.reserve(op2->getNumResults());
            for (int i = 0, j = 0, k = 0; i < extr2.qbs().size(); i ++) {
                if (j < commonIndices.size() && i == std::get<1>(commonIndices[j]))
                    updatedValues.push_back(comb1.qbs()[std::get<0>(commonIndices[j++])]);
                else
                    updatedValues.push_back(newExtr2.qbs()[k++]);
            }
            updatedValues.push_back(newExtr2.rem());
            // replace old with new op
            rewriter.replaceOp(op2, updatedValues);
        } else {
            assert(idxArr2.size() == commonIndices.size() && "something went wrong in the match!");
            // build array of replacement values from combine
            llvm::SmallVector<Value, 4> passthroughValues;
            passthroughValues.reserve(commonIndices.size());
            for (auto idxTriple : commonIndices)
                passthroughValues.push_back(comb1.qbs()[std::get<0>(idxTriple)]);
            // add reg from current extract op
            passthroughValues.push_back(extr2.reg());
            // replace
            rewriter.replaceOp(op2, passthroughValues);
        }

        // remove combine qb args, updating return type requires building new op
        if (!newIdxArr1.empty()) {
            // remove obsolete qubit arguments from combine op
            // index order in commonIndices guaranteed to be as they appear in comb1
            int delOffset = 0;
            for (auto idxTriple : commonIndices) {
                op1->eraseOperand(std::get<0>(idxTriple)+1 - delOffset++);
            }

            rewriter.finalizeRootUpdate(op1);

            // build new extract op, unless empty
            Type newregType;
            if (auto numQubits = comb1.reg().getType().cast<RstateType>().getNumQubits()) {
                int qbsLeft = *numQubits + newIdxArr1.size();
                assert(qbsLeft >= 0 && "Error during Combine-Extract common index cancelation, "
                                       "new register has negative size!");
                newregType = rewriter.getType<RstateType>(qbsLeft);
            } else {
                newregType = comb1.reg().getType();
            }
            OperationState state(op1->getLoc(), op1->getName());
            CombineStatOp::build(rewriter, state, newregType, op1->getOperands(), op1->getAttrs());

            rewriter.setInsertionPoint(op1);
            Operation *newOp1 = rewriter.createOperation(state);
            CombineStatOp newComb1 = cast<CombineStatOp>(newOp1);

            // replace old with new op
            rewriter.replaceOp(op1, newComb1.newreg());
        } else {
            rewriter.finalizeRootUpdate(op1);
            // if empty delete combine, replace newreg with reg
            rewriter.replaceOp(op1, comb1.reg());
        }
    }
};

class CombineExtractCombinePatt : public RewritePattern {
public:
    CombineExtractCombinePatt(PatternBenefit benefit, MLIRContext *context) :
        RewritePattern(CombineStatOp::getOperationName(), benefit, context) {}

    // Match (combine - extract - combine) pattern within the same region
    LogicalResult match(Operation *op3) const override {
        CombineStatOp comb3 = dyn_cast<CombineStatOp>(op3);
        Operation *op2 = comb3.reg().getDefiningOp();
        if (!isa_and_nonnull<ExtractOp>(op2) || op2->getParentRegion() != op3->getParentRegion())
            return failure();

        ExtractOp extr2 = cast<ExtractOp>(op2);
        if (extr2.dyn_idx().size())
            return failure();
        Operation *op1 = extr2.reg().getDefiningOp();
        if (!isa_and_nonnull<CombineStatOp>(op1) ||
                op1->getParentRegion() != op2->getParentRegion())
            return failure();

        // only match if there is no overlapp in the index sets
        CombineStatOp comb1 = cast<CombineStatOp>(op1);
        for (auto idx1 : comb1.const_idxAttr()) {
            int64_t idx1val = idx1.dyn_cast<IntegerAttr>().getInt();
            for (auto idx2 : extr2.const_idxAttr()) {
                int64_t idx2val = idx2.dyn_cast<IntegerAttr>().getInt();
                if (idx1val == idx2val)
                    return failure();
            }
        }

        return success();
    }

    void rewrite(Operation *op3, PatternRewriter &rewriter) const override {
        CombineStatOp comb3 = dyn_cast<CombineStatOp>(op3);
        Operation *op2 = comb3.reg().getDefiningOp();
        ExtractOp extr2 = dyn_cast<ExtractOp>(op2);
        Operation *op1 = extr2.reg().getDefiningOp();
        CombineStatOp comb1 = dyn_cast<CombineStatOp>(op1);

        rewriter.startRootUpdate(op2);
        rewriter.startRootUpdate(op3);

        // append indices from comb1 to comb3
        ArrayAttr idxArr1 = comb1.const_idx();
        ArrayAttr idxArr3 = comb3.const_idx();
        llvm::SmallVector<int64_t, 8> newIdxArr; newIdxArr.reserve(idxArr1.size() + idxArr3.size());
        for (auto idx3 : idxArr3)
            newIdxArr.push_back(idx3.dyn_cast<IntegerAttr>().getInt());
        for (auto idx1 : idxArr1)
            newIdxArr.push_back(idx1.dyn_cast<IntegerAttr>().getInt());
        op3->setAttr("const_idx", rewriter.getI64ArrayAttr(newIdxArr));
        // append qb args from comb1 to comb3
        op3->insertOperands(op3->getNumOperands(), comb1.qbs());
        rewriter.finalizeRootUpdate(op3);
        // shift indices for middle extract
        std::vector<Attribute> idxArr2 = extr2.const_idxAttr().getValue().vec();
        for (auto &idx2 : idxArr2) {
            int offset = 0;
            int64_t idx2val = idx2.dyn_cast<IntegerAttr>().getInt();
            for (auto idx1 : idxArr1) {
                int64_t idx1val = idx1.dyn_cast<IntegerAttr>().getInt();
                if (idx1val < idx2val)
                    offset++;
            }
            idx2 = rewriter.getIntegerAttr(rewriter.getI64Type(), idx2val - offset);
        }
        op2->setAttr("const_idx", rewriter.getArrayAttr(idxArr2));
        rewriter.finalizeRootUpdate(op2);
        // delete comb1, replace extr2 reg with comb1 reg
        rewriter.replaceOp(op1, comb1.reg());

        // update return type of extr2
        int newInSize = *op2->getOperand(0).getType().cast<RstateType>().getNumQubits();
        int newOutSize = newInSize - idxArr2.size();
        RstateType newRemType = rewriter.getType<RstateType>(newOutSize);
        std::vector<Type> newResTypes = op2->getResultTypes().vec();
        newResTypes[newResTypes.size()-1] = newRemType;

        OperationState state(op2->getLoc(), op2->getName());
        ExtractOp::build(rewriter, state, newResTypes, op2->getOperands(), op2->getAttrs());
        rewriter.setInsertionPoint(op2);
        Operation *newOp2 = rewriter.createOperation(state);

        rewriter.replaceOp(op2, newOp2->getResults());
    }
};

class ExtractCombinePatt : public RewritePattern {
public:
    ExtractCombinePatt(PatternBenefit benefit, MLIRContext *context) :
        RewritePattern(CombineStatOp::getOperationName(), benefit, context) {}

    // Match (extract - combine) pattern within the same region
    LogicalResult match(Operation *op2) const override {
        CombineStatOp comb2 = cast<CombineStatOp>(op2);
        Operation *op1 = comb2.reg().getDefiningOp();
        if (!isa_and_nonnull<ExtractOp>(op1) || op1->getParentRegion() != op2->getParentRegion())
            return failure();
        ExtractOp extr1 = cast<ExtractOp>(op1);
        if (extr1.dyn_idx().size())
            return failure();

        // only match if there is some direct overlapp in the qubit sets, implying unused values
        for (Value qb1 : extr1.qbs()) {
            for (Value qb2 : comb2.qbs()) {
                if (qb1 == qb2)
                    return success();
            }
        }

        return failure();
    }

    void rewrite(Operation *op2, PatternRewriter &rewriter) const override {
        CombineStatOp comb2 = cast<CombineStatOp>(op2);
        Operation *op1 = comb2.reg().getDefiningOp();
        ExtractOp extr1 = cast<ExtractOp>(op1);

        // identify common qbs
        ResultRange extrQbs = extr1.qbs();
        OperandRange combQbs = comb2.qbs();
        llvm::SmallVector<std::pair<int, int>, 4> commonQbs;
        for (int i = 0; i < combQbs.size(); i++) {
            for (int j = 0; j < extrQbs.size(); j++) {
                if (combQbs[i] == extrQbs[j]) {
                    commonQbs.push_back({i, j});
                    break;
                }
            }
        }

        // if either is empty, delete
        int combQbsLeft = comb2.qbs().size() - commonQbs.size();
        assert(combQbsLeft >= 0 && "Negative # combine qbs left during Extract-Combine!");
        if (!combQbsLeft) {
            rewriter.replaceOp(op2, comb2.reg());
        } else {
            rewriter.startRootUpdate(op2);
            // remove common qbs from comb2
            int delOffset = 0;
            for (auto idxPair : commonQbs) {
                op2->eraseOperand(idxPair.first+1 - delOffset++);
            }
            // remove indices from attribute
            ArrayAttr idxArr2 = comb2.const_idxAttr();
            llvm::SmallVector<Attribute, 3> newIdxArr2; newIdxArr2.reserve(combQbsLeft);
            for (int i = 0, j = 0; i < idxArr2.size(); i++) {
                if (j < commonQbs.size() && i != commonQbs[j].first) {
                    newIdxArr2.push_back(idxArr2[i]);
                    j++;
                }
            }
            op2->setAttr("const_idx", rewriter.getArrayAttr(newIdxArr2));
            rewriter.finalizeRootUpdate(op2);
        }

        // if either is empty, delete
        int extrQbsLeft = extr1.qbs().size() - commonQbs.size();
        assert(extrQbsLeft >= 0 && "Negative # extract qbs left during Extract-Combine!");
        if (!extrQbsLeft) {
            llvm::SmallVector<Value, 4> replacementValues(op1->getNumResults(), nullptr);
            replacementValues[op1->getNumResults()-1] = extr1.reg();
            rewriter.replaceOp(op1, replacementValues);
        } else {
            // to remove common qbs from extr1 return values, need to build new op
            llvm::SmallVector<Type, 4> newResTypes(extrQbsLeft+1, rewriter.getType<QstateType>());
            Type regType = extr1.reg().getType();
            Type newRemType = regType.isa<RstateType>() ? rewriter.getType<RstateType>(extrQbsLeft)
                                                        : extr1.rem().getType();

            newResTypes[newResTypes.size()-1] = newRemType;
            OperationState state(op1->getLoc(), op1->getName());
            ExtractOp::build(rewriter, state, newResTypes, op1->getOperands(), op1->getAttrs());
            rewriter.setInsertionPoint(op1);
            Operation *newOp1 = rewriter.createOperation(state);

            // build updated value list, deleted qbs will be replaced with null values
            // as there aren't anymore uses of those values but replaceOp expects a value anyways
            llvm::SmallVector<Value, 4> replaceVals; replaceVals.reserve(op1->getNumResults());
            for (Value retVal : newOp1->getResults())
                replaceVals.push_back(retVal);
            // to correctly insert null values basend on original indices, need to sort them first
            std::sort(commonQbs.begin(), commonQbs.end(),
                      [](auto &left, auto &right) { return left.second < right.second; });
            for (auto idxPair : commonQbs)
                    replaceVals.insert(replaceVals.begin()+idxPair.second, nullptr);
            // remove indices from attribute
            ArrayAttr idxArr1 = extr1.const_idxAttr();
            llvm::SmallVector<Attribute, 3> newIdxArr1; newIdxArr1.reserve(extrQbsLeft);
            for (int i = 0, j = 0; i < idxArr1.size(); i++) {
                if (j < commonQbs.size() && i != commonQbs[j].second) {
                    newIdxArr1.push_back(idxArr1[i]);
                    j++;
                }
            }
            rewriter.startRootUpdate(newOp1);
            newOp1->setAttr("const_idx", rewriter.getArrayAttr(newIdxArr1));
            rewriter.finalizeRootUpdate(newOp1);

            // replace old op
            rewriter.replaceOp(op1, replaceVals);
        }
    }
};
} // end namespace

// Pass creation functions declared in Passes.h
std::unique_ptr<Pass> quantum::createMemToValPass() {
    return std::make_unique<MemToValPass>();
}

// Register all patterns for rewrite by the Canonicalization framework
void ExtractOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context) {
    patterns.insert<ExtractExtractPatt>(3, context);
    patterns.insert<CombineExtractPatt>(2, context);
}

void CombineStatOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                                MLIRContext *context) {
    patterns.insert<CombineCombinePatt>(3, context);
    patterns.insert<CombineExtractCombinePatt>(1, context);
    patterns.insert<ExtractCombinePatt>(2, context);
}
