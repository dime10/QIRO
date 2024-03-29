#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"

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

bool isQSSAData(Type ty) {
    return ty.isa<QstateType>() || ty.isa<RstateType>();
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
    MemToValPass() : OperationPass<ModuleOp>(TypeID::get<MemToValPass>()) {}
    MemToValPass(const MemToValPass &) : OperationPass<ModuleOp>(TypeID::get<MemToValPass>()) {}

    StringRef getName() const override {
        return "MemToValPass";
    }

    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<MemToValPass>(*this);
    }

private:
    using value_map = std::unordered_map<Value, Value>;
    OpBuilder *opBuilder;
    // keep a record of the latest qubit state to replace qubit values with
    value_map globalStateMap;
    // temporary storage for newly created circuits
    Operation *circInProg;

    void walk(Operation *op, value_map &outerStateMap, SmallVectorImpl<Value> &outerUniqueValues,
              function_ref<void(Operation*, value_map&, const SmallVectorImpl<Value>&,
                           const SmallVectorImpl<Value>&)> callback) {
        // the qubit state mapping and list of values used inside op region needs to be nestable
        value_map innerStateMap;
        SmallVector<Value, 8> innerUniqueValues;
        SmallVector<Value, 8> iterArgs;
        SmallVector<Value, 0> dummy;

        // do not recurse into functions as they do not contain any quantum operations
        if (isa<FuncOp>(op))
            return callback(op, innerStateMap, dummy, dummy);

        // some ops need some preprocessing before nested ops are recursed into
        if (isa<quantum::CircuitOp>(op))  // circuit ops need a fully local (empty) inner state map
            callback(op, innerStateMap, dummy, dummy);
        else if (isa<scf::IfOp>(op))      // map is unused here
            prepControlFlow(op, *opBuilder, innerStateMap, innerUniqueValues, iterArgs);
        else if (isa<scf::ForOp>(op)) {  // CF needs a temperary local copy of the outer map
            innerStateMap = outerStateMap;
            prepControlFlow(op, *opBuilder, innerStateMap, innerUniqueValues, iterArgs);
        }

        // walk nested operations
        bool isSCF = isa<scf::ForOp>(op) || isa<scf::IfOp>(op);
        bool isNesting = isa<quantum::CircuitOp>(op) || isSCF;
        value_map &mapToUse = isNesting ? innerStateMap : outerStateMap;
        SmallVectorImpl<Value> &valsToUse = isSCF ? innerUniqueValues : outerUniqueValues;
        for (auto &region : op->getRegions()) {
            if (isa<scf::IfOp>(op))       // SCF:If as above, but temp copy between each region
                innerStateMap = outerStateMap;
            for (auto &block : region)
                for (auto &nestedOp : llvm::make_early_inc_range(block))
                    walk(&nestedOp, mapToUse, innerUniqueValues, callback);
        }

        // perform postorder call -> most ops as they can only be deleted after recursing
        if (isa<quantum::CircuitOp>(op))
            finalizeCircuit(op, circInProg);
        else
            callback(op, outerStateMap, valsToUse, iterArgs);
    }

    static Type convDialectType(Builder &b, Type inType) {
        Type outType;
        if (inType.isa<quantum::U1Type>())
            outType = b.getType<U1Type>();
        else if (inType.isa<quantum::U2Type>())
            outType = b.getType<U2Type>();
        else if (auto copType = inType.dyn_cast<quantum::COpType>())
            outType = b.getType<COpType>(copType.getNumCtrls(),
                                         convDialectType(b, copType.getBaseType()));
        else if (inType.isa<quantum::CircType>())
            outType = b.getType<CircType>();
        else if (inType.isa<quantum::QubitType>())
            outType = b.getType<QstateType>();
        else if (auto regType = inType.dyn_cast<quantum::QuregType>())
            outType = b.getType<RstateType>(regType.getNumQubits());
        else
            assert(false && "Unrecognized dialect type encountered during conversion!");
        return outType;
    }

    static Operation* buildExtr(OpBuilder &b, Operation *op, const Value &reg,
                                const ValueRange &range, const ArrayAttr &staticRange) {
        assert(staticRange.size() >= 1 && "Need at least one index to extract qubits!");

        llvm::Optional<int> numQubits = reg.getType().cast<RstateType>().getNumQubits();
        numQubits = numQubits ? llvm::Optional<int>(*numQubits - staticRange.size()) : llvm::None;
        SmallVector<Type, 3> retTypes(staticRange.size()+1, b.getType<QstateType>());
        retTypes[staticRange.size()] = b.getType<RstateType>(numQubits);
        bool dynamic = staticRange[0].cast<IntegerAttr>().getInt() == -1;

        OperationState extrState(op->getLoc(), ExtractOp::getOperationName());
        if (dynamic)
            ExtractOp::build(b, extrState, retTypes, reg, range, nullptr);
        else
            ExtractOp::build(b, extrState, retTypes, reg, {}, staticRange);

        return b.createOperation(extrState);
    }

    static Operation* buildComb(OpBuilder &b, Operation *op, const Value &reg, const ValueRange &qbs,
                                const ValueRange &range, const ArrayAttr &staticRange) {
        assert(staticRange.size() >= 1 && "Need at least one index to extract qubits!");

        llvm::Optional<int> numQubits = reg.getType().cast<RstateType>().getNumQubits();
        numQubits = numQubits ? llvm::Optional<int>(*numQubits + staticRange.size()) : llvm::None;
        Type retType = b.getType<RstateType>(numQubits);
        bool dynamic = staticRange[0].cast<IntegerAttr>().getInt() == -1;

        if (dynamic) {
            OperationState combState(op->getLoc(), CombineDynOp::getOperationName());
            CombineDynOp::build(b, combState, retType, reg, range, qbs);
            return b.createOperation(combState);
        } else {
            OperationState combState(op->getLoc(), CombineStatOp::getOperationName());
            CombineStatOp::build(b, combState, retType, reg, staticRange, qbs);
            return b.createOperation(combState);
        }
    }

    static void parseGateParams(OpBuilder &b, value_map &qbmap, Operation *op,
                                SmallVectorImpl<Value> &operands, SmallVectorImpl<Value> &qargs,
                                SmallVectorImpl<Type> &retTypes, bool hasPhi) {

        if (hasPhi && (!op->getNumOperands() || !op->getOperand(0).getType().isa<FloatType>())) {
            OperationState auxState(op->getLoc(), ConstantOp::getOperationName());
            ConstantOp::build(b, auxState, op->getAttr("static_phi"));
            operands.push_back(b.createOperation(auxState)->getResult(0));
        }

        for (auto arg : op->getOperands()) {
            if (arg.getType().isa<IndexType>()) {
                continue;
            } else if (isQData(arg.getType())) {
                operands.push_back(qbmap[arg]);
                qargs.push_back(arg);
                retTypes.push_back(qbmap[arg].getType());
            } else {
                operands.push_back(arg);
            }
        }

        if (op->getNumResults())
            retTypes.push_back(convDialectType(b, op->getResult(0).getType()));
    }

    template<class G> static Operation* buildGate(OpBuilder &b, value_map &qbmap, Operation *op,
                                                  ArrayRef<ValueRange> ranges,
                                                  ArrayRef<ArrayAttr> staticRanges) {
        SmallVector<Value, 4> operands;  // all operands to the new op, qdata replaced from qbmap
        SmallVector<Value, 3> qargs;     // original qdata arguments to input op
        SmallVector<Value, 3> newStates; // qdata return values to update qbmap with
        SmallVector<Type, 3> retTypes;   // return types of the new op
        bool hasPhi = std::is_same<G, RzOp>::value || std::is_same<G, ROp>::value;
        parseGateParams(b, qbmap, op, operands, qargs, retTypes, hasPhi);
        SmallVector<Operation*, 2> extrOps, combOps;

        // collect information about indexed registers, each distinct register needs its own
        // extract/combine ops, multiple uses of the same register are combined into one pair
        using access_info = std::tuple<Value, SmallVector<Value, 2>, SmallVector<int64_t, 2>,
                                       SmallVector<std::pair<int, int>, 2>>;
        SmallVector<access_info, 2> regsToExtr;
        std::unordered_map<Value, int> seen;
        for (size_t i = 0, j = 0; i < operands.size(); i++) {
            if (isQSSAData(operands[i].getType())) {
                if (staticRanges[j].size() == 1) {
                    if (seen.count(operands[i])) {
                        access_info &t = regsToExtr[seen[operands[i]]];
                        if (ranges[j].size())
                            std::get<1>(t).push_back(ranges[j][0]);
                        std::get<2>(t).push_back(staticRanges[j][0].cast<IntegerAttr>().getInt());
                        std::get<3>(t).push_back({i, j});
                    } else {
                        seen[operands[i]] = regsToExtr.size();
                        regsToExtr.push_back(std::make_tuple(
                            operands[i],
                            ranges[j].size() ? SmallVector<Value, 2>(1, ranges[j][0])
                                             : SmallVector<Value, 2>(),
                            SmallVector<int64_t, 2>(1, staticRanges[j][0].cast<IntegerAttr>().getInt()),
                            SmallVector<std::pair<int, int>, 2>(1, {i, j})
                        ));
                    }
                } else if (staticRanges[j].size() > 1) {
                    assert(false && "Ranges are not yet supported!");
                }
                j++;
            }
        }

        // build extract op for each distinct register argument with single index only
        for (access_info &t : regsToExtr) {
            Operation *extrOp = buildExtr(b, op, std::get<0>(t), std::get<1>(t),
                                          b.getI64ArrayAttr(std::get<2>(t)));
            auto resIt = extrOp->result_begin();
            for (auto &pair : std::get<3>(t)) {
                int i = pair.first, j = pair.second;
                operands[i] = *resIt++;
                retTypes[j] = operands[i].getType();
            }
            extrOps.push_back(extrOp);
            #ifdef DEBUG
                extrOp->dump();
            #endif
        }

        OperationState opState(op->getLoc(), G::getOperationName());
        G::build(b, opState, retTypes, operands, {});
        Operation *newOp = b.createOperation(opState);
        newStates.append(newOp->result_begin(), op->getNumResults() ? --newOp->result_end()
                                                                    : newOp->result_end());
        if (op->getAttr("compute"))
            newOp->setAttr("compute", b.getUnitAttr());
        if (op->getAttr("uncompute"))
            newOp->setAttr("uncompute", b.getUnitAttr());

        for (access_info &t : regsToExtr) {
            SmallVector<Value, 2> qbs;
            for (auto &pair : std::get<3>(t)) {
                int j = pair.second;
                qbs.push_back(newOp->getResult(j));
            }
            Value reg = extrOps[combOps.size()]->getResults().back();
            Operation *combOp = buildComb(b, op, reg, qbs, std::get<1>(t),
                                          b.getI64ArrayAttr(std::get<2>(t)));
            for (auto &pair : std::get<3>(t)) {
                int j = pair.second;
                newStates[j] = combOp->getResult(0);
            }
            combOps.push_back(combOp);
            #ifdef DEBUG
                extrOp->dump();
            #endif
        }

        auto resIt = newStates.begin();
        for (auto arg : qargs)
            qbmap[arg] = *resIt++;
        if (op->getNumResults())
            op->getResult(0).replaceAllUsesWith(newOp->getResult(retTypes.size()-1));
        op->erase();

        return newOp;
    }

    template<class A> static Operation* buildAlloc(OpBuilder &b, value_map &qbmap, Operation *op) {
        Value ret = op->getResult(0);
        SmallVector<Type, 1> retTypes(1, convDialectType(b, ret.getType()));
        SmallVector<Value, 1> operands;
        SmallVector<NamedAttribute, 1> attrs;

        if (op->getNumOperands())
            operands.push_back(op->getOperand(0));
        if (auto attr = op->getAttr("static_size"))
            attrs.push_back(NamedAttribute(Identifier::get("static_size", b.getContext()), attr));

        OperationState opState(op->getLoc(), A::getOperationName());
        A::build(b, opState, retTypes, operands, attrs);
        Operation *newOp = b.createOperation(opState);

        qbmap[ret] = newOp->getResult(0);
        // can't remove the op yet as it's return value is still needed for the qubit map

        return newOp;
    }

    template<class F> static Operation* buildFree(OpBuilder &b, value_map &qbmap, Operation *op) {
        Value arg = op->getOperand(0);

        OperationState opState(op->getLoc(), F::getOperationName());
        F::build(b, opState, {}, {qbmap[arg]}, {});
        Operation *newOp = b.createOperation(opState);

        qbmap.erase(arg);
        op->erase();

        return newOp;
    }

    template<class M> static Operation* buildMeas(OpBuilder &b, value_map &qbmap, Operation *op,
                                                  const ValueRange &range,
                                                  const ArrayAttr &staticRange) {
        Value arg = op->getOperand(0);
        Value mres = op->getResult(0);
        SmallVector<Value, 1> operands(1, qbmap[arg]);
        SmallVector<Type, 2> retTypes({convDialectType(b, arg.getType()), mres.getType()});
        Operation *extrOp, *combOp;

        if (staticRange.size() == 1) {
            extrOp = buildExtr(b, op, qbmap[arg], range, staticRange);
            operands[0] = extrOp->getResult(0);
            retTypes[0] = operands[0].getType();
        }

        OperationState opState(op->getLoc(), M::getOperationName());
        M::build(b, opState, retTypes, operands, {});
        Operation *newOp = b.createOperation(opState);
        if (op->getAttr("compute"))
            newOp->setAttr("compute", b.getUnitAttr());
        if (op->getAttr("uncompute"))
            newOp->setAttr("uncompute", b.getUnitAttr());

        if (staticRange.size() == 1) {
            Value reg = extrOp->getResult(1);
            ValueRange qbs = {newOp->getResults()[0]};
            combOp = buildComb(b, op, reg, qbs, range, staticRange);
        }

        qbmap[arg] = staticRange.size() == 1 ? combOp->getResult(0) : newOp->getResult(0);
        mres.replaceAllUsesWith(newOp->getResult(1));
        op->erase();

        return newOp;
    }

    template<class C> static Operation* buildCall(OpBuilder &b, value_map &qbmap, Operation *op) {
        SmallVector<Value, 4> operands;
        SmallVector<Value, 4> qargs;
        SmallVector<Type, 4> resultTypes;
        SmallVector<NamedAttribute, 1> attrs;

        for (auto arg : op->getOperands()) {
            if (isQData(arg.getType())) {
                operands.push_back(qbmap[arg]);
                qargs.push_back(arg);
                resultTypes.push_back(qbmap[arg].getType());
            } else {
                operands.push_back(arg);
            }
        }
        if (auto attr = op->getAttr("circref"))
            attrs.push_back(NamedAttribute(Identifier::get("circref", b.getContext()), attr));
        if (auto array = op->getAttrOfType<ArrayAttr>("size_params")) {
            auto operIt = operands.begin();
            for (auto attr : array) {
                int64_t size = attr.cast<IntegerAttr>().getInt();
                if (size != -1) {
                    OperationState auxState(op->getLoc(), ConstantOp::getOperationName());
                    ConstantOp::build(b, auxState, b.getIndexType(), b.getIndexAttr(size));
                    operands.insert(operIt, b.createOperation(auxState)->getResult(0));
                }
                operIt++;
            }
        }

        OperationState opState(op->getLoc(), C::getOperationName());
        C::build(b, opState, resultTypes, operands, attrs);
        Operation *newOp = b.createOperation(opState);
        if (op->getAttr("compute"))
            newOp->setAttr("compute", b.getUnitAttr());
        if (op->getAttr("uncompute"))
            newOp->setAttr("uncompute", b.getUnitAttr());

        auto resIt = newOp->result_begin();
        for (auto arg : qargs)
            qbmap[arg] = *resIt++;
        op->erase();

        return newOp;
    }

    template<class T> static Operation* buildTerm(OpBuilder &b, value_map &qbmap, Operation *op,
                                                  ValueRange args) {
        SmallVector<Value, 4> states;
        for (auto arg : args) {
            if (isQData(arg.getType()))
                states.push_back(qbmap[arg]);
        }

        OperationState opState(op->getLoc(), T::getOperationName());
        T::build(b, opState, ValueRange(states));
        Operation *newOp = b.createOperation(opState);

        op->erase();

        return newOp;
    }

    static void gatherUniqueValues(Operation *op, std::unordered_set<Value> &valueSet) {
        for (auto &region : op->getRegions()) {
            for (auto &childOp : region.getOps()) {
                for (auto arg : childOp.getOperands())
                    if (isQData(arg.getType()))
                        valueSet.insert(arg);
                gatherUniqueValues(&childOp, valueSet);
            }
        }
    }

    static void prepControlFlow(Operation *op, OpBuilder &b, value_map &qbmap,
                                SmallVectorImpl<Value> &uniqueValues,
                                SmallVectorImpl<Value> &iterArgs) {
        std::unordered_set<Value> valueSet;
        gatherUniqueValues(op, valueSet);
        uniqueValues.append(valueSet.begin(), valueSet.end());

        if (isa<scf::ForOp>(op)) {
            SmallVector<Type, 4> iterTypes;
            iterTypes.reserve(uniqueValues.size());
            iterArgs.reserve(uniqueValues.size());
            for (auto arg : uniqueValues)
                iterArgs.push_back(qbmap[arg]);
            for (auto arg : iterArgs)
                iterTypes.push_back(arg.getType());

            op->insertOperands(op->getNumOperands(), iterArgs);
            op->getRegion(0).addArguments(iterTypes);

            auto argIt = uniqueValues.begin();
            for (auto blockArg : op->getRegion(0).getArguments())
                if (isQSSAData(blockArg.getType()))
                    qbmap[*argIt++] = blockArg;
        } else {
            // prep if/else regions: since we are returning values from the IfOp,
            // we always need atleast a yield statement in the else region
            Region &region = op->getRegion(1);
            cast<scf::IfOp>(op).ensureTerminator(region, b, region.getLoc());
        }
    }

    static void finalizeCircuit(Operation *op, Operation *newOp) {
        // finish up the new function by transfering all operations
        quantum::CircuitOp circ = cast<quantum::CircuitOp>(op);
        CircuitOp newCirc = cast<CircuitOp>(newOp);
        BlockAndValueMapping valmap;
        valmap.map<Block::BlockArgListType, Block::BlockArgListType>(
            circ.getArguments(), newCirc.getArguments());

        // the first block needs to be cloned into our existing entry block
        for (auto &op : circ.front()) {
            newCirc.front().push_back(op.clone(valmap));
        }

        circ.erase();
        #ifdef DEBUG
            printf("\n -- finalized circuit:\n");
            newCirc.dump();
        #endif
    }

public:
    void runOnOperation() override {
        Operation *module = getOperation().getOperation();
        // the op builder can create new ops and types for us
        OpBuilder b(module->getContext());
        this->opBuilder = &b;
        SmallVector<Value, 0> dummy;

        walk(module, globalStateMap, dummy,
        [this, &b] (Operation *op, value_map &qbmap,
                            const SmallVectorImpl<Value> &uniqueValues,
                            const SmallVectorImpl<Value> &iterArgs) {
            #ifdef DEBUG
                op->dump();
            #endif

            // nothing to do for operations outside the Quantum dialect
            if (!isa<quantum::QuantumDialect>(op->getDialect()) &&
                    !isa<scf::SCFDialect>(op->getDialect()))
                return;

            // set up common resources
            b.setInsertionPoint(op);
            Operation *newOp = nullptr;
            unsigned n;

            if (isa<quantum::AllocOp>(op)) {
                newOp = buildAlloc<quantumssa::AllocOp>(b, qbmap, op);
            } else if (isa<quantum::AllocRegOp>(op)) {
                newOp = buildAlloc<AllocRegOp>(b, qbmap, op);

            } else if (isa<quantum::FreeOp>(op)) {
                newOp = buildFree<FreeOp>(b, qbmap, op);
            } else if (isa<quantum::FreeRegOp>(op)) {
                newOp = buildFree<FreeRegOp>(b, qbmap, op);

            } else if (auto h = dyn_cast<quantum::HOp>(op)) {
                newOp = buildGate<HOp>(b, qbmap, op, {h.range()}, {h.static_range()});
            } else if (auto x = dyn_cast<quantum::XOp>(op)) {
                newOp = buildGate<XOp>(b, qbmap, op, {x.range()}, {x.static_range()});
            } else if (auto rz = dyn_cast<quantum::RzOp>(op)) {
                newOp = buildGate<RzOp>(b, qbmap, op, {rz.range()}, {rz.static_range()});
            } else if (auto r = dyn_cast<quantum::ROp>(op)) {
                newOp = buildGate<ROp>(b, qbmap, op, {r.range()}, {r.static_range()});

            } else if (auto cx = dyn_cast<quantum::CNotOp>(op)) {
                newOp = buildGate<CNotOp>(b, qbmap, op,
                    {cx.crange(), cx.qrange()}, {cx.static_crange(), cx.static_qrange()});
                n = newOp->getNumOperands();
                newOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({!!n, !!n}));

            } else if (auto sw = dyn_cast<quantum::SwapOp>(op)) {
                newOp = buildGate<SwapOp>(b, qbmap, op,
                    {sw.range(), sw.range2()}, {sw.static_range(), sw.static_range2()});
                n = newOp->getNumOperands();
                newOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({!!n, !!n}));

            } else if (auto ctrl = dyn_cast<quantum::ControlOp>(op)) {
                newOp = buildGate<ControlOp>(b, qbmap, op,
                    {ctrl.crange(), ctrl.range(), ctrl.range2()},
                    {ctrl.static_crange(), ctrl.static_range(), ctrl.static_range2()});
                n = newOp->getNumOperands();
                newOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, n==4, n>=3}));

            } else if (auto adj = dyn_cast<quantum::AdjointOp>(op)) {
                newOp = buildGate<AdjointOp>(b, qbmap, op,
                    {adj.range(), adj.range2()}, {adj.static_range(), adj.static_range2()});
                n = newOp->getNumOperands();
                newOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, n==3, n>=2}));

            } else if (auto m = dyn_cast<quantum::MeasurementOp>(op)) {
                newOp = buildMeas<MeasurementOp>(b, qbmap, op, m.range(), m.static_range());

            } else if (auto circ = dyn_cast<quantum::CircuitOp>(op)) {
                // generate new function argument and return type, replace qubit -> state
                std::vector<Type> inputTypes = circ.getType().getInputs().vec();
                SmallVector<Type, 4> returnTypes;
                for (auto &argType : inputTypes) {
                    if (isQData(argType)) {
                        argType = convDialectType(b, argType);
                        returnTypes.push_back(argType);
                    }
                }
                FunctionType newType = b.getFunctionType(inputTypes, returnTypes);

                // create new circuit object with empty entry block (to be populated later)
                OperationState opState(op->getLoc(), CircuitOp::getOperationName());
                CircuitOp::build(b, opState, circ.getName(), newType);
                if (circ.no_inline())
                    opState.addAttribute("no_inline", b.getUnitAttr());
                if (circ.no_inline_target())
                    opState.addAttribute("no_inline_target", b.getUnitAttr());
                newOp = circInProg = b.createOperation(opState);
                CircuitOp newCirc = cast<CircuitOp>(newOp);
                newCirc.addEntryBlock();

                // add new block argument values to the local storage
                for (unsigned i = 0; i < newCirc.getNumArguments(); i++) {
                    if (isQData(circ.getArgument(i).getType()))
                        qbmap[circ.getArgument(i)] = newCirc.getArgument(i);
                }

            } else if (isa<quantum::CircuitValueOp>(op)) {
                OperationState opState(op->getLoc(), CircuitValueOp::getOperationName());
                CircuitValueOp::build(b, opState, b.getType<CircType>(),
                                      op->getAttrOfType<FlatSymbolRefAttr>("circref"));
                newOp = b.createOperation(opState);
                op->replaceAllUsesWith(newOp);
                op->erase();

            } else if (isa<quantum::CallCircOp>(op)) {
                newOp = buildCall<CallCircOp>(b, qbmap, op);
            } else if (isa<quantum::ApplyCircOp>(op)) {
                newOp = buildCall<ApplyCircOp>(b, qbmap, op);

            } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                // collect all Qdata ssa values accessed inside to generate return values
                OperationState opState(op->getLoc(), scf::ForOp::getOperationName());
                scf::ForOp::build(b, opState, forOp.lowerBound(), forOp.upperBound(), forOp.step(),
                                  iterArgs);
                newOp = b.createOperation(opState);

                // handle the loop body region, region args already handled in prep function
                BlockAndValueMapping mapping;
                newOp->getRegion(0).front().erase();
                op->getRegion(0).cloneInto(&newOp->getRegion(0), mapping);

                auto resIt = newOp->getResults().begin();
                for (auto arg : uniqueValues) {
                    qbmap[arg] = *resIt++;
                }

                op->erase();

            } else if (isa<scf::IfOp>(op)) {
                // collect all Qdata ssa values accessed inside to generate return values
                Value cond = op->getOperand(0);
                SmallVector<Type, 4> returnTypes;
                for (auto arg : uniqueValues)
                    returnTypes.push_back(convDialectType(b, arg.getType()));

                OperationState opState(op->getLoc(), scf::IfOp::getOperationName());
                scf::IfOp::build(b, opState, returnTypes, cond, true);
                newOp = b.createOperation(opState);


                BlockAndValueMapping mapping;
                newOp->getRegion(0).front().erase();
                op->getRegion(0).cloneInto(&newOp->getRegion(0), mapping);
                newOp->getRegion(1).front().erase();
                op->getRegion(1).cloneInto(&newOp->getRegion(1), mapping);

                auto resIt = newOp->getResults().begin();
                for (auto arg : uniqueValues) {
                    qbmap[arg] = *resIt++;
                }

                op->erase();

            } else if (isa<quantum::TerminatorOp>(op)) {
                newOp = buildTerm<ReturnStateOp>(b, qbmap, op,
                    op->getParentOfType<quantum::CircuitOp>().getArguments());
            } else if (isa<scf::YieldOp>(op)) {
                newOp = buildTerm<scf::YieldOp>(b, qbmap, op, uniqueValues);
            }

            #ifdef DEBUG
                if (newOp) {
                    printf(" -- Building new op...\n  ");
                    newOp->dump();
                } else {
                    printf("\n");
                }
            #endif
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
        for (size_t i = 0; i < idxArr1.size(); i++)
            resultsOp1.push_back(newOp1->getResult(i));
        resultsOp1.push_back(newExtr1.rem());
        for (int i = idxArr1.size(); i < totalNumIdx; i++)
            resultsOp2.push_back(newOp1->getResult(i));
        resultsOp2.push_back(newExtr1.rem());

        // this instruciton erases the given op and replaces all its uses with the provided values
        rewriter.replaceOp(op2, resultsOp2);
        rewriter.replaceOp(op1, resultsOp1);

        newExtr1.verify();
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

        cast<CombineStatOp>(op2).verify();
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
        llvm::SmallVector<std::tuple<size_t, size_t, int64_t>, 4> commonIndices;
        for (size_t i = 0; i < idxArr1.size(); i++) {
            int64_t idx1val = idxArr1[i].dyn_cast<IntegerAttr>().getInt();
            bool match = false;
            for (size_t j = 0; j < idxArr2.size(); j++) {
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
            for (size_t i = 0, j = 0, k = 0; i < extr2.qbs().size(); i ++) {
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
        llvm::SmallVector<std::pair<size_t, size_t>, 4> commonQbs;
        for (size_t i = 0; i < combQbs.size(); i++) {
            for (size_t j = 0; j < extrQbs.size(); j++) {
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
            for (size_t i = 0, j = 0; i < idxArr2.size(); i++) {
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
            for (size_t i = 0, j = 0; i < idxArr1.size(); i++) {
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

/*--- Patterns with dynamic indices ---*/

class ExtractDynCombinePatt : public RewritePattern {
public:
    ExtractDynCombinePatt(PatternBenefit benefit, MLIRContext *context) :
        RewritePattern(CombineDynOp::getOperationName(), benefit, context) {}

    // Match (extract - combine) pattern within the same region
    LogicalResult match(Operation *op2) const override {
        CombineDynOp comb2 = cast<CombineDynOp>(op2);
        Operation *op1 = comb2.reg().getDefiningOp();
        if (!isa_and_nonnull<ExtractOp>(op1) || op1->getParentRegion() != op2->getParentRegion())
            return failure();
        ExtractOp extr1 = cast<ExtractOp>(op1);
        if (extr1.const_idx())
            return failure();

        // only match if all qubits match
        auto extr1QbIt = extr1.qbs().begin();
        for (Value qb2 : comb2.qbs()) {
            if (*extr1QbIt++ != qb2)
                return failure();
        }

        // only match if all indices match
        auto extr1IdxIt = extr1.dyn_idx().begin();
        for (Value idx2 : comb2.dyn_idx()) {
            if (*extr1IdxIt++ != idx2)
                return failure();
        }
        return success();
    }

    void rewrite(Operation *op2, PatternRewriter &rewriter) const override {
        CombineDynOp comb2 = cast<CombineDynOp>(op2);
        Operation *op1 = comb2.reg().getDefiningOp();
        ExtractOp extr1 = cast<ExtractOp>(op1);

        // delete both of them, replacing output register with input register
        rewriter.replaceOp(comb2, {extr1.reg()});
        rewriter.eraseOp(extr1);
    }
};


//===------------------------------------------------------------------------------------------===//
// Quantum Peephole Optimization Patterns
//===------------------------------------------------------------------------------------------===//

// general rewrite pattern that cancels two successive hermitian ops
struct HermitianCancel : public RewritePattern {
    // Constructor: benefit = "how much computation" the transformation saves
    HermitianCancel(PatternBenefit benefit) : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

    // match an op if it has the hermitian trait
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::HermitianTrait>())
            return failure();

        // all quantum data arguments must have originated from a single op of the same type
        Operation *parent = nullptr;
        for (auto arg : op->getOperands()) {
            if (isQSSAData(arg.getType())) {
                if (!parent) {
                    parent = arg.getDefiningOp();
                    if (!parent || parent->getName() != op->getName()
                                || op->getParentRegion() != parent->getParentRegion())
                        return failure();
                } else {
                    if (arg.getDefiningOp() != parent)
                        return failure();
                }
            }
        }
        if (!parent)
            return failure();

        // assume return values consist exclusively of updated qubit/register states
        llvm::SmallVector<Value, 2> replacementValues;
        for (Value arg : parent->getOperands()) {
            if (isQSSAData(arg.getType()))
                replacementValues.push_back(arg);
        }

        rewriter.replaceOp(op, replacementValues);
        rewriter.eraseOp(parent);

        return success();
    }
};

struct AdjointCancelBw : public OpRewritePattern<AdjointOp> {
    // Constructor: benefit = "how much computation" the transformation saves
    AdjointCancelBw(MLIRContext *context) : OpRewritePattern<AdjointOp>(context, /*benefit=*/2) {}

    // Match (op - adjoint) and (heldOp - adjoint) pattern
    LogicalResult match(AdjointOp adj) const override {
        // both quantum data arguments must originate from the same op
        Operation *qbParent = nullptr;
        if (adj.qbs()) {
            qbParent = adj.qbs().getDefiningOp();
            if (!qbParent || (adj.qbs2() && adj.qbs2().getDefiningOp() != qbParent))
                return failure();
            if (adj.getParentRegion() != qbParent->getParentRegion())
                return failure();
        } else {
            return failure();
        }

        // held op argument should be of same op type as qdata parent
        Operation *opParent = adj.heldOp().getDefiningOp();
        if (!opParent || qbParent->getName() != opParent->getName())
            return failure();

        // if any extra args, they need to be the same. possible: rot gates, adj, ctrl
        // opParent has mandatory args only: e.g. rot angle, held ops, ctrl qubits
        // qbParent has mandatory plus the ones already checked above
        if (auto ctrlOp = dyn_cast<ControlOp>(opParent)) {
            ControlOp parentCtrlOp = cast<ControlOp>(qbParent);
            if (ctrlOp.heldOp() != parentCtrlOp.heldOp())
                return failure();
            if (ctrlOp.ctrls() != parentCtrlOp.new_ctrls())
                return failure();
        } else if (auto adjOp = dyn_cast<AdjointOp>(opParent)) {
            AdjointOp parentAdjOp = cast<AdjointOp>(qbParent);
            if (adjOp.heldOp() != parentAdjOp.heldOp())
                return failure();
        } else if (auto rzOp = dyn_cast<RzOp>(opParent)) {
            RzOp parentRzOp = cast<RzOp>(qbParent);
            if (rzOp.phi() != parentRzOp.phi())
                return failure();
        } else if (auto rOp = dyn_cast<ROp>(opParent)) {
            ROp parentROp = cast<ROp>(qbParent);
            if (rOp.phi() != parentROp.phi())
                return failure();
        }

        return success();
    }

    void rewrite(AdjointOp adj, PatternRewriter &rewriter) const override {
        // get op to which this adjoint is the inverse
        Operation *parent = adj.qbs().getDefiningOp();

        // get original state values, qdata args are always last
        SmallVector<Value, 2> origStates;
        if (adj.qbs2())
            origStates.push_back(parent->getOperands().drop_back().back());
        origStates.push_back(parent->getOperands().back());

        // erase adjoint then parent
        rewriter.replaceOp(adj, origStates);
        if (auto ctrl = dyn_cast<ControlOp>(parent))
            rewriter.replaceOp(ctrl, {ctrl.ctrls(), nullptr});
        else
            rewriter.eraseOp(parent);
    }
};

struct AdjointCancelFw : public RewritePattern {
    // Constructor: benefit = "how much computation" the transformation saves
    AdjointCancelFw(PatternBenefit benefit) : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

    // Match (heldOp - adjoint - op) pattern
    LogicalResult match(Operation *op) const override {
        if (!op->hasTrait<OpTrait::HermitianTrait>() && !op->hasTrait<OpTrait::MetaOpTrait>())
            return failure();

        // check if parent is an adjoint op
        // qdata args always last, ctrl might have qdata but no application
        Operation *parent;
        if (op->getNumOperands() && isQSSAData(op->getOperands().back().getType())) {
            if (isa<ControlOp>(op) && op->getNumOperands() < 3)
                return failure();
            parent = op->getOperands().back().getDefiningOp();
            if (!parent || !isa<AdjointOp>(parent)
                        || op->getParentRegion() != parent->getParentRegion())
                return failure();
        } else {
            return failure();
        }
        AdjointOp adj = cast<AdjointOp>(parent);

        // both quantum data arguments must originate from the same op
        if (adj.qbs2() && (op->getNumOperands() < 2 ||
                           !isQSSAData(op->getOperands().drop_back().back().getType()) ||
                           op->getOperands().drop_back().back().getDefiningOp() != parent))
            return failure();

        // held op argument (of adjoint) should be of same op type as this
        Operation *opParent = adj.heldOp().getDefiningOp();
        if (!opParent || op->getName() != opParent->getName())
            return failure();

        // if any extra args, they need to be the same. possible: rot gates, adj, ctrl
        // opParent has mandatory args only: e.g. rot angle, held ops, ctrl qubits
        // (this) op has mandatory plus the ones already checked above
        if (auto ctrlOp = dyn_cast<ControlOp>(op)) {
            ControlOp parentCtrlOp = cast<ControlOp>(opParent);
            if (ctrlOp.heldOp() != parentCtrlOp.heldOp())
                return failure();
            if (ctrlOp.ctrls() != parentCtrlOp.new_ctrls())
                return failure();
        } else if (auto adjOp = dyn_cast<AdjointOp>(op)) {
            AdjointOp parentAdjOp = cast<AdjointOp>(opParent);
            if (adjOp.heldOp() != parentAdjOp.heldOp())
                return failure();
        } else if (auto rzOp = dyn_cast<RzOp>(op)) {
            RzOp parentRzOp = cast<RzOp>(opParent);
            if (rzOp.phi() != parentRzOp.phi())
                return failure();
        } else if (auto rOp = dyn_cast<ROp>(op)) {
            ROp parentROp = cast<ROp>(opParent);
            if (rOp.phi() != parentROp.phi())
                return failure();
        }

        return success();
    }

    void rewrite(Operation *op, PatternRewriter &rewriter) const override {
        // get the adjoint parent op
        AdjointOp adj = cast<AdjointOp>(op->getOperands().back().getDefiningOp());

        // get original state values
        SmallVector<Value, 3> origStates;
        if (auto ctrl = dyn_cast<ControlOp>(op))
            origStates.push_back(ctrl.ctrls());
        if (adj.qbs2())
            origStates.push_back(adj.qbs2());
        origStates.push_back(adj.qbs());

        // erase this op then parent adjoint
        rewriter.replaceOp(op, origStates);
        rewriter.eraseOp(adj);
    }
};

struct CircuitCancelBw : public OpRewritePattern<ApplyCircOp> {
    // Constructor: benefit = "how much computation" the transformation saves
    CircuitCancelBw(MLIRContext *context) : OpRewritePattern<ApplyCircOp>(context, /*benefit=*/5) {}

    // Match (circuit - getval - adjoint - apply) and (circuit - call - apply) pattern
    LogicalResult match(ApplyCircOp apply) const override {
        SmallVector<Value, 4> nonQArgs;

        // all quantum data arguments must originate from the same call op
        Operation *parent = nullptr;
        for (auto arg : apply.args()) {
            if (isQSSAData(arg.getType())) {
                if (!parent) {
                    parent = arg.getDefiningOp();
                    if (!parent || !isa<CallCircOp>(parent)
                                || apply.getParentRegion() != parent->getParentRegion())
                        return failure();
                } else if (arg.getDefiningOp() != parent) {
                    return failure();
                }
            } else {
                nonQArgs.push_back(arg);
            }
        }
        if (!parent)
            return failure();
        CallCircOp call = cast<CallCircOp>(parent);

        // check both ops reference the same circuit, with an intermediate adjoint
        if (auto adj = dyn_cast_or_null<AdjointOp>(apply.circval().getDefiningOp())) {
            if (auto getval = dyn_cast_or_null<CircuitValueOp>(adj.heldOp().getDefiningOp())) {
                if (getval.circref() != call.circref())
                    return failure();
            } else {
                return failure();
            }
        } else {
            return failure();
        }

        // finally remaining arguments should be identitical
        auto nonQArgsIt = nonQArgs.begin();
        for (auto arg : call.args()) {
            if (!isQSSAData(arg.getType()))
                if (arg != *nonQArgsIt++)
                    return failure();
        }

        return success();
    }

    void rewrite(ApplyCircOp apply, PatternRewriter &rewriter) const override {
        // get call op
        Operation *callOp = nullptr;
        auto argsIt = apply.args().begin();
        while (!callOp) {
            if (isQSSAData((*argsIt).getType()))
                callOp = (*argsIt).getDefiningOp();
            argsIt++;
        }

        // get original state values
        SmallVector<Value, 4> origStates;
        for (auto arg : callOp->getOperands())
            if (isQSSAData(arg.getType()))
                origStates.push_back(arg);

        // erase call then apply
        rewriter.replaceOp(apply, origStates);
        rewriter.eraseOp(callOp);
    }
};

struct CircuitCancelFw : public OpRewritePattern<CallCircOp> {
    // Constructor: benefit = "how much computation" the transformation saves
    CircuitCancelFw(MLIRContext *context) : OpRewritePattern<CallCircOp>(context, /*benefit=*/5) {}

    // Match (circuit - getval - adjoint - apply - call) and (circuit - call) pattern
    LogicalResult match(CallCircOp call) const override {
        SmallVector<Value, 4> nonQArgs;

        // all quantum data arguments must originate from the same apply op
        Operation *parent = nullptr;
        for (auto arg : call.args()) {
            if (isQSSAData(arg.getType())) {
                if (!parent) {
                    parent = arg.getDefiningOp();
                    if (!parent || !isa<ApplyCircOp>(parent)
                                || call.getParentRegion() != parent->getParentRegion())
                        return failure();
                } else if (arg.getDefiningOp() != parent) {
                    return failure();
                }
            } else {
                nonQArgs.push_back(arg);
            }
        }
        if (!parent)
            return failure();
        ApplyCircOp apply = cast<ApplyCircOp>(parent);

        // check both ops reference the same circuit, with an intermediate adjoint
        if (auto adj = dyn_cast_or_null<AdjointOp>(apply.circval().getDefiningOp())) {
            if (auto getval = dyn_cast_or_null<CircuitValueOp>(adj.heldOp().getDefiningOp())) {
                if (getval.circref() != call.circref())
                    return failure();
            } else {
                return failure();
            }
        } else {
            return failure();
        }

        // finally remaining arguments should be identitical
        auto nonQArgsIt = nonQArgs.begin();
        for (auto arg : apply.args()) {
            if (!isQSSAData(arg.getType()))
                if (arg != *nonQArgsIt++)
                    return failure();
        }

        return success();
    }

    void rewrite(CallCircOp call, PatternRewriter &rewriter) const override {
        // get call op
        Operation *applyOp = nullptr;
        auto argsIt = call.args().begin();
        while (!applyOp) {
            if (isQSSAData((*argsIt).getType()))
                applyOp = (*argsIt).getDefiningOp();
            argsIt++;
        }

        // get original state values
        SmallVector<Value, 4> origStates;
        for (auto arg : applyOp->getOperands())
            if (isQSSAData(arg.getType()))
                origStates.push_back(arg);

        // erase call then apply
        rewriter.replaceOp(call, origStates);
        rewriter.eraseOp(applyOp);
    }
};

// two successive rotation gates can be combined together and their angles added
template<class R> struct FoldRotation : public OpRewritePattern<R> {
    // Constructor: benefit = "how much computation" the transformation saves
    FoldRotation(MLIRContext *context) : OpRewritePattern<R>(context, /*benefit=*/1) {}

    // match (rz - rz) pattern
    LogicalResult matchAndRewrite(R op2, PatternRewriter &rewriter) const override {
        if (!op2.qbs())
            return failure();

        R op1 = dyn_cast_or_null<R>(op2.qbs().getDefiningOp());
        if (!op1)
            return failure();

        if (op1.getParentRegion() != op2.getParentRegion())
            return failure();

        rewriter.setInsertionPoint(op1);
        OperationState addState(op1.getLoc(), AddFOp::getOperationName());
        AddFOp::build(rewriter, addState, op1.phi(), op2.phi());
        Operation *addOp = rewriter.createOperation(addState);

        rewriter.startRootUpdate(op2);
        op2.setOperand(0, addOp->getResult(0)); // phi1 + phi2
        op2.setOperand(1, op1.qbs());
        rewriter.finalizeRootUpdate(op2);
        rewriter.eraseOp(op1);

        return success();
    }
};

// two successive controlled rotation gates can be combined together if they have the same controls
struct FoldControlledRotations : public OpRewritePattern<ControlOp> {
    // Constructor: benefit = "how much computation" the transformation saves
    FoldControlledRotations(MLIRContext *context) :
        OpRewritePattern<ControlOp>(context, /*benefit=*/3) {}

    // match (rz - ctrl... - rz - ctrl...) pattern
    LogicalResult matchAndRewrite(ControlOp ctrl2, PatternRewriter &rewriter) const override {
        // first check that we have two succesive single target control gates
        if (ctrl2.qbs2() || !ctrl2.qbs() || !isa<ControlOp>(ctrl2.qbs().getDefiningOp()))
            return failure();
        ControlOp ctrl1 = cast<ControlOp>(ctrl2.qbs().getDefiningOp());
        if (ctrl1.qbs2() || !ctrl1.qbs() || ctrl1.getParentRegion() != ctrl2.getParentRegion())
            return failure();

        // resolve and compare control chain, also needs to be successive
        SmallVector<ControlOp, 4> ctrlsVec1, ctrlsVec2;
        Operation *rot1 = nullptr, *rot2 = nullptr;
        if (failed(resolveControlChain(ctrl1, rot1, ctrlsVec1)) ||
                failed(resolveControlChain(ctrl2, rot2, ctrlsVec2)))
            return failure();
        if (rot1->getParentRegion() != rot2->getParentRegion())
            return failure();

        // check for now only one use on rot1
        if (!rot1->getResult(0).hasOneUse()) {
            llvm::errs() << "Rotation with more than one use!";
            return failure();
        }

        // ctrl arguments to each controlOp in the second chain need to come
        // from the corresponding op in the first chain
        auto ctrls1It = ctrlsVec1.begin();
        for (auto cop2 : ctrlsVec2)
            if (cop2.qbs().getDefiningOp() != (*ctrls1It++).getOperation())
                return failure();

        // combine rotations, reuse first chain
        Value phi1 = rot1->getOperand(0);
        Value phi2 = rot2->getOperand(0);

        rewriter.setInsertionPoint(rot1);
        OperationState addState(rot1->getLoc(), AddFOp::getOperationName());
        AddFOp::build(rewriter, addState, phi1, phi2);
        Operation *addOp = rewriter.createOperation(addState);

        rewriter.startRootUpdate(rot1);
        rot1->setOperand(0, addOp->getResult(0)); // phi1 + phi2
        rewriter.finalizeRootUpdate(rot1);

        // replace all of the second chain with return values of the first
        ctrls1It = ctrlsVec1.begin();
        for (auto cop2 : llvm::make_early_inc_range(ctrlsVec2))
            rewriter.replaceOp(cop2, (*ctrls1It++).getResults());

        if (rot2->getResult(0).getUses().empty())
            rewriter.eraseOp(rot2);

        return success();
    }

private:
    static LogicalResult resolveControlChain(Operation *op, Operation* &rot,
                                             SmallVectorImpl<ControlOp> &ctrls) {
        rot = nullptr;
        while (!rot) {
            if (auto ctrl = dyn_cast<ControlOp>(op)) {
                op = ctrl.heldOp().getDefiningOp();
                ctrls.push_back(ctrl);
            } else if (isa<RzOp>(op)) {
                rot = op;
            } else if (isa<ROp>(op)) {
                rot = op;
            } else {
                return failure();
            }
        }
        return success();
    }
};

// remove circuit adjoints that have no more uses
struct FoldDanglingControl : public OpRewritePattern<ControlOp> {
    // Constructor: benefit = "how much computation" the transformation saves
    FoldDanglingControl(MLIRContext *context) : OpRewritePattern<ControlOp>(context, /*benefit=*/1) {}

    // Matching logic
    LogicalResult matchAndRewrite(ControlOp ctrl, PatternRewriter &rewriter) const override {
        if (!ctrl.qbs() && ctrl.res().getUses().empty()) {
            rewriter.replaceOp(ctrl, {ctrl.ctrls(), nullptr});
            return success();
        }

        return failure();
    }
};

// remove circuit adjoints that have no more uses
struct FoldDanglingAdjoint : public OpRewritePattern<AdjointOp> {
    // Constructor: benefit = "how much computation" the transformation saves
    FoldDanglingAdjoint(MLIRContext *context) : OpRewritePattern<AdjointOp>(context, /*benefit=*/1) {}

    // Matching logic
    LogicalResult matchAndRewrite(AdjointOp adj, PatternRewriter &rewriter) const override {
        if (!adj.qbs() && adj.res().getUses().empty()) {
            rewriter.eraseOp(adj);
            return success();
        }

        return failure();
    }
};

// change apply to call if it directly succeeds a getval op
struct FoldApply : public OpRewritePattern<ApplyCircOp> {
    FoldApply(MLIRContext *context) : OpRewritePattern<ApplyCircOp>(context, /*benefit*/1) {}

    // Matching logic
    LogicalResult matchAndRewrite(ApplyCircOp apply, PatternRewriter &rewriter) const override {
        if (auto valOp = dyn_cast<CircuitValueOp>(apply.circval().getDefiningOp())) {
            OperationState callState(apply.getLoc(), CallCircOp::getOperationName());
            CallCircOp::build(rewriter, callState, apply.getResultTypes(), valOp.circrefAttr(),
                              apply.getOperands().drop_front());
            Operation *newOp = rewriter.createOperation(callState);
            for (auto attr : apply.getAttrs())
                newOp->setAttr(attr.first, attr.second);
            rewriter.replaceOp(apply, newOp->getResults());
            return success();
        }

        return failure();
    }
};

// remove circuit value op if its result has no uses
struct FoldCircVal : public OpRewritePattern<CircuitValueOp> {
    FoldCircVal(MLIRContext *context) : OpRewritePattern<CircuitValueOp>(context, /*benefit*/1) {}

    // Matching logic
    LogicalResult matchAndRewrite(CircuitValueOp cval, PatternRewriter &rewriter) const override {
        if (cval.circval().getUses().empty()) {
            rewriter.eraseOp(cval);
            return success();
        }

        return failure();
    }
};

struct QuantumGateOptimizationPass : public OperationPass<ModuleOp> {
    QuantumGateOptimizationPass()
        : OperationPass<ModuleOp>(TypeID::get<QuantumGateOptimizationPass>()) {}
    QuantumGateOptimizationPass(const QuantumGateOptimizationPass &)
        : OperationPass<ModuleOp>(TypeID::get<QuantumGateOptimizationPass>()) {}

    StringRef getName() const override {
        return "QuantumGateOptimizationPass";
    }

    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<QuantumGateOptimizationPass>(*this);
    }

    void runOnOperation() override {
        MLIRContext &context = getContext();

        OwningRewritePatternList patterns;
        patterns.insert<HermitianCancel>(2);
        patterns.insert<AdjointCancelBw>(&context);
        patterns.insert<AdjointCancelFw>(2);
        patterns.insert<CircuitCancelBw>(&context);
        patterns.insert<CircuitCancelFw>(&context);
        patterns.insert<FoldRotation<RzOp>>(&context);
        patterns.insert<FoldRotation<ROp>>(&context);
        patterns.insert<FoldControlledRotations>(&context);

        applyPatternsAndFoldGreedily(getOperation(), patterns);
    }
};

struct StripUnusedCircuitPass : public OperationPass<ModuleOp> {
    StripUnusedCircuitPass()
        : OperationPass<ModuleOp>(TypeID::get<StripUnusedCircuitPass>()) {}
    StripUnusedCircuitPass(const StripUnusedCircuitPass &)
        : OperationPass<ModuleOp>(TypeID::get<StripUnusedCircuitPass>()) {}

    StringRef getName() const override {
        return "StripUnusedCircuitPass";
    }

    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<StripUnusedCircuitPass>(*this);
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder b(module.getContext());

        for (auto &block : module.getBodyRegion()) {
            for (auto &op : llvm::make_early_inc_range(block)) {
                if (auto circ = dyn_cast<CircuitOp>(op)) {
                    if (circ.symbolKnownUseEmpty(module) && circ.getName() != "mlir_main"
                                                         && circ.getName() != "main")
                        circ.erase();
                }
            }
        }

    }
};

struct LowerControlledCircuitsPass : public OperationPass<ModuleOp> {
    LowerControlledCircuitsPass()
        : OperationPass<ModuleOp>(TypeID::get<LowerControlledCircuitsPass>()) {}
    LowerControlledCircuitsPass(const LowerControlledCircuitsPass &)
        : OperationPass<ModuleOp>(TypeID::get<LowerControlledCircuitsPass>()) {}

    StringRef getName() const override {
        return "LowerControlledCircuitsPass";
    }

    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<LowerControlledCircuitsPass>(*this);
    }

private:
    std::unordered_set<std::string> alreadyTraversed;
    Operation *main;
    SmallVector<Value, 4> currentCtrls;

    void makeControlled(OpBuilder &b, Operation* &op, Value &ctrls) {
        assert(ctrls.getType().isa<QstateType>() || ctrls.getType().isa<RstateType>());

        if (op->getAttr("compute") || op->getAttr("uncompute"))
            return;

        llvm::Optional<int> nctrls;
        if (ctrls.getType().isa<QstateType>())
            nctrls = 1;
        else
            nctrls = ctrls.getType().cast<RstateType>().getNumQubits();

        // create op on hold
        Operation *opOnHold;
        Value qbs, qbs2(nullptr);
        OperationState opState(op->getLoc(), op->getName());
        b.setInsertionPoint(op);
        if (auto h = dyn_cast<HOp>(op)) {
            if (!h.qbs())
                return;
            qbs = h.qbs();
            HOp::build(b, opState, {b.getType<U1Type>()}, {}, op->getAttrs());
            opOnHold = b.createOperation(opState);
        } else if (auto x = dyn_cast<XOp>(op)) {
            if (!x.qbs())
                return;
            qbs = x.qbs();
            XOp::build(b, opState, {b.getType<U1Type>()}, {}, op->getAttrs());
            opOnHold = b.createOperation(opState);
        } else if (auto rz = dyn_cast<RzOp>(op)) {
            if (!rz.qbs())
                return;
            qbs = rz.qbs();
            RzOp::build(b, opState, {b.getType<U1Type>()}, {rz.phi()}, op->getAttrs());
            opOnHold = b.createOperation(opState);
        } else if (auto r = dyn_cast<ROp>(op)) {
            if (!r.qbs())
                return;
            qbs = r.qbs();
            ROp::build(b, opState, {b.getType<U1Type>()}, {r.phi()}, op->getAttrs());
            opOnHold = b.createOperation(opState);
        } else if (auto cx = dyn_cast<CNotOp>(op)) {
            if (!cx.qbs())
                return;
            qbs = cx.qbs();
            qbs2 = cx.ctrl();
            CNotOp::build(b, opState, {b.getType<U2Type>()}, {}, op->getAttrs());
            opOnHold = b.createOperation(opState);
            opOnHold->setAttr("operand_segment_sizes", b.getI32VectorAttr({0, 0}));
        } else if (auto sw = dyn_cast<SwapOp>(op)) {
            if (!sw.qbs())
                return;
            qbs = sw.qbs();
            qbs2 = sw.qbs2();
            SwapOp::build(b, opState, {b.getType<U2Type>()}, {}, op->getAttrs());
            opOnHold = b.createOperation(opState);
            opOnHold->setAttr("operand_segment_sizes", b.getI32VectorAttr({0, 0}));
        } else if (auto ctrl = dyn_cast<ControlOp>(op)) {
            if (!ctrl.qbs())
                return;

            llvm::Optional<int> newNCtrls;
            Type baseType = ctrl.heldOp().getType();
            if (auto ty = baseType.dyn_cast<COpType>()) {
                newNCtrls = ty.getNumCtrls() && nctrls ? llvm::Optional<int>(*nctrls + *ty.getNumCtrls())
                                        : llvm::None;
                baseType = ty.getBaseType();
            } else {
                newNCtrls = nctrls;
            }

            COpType cType = b.getType<COpType>(newNCtrls, baseType);
            OperationState ctrlState(op->getLoc(), ControlOp::getOperationName());
            ControlOp::build(b, ctrlState, {ctrls.getType(), cType}, {ctrl.heldOp(), ctrls}, {});
            Operation *ctrlOp = b.createOperation(ctrlState);
            ctrlOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, 0, 0}));

            ctrl.setOperand(0, ctrlOp->getResult(1)); // heldOp
            // op = op;
            ctrls = ctrlOp->getResult(0); // new_ctrls
            return;
        } else if (auto adj = dyn_cast<AdjointOp>(op)) {
            if (!adj.qbs())
                return;

            llvm::Optional<int> newNCtrls;
            Type baseType = adj.heldOp().getType();
            if (auto ty = baseType.dyn_cast<COpType>()) {
                newNCtrls = ty.getNumCtrls() && nctrls ? llvm::Optional<int>(*nctrls + *ty.getNumCtrls())
                                        : llvm::None;
                baseType = ty.getBaseType();
            } else {
                newNCtrls = nctrls;
            }

            COpType cType = b.getType<COpType>(newNCtrls, baseType);
            OperationState ctrlState(op->getLoc(), ControlOp::getOperationName());
            ControlOp::build(b, ctrlState, {ctrls.getType(), cType}, {adj.heldOp(), ctrls}, {});
            Operation *ctrlOp = b.createOperation(ctrlState);
            ctrlOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, 0, 0}));

            adj.setOperand(0, ctrlOp->getResult(1)); // heldOp
            // op = op;
            ctrls = ctrlOp->getResult(0); // new_ctrls
            return;
        } else if (auto call = dyn_cast<CallCircOp>(op)) {
            OperationState getvalState(op->getLoc(), CircuitValueOp::getOperationName());
            CircuitValueOp::build(b, getvalState, b.getType<CircType>(), call.circref());
            Operation *getvalOp = b.createOperation(getvalState);

            COpType cType = b.getType<COpType>(nctrls, b.getType<CircType>());
            OperationState ctrlState(op->getLoc(), ControlOp::getOperationName());
            ControlOp::build(b, ctrlState, {ctrls.getType(), cType}, {getvalOp->getResult(0), ctrls}, {});
            Operation *ctrlOp = b.createOperation(ctrlState);
            ctrlOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, 0, 0}));

            SmallVector<Value, 6> operands({ctrlOp->getResult(1)});
            for (auto arg : call.args())
                operands.push_back(arg);
            OperationState applyState(op->getLoc(), ApplyCircOp::getOperationName());
            ApplyCircOp::build(b, applyState, call.getResultTypes(), operands);
            Operation *applyOp = b.createOperation(applyState);

            call.replaceAllUsesWith(applyOp);
            call.erase();
            op = applyOp;
            ctrls = ctrlOp->getResult(0);
            return;
        } else if (auto apply = dyn_cast<ApplyCircOp>(op)) {
            llvm::Optional<int> newNCtrls;
            Type baseType = apply.circval().getType();
            if (auto ty = baseType.dyn_cast<COpType>()) {
                newNCtrls = ty.getNumCtrls() && nctrls ? llvm::Optional<int>(*nctrls + *ty.getNumCtrls())
                                           : llvm::None;
                baseType = ty.getBaseType();
            } else {
                newNCtrls = nctrls;
            }

            COpType cType = b.getType<COpType>(newNCtrls, baseType);
            OperationState ctrlState(op->getLoc(), ControlOp::getOperationName());
            ControlOp::build(b, ctrlState, {ctrls.getType(), cType}, {apply.circval(), ctrls}, {});
            Operation *ctrlOp = b.createOperation(ctrlState);
            ctrlOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, 0, 0}));

            apply.setOperand(0, ctrlOp->getResult(1));
            // op = op;
            ctrls = ctrlOp->getResult(0);
            return;
        } else if (auto m = dyn_cast<MeasurementOp>(op)) {
            llvm_unreachable("Attempting to control non-unitary circuit!");
        } else {
            return;
        }

        // create applied controlled operation
        SmallVector<Value, 4> operands({opOnHold->getResult(0), ctrls});
        SmallVector<Type, 3> resTypes({ctrls.getType()});
        if (qbs2) {
            operands.push_back(qbs2);
            resTypes.push_back(qbs2.getType());
        }
        operands.push_back(qbs);
        resTypes.push_back(qbs.getType());
        OperationState ctrlState(op->getLoc(), ControlOp::getOperationName());
        ControlOp::build(b, ctrlState, resTypes, operands, op->getAttrs());
        Operation *ctrlOp = b.createOperation(ctrlState);
        ctrlOp->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, !!qbs2, 1}));

        op->replaceAllUsesWith(ctrlOp->getResults().drop_front());
        op->erase();
        op = ctrlOp;
        ctrls = ctrlOp->getResult(0);
    }

    Operation* resolveCall(Operation* &op, SmallVectorImpl<ControlOp> &ctrls, bool &foundAdj) {
        Operation *circuit = nullptr;
        while (!circuit) {
            if (auto ctrl = dyn_cast<ControlOp>(op)) {
                op = ctrl.heldOp().getDefiningOp();
                ctrls.push_back(ctrl);
                assert(ctrl.ctrls().getType().isa<QstateType>() && "Only single controls for now!");
            } else if (auto adj = dyn_cast<AdjointOp>(op)) {
                op = adj.heldOp().getDefiningOp();
                foundAdj = true;
            } else if (auto getval = dyn_cast<CircuitValueOp>(op)) {
                circuit = SymbolTable::lookupNearestSymbolFrom(op, getval.circref());
                assert(circuit && "Could not resolve symbol!");
            } else {
                llvm_unreachable("Unknown op in apply chain!");
            }
        }
        return circuit;
    }

    Operation* createForOp(OpBuilder &b, scf::ForOp forOp) {
        SmallVector<Value, 8> iterArgs;
        for (auto arg : forOp.getIterOperands())
            iterArgs.push_back(arg);
        for (auto cqbs : currentCtrls)
            iterArgs.push_back(cqbs);

        b.setInsertionPoint(forOp);
        OperationState forState(forOp.getLoc(), scf::ForOp::getOperationName());
        scf::ForOp::build(b, forState, forOp.lowerBound(), forOp.upperBound(), forOp.step(), iterArgs);
        Operation *newOp = b.createOperation(forState);

        // add body
        newOp->getRegion(0).takeBody(forOp.getLoopBody());
        for (auto cqbs : currentCtrls)
            newOp->getRegion(0).front().insertArgument(newOp->getRegion(0).getNumArguments(), cqbs.getType());

        // replace values
        forOp.replaceAllUsesWith(newOp->getResults().drop_back(currentCtrls.size()));
        forOp.erase();

        int i = 0;
        for (auto cqbs : cast<scf::ForOp>(newOp).getRegionIterArgs().take_back(currentCtrls.size()))
            currentCtrls[i++] = cqbs;
        return newOp;
    }

    Operation* createIfOp(OpBuilder &b, scf::IfOp ifOp) {
        SmallVector<Type, 8> resTypes;
        for (auto ty : ifOp.getResultTypes())
            resTypes.push_back(ty);
        for (auto cqbs : currentCtrls)
            resTypes.push_back(cqbs.getType());

        b.setInsertionPoint(ifOp);
        OperationState ifState(ifOp.getLoc(), scf::IfOp::getOperationName());
        scf::IfOp::build(b, ifState, resTypes, ifOp.condition());
        Operation *newOp = b.createOperation(ifState);

        // add body
        newOp->getRegion(0).takeBody(ifOp.thenRegion());
        newOp->getRegion(1).takeBody(ifOp.elseRegion());

        // replace values
        ifOp.replaceAllUsesWith(newOp->getResults().drop_back(currentCtrls.size()));
        ifOp.erase();

        return newOp;
    }

    void updateTerminator(OpBuilder &b, Operation *term) {
        SmallVector<Value, 8> retVals(term->getOperands());
        for (auto cqbs : currentCtrls)
            retVals.push_back(cqbs);
        b.setInsertionPoint(term);
        OperationState termState(term->getLoc(), scf::YieldOp::getOperationName());
        scf::YieldOp::build(b, termState, retVals);
        b.createOperation(termState);
        term->erase();
    }

    bool willUseControls(Operation *op) {
        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (isa<scf::ForOp>(nestedOp)) {
                        if (willUseControls(&nestedOp))
                            return true;
                    } else if (isa<scf::IfOp>(nestedOp)) {
                        if (willUseControls(&nestedOp))
                            return true;
                    } else if (isa<HOp>(nestedOp) || isa<XOp>(nestedOp) || isa<RzOp>(nestedOp) ||
                            isa<ROp>(nestedOp) || isa<CNotOp>(nestedOp) || isa<SwapOp>(nestedOp) ||
                            isa<ControlOp>(nestedOp) || isa<AdjointOp>(nestedOp) ||
                            isa<CallCircOp>(nestedOp) || isa<ApplyCircOp>(nestedOp)) {
                        if (!nestedOp.getAttr("compute") && !nestedOp.getAttr("uncompute"))
                            return true;
                    }
                }
            }
        }
        return false;
    }

    void propControls(OpBuilder &b, Operation *op) {
        // also need to setup control arguments/returns for if and for operations
        for (auto &region : op->getRegions()) {
            SmallVector<Value, 4> tmp = currentCtrls;
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (auto forOp = dyn_cast<scf::ForOp>(nestedOp)) { // control args fully local
                        if (willUseControls(forOp)) {
                            forOp = cast<scf::ForOp>(createForOp(b, forOp));
                            propControls(b, forOp);
                            updateTerminator(b, forOp.getBody()->getTerminator());
                            currentCtrls = forOp.getResults().take_back(currentCtrls.size());
                        }
                    } else if (auto ifOp = dyn_cast<scf::IfOp>(nestedOp)) { // control args not local but need to reset between regions
                        if (willUseControls(ifOp)) {
                            ifOp = cast<scf::IfOp>(createIfOp(b, ifOp));
                            propControls(b, ifOp);
                            currentCtrls = ifOp.getResults().take_back(currentCtrls.size());
                        }
                    } else {
                        Operation *currOp = &nestedOp;
                        for (size_t i = 0; i < currentCtrls.size(); i++)
                            makeControlled(b, currOp, currentCtrls[i]); // currOp/currCtrls will be updated here
                    }
                }
            }
            if (isa<scf::IfOp>(op)) {
                updateTerminator(b, region.back().getTerminator());
                currentCtrls = tmp;
            }
        }
    }

    Operation* createControlledCircuit(OpBuilder &b, Operation *circ, ApplyCircOp call,
                                       SmallVectorImpl<ControlOp> &ctrlsVec, Operation *valOp) {
        // copy circuit with new name to which controls will be propagated
        StringAttr circName = circ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
        std::string newCircName = llvm::join<ArrayRef<StringRef>>(
            {circName.getValue(), std::to_string(ctrlsVec.size())}, "_lc");
        Operation *newCirc = SymbolTable::lookupNearestSymbolFrom(call, newCircName);
        if (newCirc) {
            auto ftypeIt = cast<CircuitOp>(newCirc).getType().getInputs().take_back(ctrlsVec.size()).begin();
            for (auto ctrl : ctrlsVec)
                assert(ctrl.ctrls().getType() == *ftypeIt++ && "Different types in matched circuit!");
        }

        // if it doesn't already exist, we create it
        if (!newCirc) {
            newCirc = circ->clone();
            CircuitOp newCircOp = cast<CircuitOp>(newCirc);
            newCircOp.setAttr(SymbolTable::getSymbolAttrName(), b.getStringAttr(newCircName));
            auto arrayIn = cast<CircuitOp>(circ).getType().getInputs();
            auto arrayRe = cast<CircuitOp>(circ).getType().getResults();
            SmallVector<Type, 8> argTypes(arrayIn.begin(), arrayIn.end());
            SmallVector<Type, 8> resTypes(arrayRe.begin(), arrayRe.end());
            for (auto ctrl : ctrlsVec) {
                argTypes.push_back(ctrl.ctrls().getType());
                resTypes.push_back(ctrl.ctrls().getType());
                newCircOp.front().insertArgument(newCircOp.front().getNumArguments(),
                    ctrl.ctrls().getType());
            }
            newCircOp.setType(b.getFunctionType(argTypes, resTypes));
            currentCtrls.clear();
            for (auto cqbs : newCircOp.getArguments().take_back(ctrlsVec.size()))
                currentCtrls.push_back(cqbs);
            b.setInsertionPointAfter(circ);
            b.insert(newCirc);

            // propagate controls TODO
            propControls(b, newCirc);

            // update return op
            Operation *term = newCircOp.back().getTerminator();
            assert(isa<ReturnStateOp>(term) && "Not a return op!");
            for (auto cqbs : currentCtrls)
                term->insertOperands(term->getNumOperands(), cqbs);
        }

        // replace call to point to new circuit
        SmallVector<Value, 8> operands(call.getOperands().drop_front());
        SmallVector<Type, 8> resTypes(call.getResultTypes().begin(), call.getResultTypes().end());
        for (auto ctrl : ctrlsVec) {
            operands.push_back(ctrl.ctrls());
            resTypes.push_back(ctrl.ctrls().getType());
        }
        b.setInsertionPoint(call);
        OperationState callState(call.getLoc(), CallCircOp::getOperationName());
        CallCircOp::build(b, callState, resTypes, newCircName, operands);
        Operation *newCallOp = b.createOperation(callState);
        call.replaceAllUsesWith(newCallOp->getResults().drop_back(ctrlsVec.size()));
        call.erase();

        // erase control ops and replace their returned ctrl states
        auto newCtrlsIt = newCallOp->getResults().take_back(ctrlsVec.size()).begin();
        for (auto ctrl : ctrlsVec) {
            ctrl.new_ctrls().replaceAllUsesWith(*newCtrlsIt++);
            ctrl.erase();
        }

        // (sketch) need to be careful that we are not moving the call past uses of the values it defines
        //assert(ctrlsVec.size() == 1 && "Only handle size 1 chains right now!");
        Operation *comb = nullptr;
        for (auto res : newCallOp->getResults()) {
            if (res.getType().isa<QstateType>()) {
                for(auto user : res.getUsers()) {
                    if (isa<CombineStatOp>(user) || isa<CombineDynOp>(user)) {
                        assert(!comb && "Multiple combines of the same value!");
                        comb = user;
                    }
                }
            }
        }
        if (comb)
            comb->moveAfter(newCallOp);

        assert(isa<CircuitValueOp>(valOp) && "Not a circuit value op!");
        if (valOp->getResult(0).getUses().empty())
            valOp->erase();

        assert(newCirc && "Did not obtain controlled circuit!");
        return newCirc;
    }

    void walkCallTree(OpBuilder &b, Operation *op) {
        if (auto circ = dyn_cast<CircuitOp>(op))
            if (alreadyTraversed.count(circ.getName().str()))
                return;

        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (auto call = dyn_cast<CallCircOp>(nestedOp)) {
                        Operation *calledCirc = SymbolTable::lookupNearestSymbolFrom(
                            call, call.circref());
                        assert(calledCirc && "Unresolved direct circuit call!");
                        walkCallTree(b, calledCirc);
                    } else if (auto apply = dyn_cast<ApplyCircOp>(nestedOp)) {
                        Operation *valOp = apply.circval().getDefiningOp(); // will be updated to CircValOp by resolveCall
                        SmallVector<ControlOp, 4> ctrlsVec;
                        bool foundAdj = false;
                        Operation *calledCirc = resolveCall(valOp, ctrlsVec, foundAdj);
                        assert(calledCirc && "Unresolved indirect circuit call!");
                        if (ctrlsVec.size() && !foundAdj)
                            walkCallTree(b, createControlledCircuit(b, calledCirc, apply, ctrlsVec, valOp));
                        else
                            walkCallTree(b, calledCirc);
                    } else if (isa<scf::ForOp>(nestedOp) || isa<scf::IfOp>(nestedOp)) {
                        walkCallTree(b, &nestedOp);
                    }
                }
            }
        }

        if (auto circ = dyn_cast<CircuitOp>(op))
            alreadyTraversed.insert(circ.getName().str());
    }

public:
    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder b(module.getContext());

        main = module.lookupSymbol("mlir_main");
        assert(main && "Need circuit entry point!");
        walkCallTree(b, main);
    }
};
} // end namespace

// Pass creation functions declared in Passes.h
std::unique_ptr<Pass> quantum::createMemToValPass() {
    return std::make_unique<MemToValPass>();
}

std::unique_ptr<Pass> quantum::createQuantumGateOptimizationPass() {
    return std::make_unique<QuantumGateOptimizationPass>();
}

std::unique_ptr<Pass> quantum::createStripUnusedCircuitPass() {
    return std::make_unique<StripUnusedCircuitPass>();
}

std::unique_ptr<Pass> quantum::createLowerControlledCircuitsPass() {
    return std::make_unique<LowerControlledCircuitsPass>();
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

void CombineDynOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                               MLIRContext *context) {
    patterns.insert<ExtractDynCombinePatt>(2, context);
}

void ControlOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context) {
    patterns.insert<FoldDanglingControl>(context);
}


void AdjointOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context) {
    patterns.insert<FoldDanglingAdjoint>(context);
}

void ApplyCircOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                              MLIRContext *context) {
    patterns.insert<FoldApply>(context);
}

void CircuitValueOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                                 MLIRContext *context) {
    patterns.insert<FoldCircVal>(context);
}
