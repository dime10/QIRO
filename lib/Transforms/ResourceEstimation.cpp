#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"

#include "QuantumDialect.h"
#include "QuantumSSADialect.h"
#include "Passes.h"

#include <unordered_map>
#include <unordered_set>
#include <string>

using namespace mlir;
using namespace mlir::quantumssa;


namespace {

bool isGate(Operation *op) {
    if (!isa<QuantumSSADialect>(op->getDialect()))
        return false;

    return isa<HOp>(op) || isa<XOp>(op) || isa<RzOp>(op) || isa<ROp>(op) || isa<CNotOp>(op) || isa<SwapOp>(op);
}

struct ResourceCounterPass : public OperationPass<ModuleOp> {
    ResourceCounterPass() : OperationPass<ModuleOp>(TypeID::get<ResourceCounterPass>()) {}
    ResourceCounterPass(const ResourceCounterPass &) :
        OperationPass<ModuleOp>(TypeID::get<ResourceCounterPass>()) {}

    StringRef getName() const override {
        return "ResourceCounterPass";
    }

    std::unique_ptr<Pass> clonePass() const override {
        return std::make_unique<ResourceCounterPass>(*this);
    }

private:
    std::unordered_set<std::string> alreadyBuilt;
    Operation *main;
    Value Rcounter;
    Value Tcounter;
    Value const0;
    Value const1;
    Value const2;
    Value const3;
    Value const5;
    Value const7;
    Value const14;
    int nrec;

    void initialize(OpBuilder &b, CircuitOp circ) {
        Location loc = circ.front().front().getLoc();
        b.setInsertionPointToStart(&circ.front());

        OperationState countState(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, countState, b.getI64IntegerAttr(0));
        Rcounter = b.createOperation(countState)->getResult(0);
        Tcounter = Rcounter;

        OperationState const0State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const0State, b.getI64IntegerAttr(0));
        const0 = b.createOperation(const0State)->getResult(0);

        OperationState const1State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const1State, b.getI64IntegerAttr(1));
        const1 = b.createOperation(const1State)->getResult(0);

        OperationState const2State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const2State, b.getI64IntegerAttr(2));
        const2 = b.createOperation(const2State)->getResult(0);

        OperationState const3State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const3State, b.getI64IntegerAttr(3));
        const3 = b.createOperation(const3State)->getResult(0);

        OperationState const5State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const5State, b.getI64IntegerAttr(5));
        const5 = b.createOperation(const5State)->getResult(0);

        OperationState const7State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const7State, b.getI64IntegerAttr(7));
        const7 = b.createOperation(const7State)->getResult(0);

        OperationState const14State(loc, ConstantOp::getOperationName());
        ConstantOp::build(b, const14State, b.getI64IntegerAttr(14));
        const14 = b.createOperation(const14State)->getResult(0);
    }

    void genCounterInc(OpBuilder &b, Operation *op, int64_t n) {
        Value *incR;
        Value *incT;

        if (isa<ROp>(op)) {
            switch (n) {
                case 0 : incR = &const1; incT = &const0; break;
                case 1 : incR = &const3; incT = &const0; break;
                case 2 : incR = &const5; incT = &const14; break;
                default : llvm_unreachable("Not yet implemented other increment values!");
            }
        } else if (isa<HOp>(op)) {
            switch (n) {
                case 1 : incR = &const2; incT = &const0; break;
                default : llvm_unreachable("Not yet implemented other increment values!");
            }
        } else if (isa<SwapOp>(op)) {
            switch (n) {
                case 1 : incR = &const0; incT = &const7; break;
                default : llvm_unreachable("Not yet implemented other increment values!");
            }
        } else {
            llvm_unreachable("Not yet implemented increments for other ops!");
        }

        b.setInsertionPoint(op);
        OperationState addState(op->getLoc(), AddIOp::getOperationName());
        AddIOp::build(b, addState, Rcounter, *incR);
        Rcounter = b.createOperation(addState)->getResult(0);
        addState = OperationState(op->getLoc(), AddIOp::getOperationName());
        AddIOp::build(b, addState, Tcounter, *incT);
        Tcounter = b.createOperation(addState)->getResult(0);
    }

    void stripAdjoint(OpBuilder &b, Operation *adjOp) {
        AdjointOp adj = cast<AdjointOp>(adjOp);

        if (!adj.qbs()) { // intermediate adjoint, pass through value
            adj.res().replaceAllUsesWith(adj.heldOp());
            adj.erase();
        } else { // gate application adjoint
            Operation *gate = adj.heldOp().getDefiningOp();
            assert(isa<HOp>(gate) || isa<XOp>(gate) || isa<RzOp>(gate) || isa<ROp>(gate) || isa<CNotOp>(gate) || isa<SwapOp>(gate) || isa<ControlOp>(gate));

            // Possible gate operations: H, X, Rz, R, CX, SWAP, ctrl (adj taken care of by if)
            // We cannot move the arguments provided to the adjoint to the gate,
            // as it might be in a different region and its value might be used elsewhere.
            // Need to keep the gate information here at the adjoint "call site".
            SmallVector<Value, 4> args;
            SmallVector<Type, 3> resTypes;
            args.append(gate->operand_begin(), gate->operand_end());
            args.append(adj.operand_begin()+1, adj.operand_end());
            resTypes.append(gate->result_type_begin(), gate->result_type_end()-1);
            resTypes.append(adj.result_type_begin(), adj.result_type_end());

            b.setInsertionPoint(adj);
            Operation *newGate = Operation::create(adj.getLoc(), gate->getName(), resTypes, args, gate->getAttrs());
            b.insert(newGate);

            // add segment sizes
            bool q2 = !!adj.qbs2();
            if (isa<CNotOp>(gate) || isa<SwapOp>(gate))
                newGate->setAttr("operand_segment_sizes", b.getI32VectorAttr({q2, 1}));
            else if (isa<ControlOp>(gate))
                newGate->setAttr("operand_segment_sizes", b.getI32VectorAttr({1, 1, q2, 1}));

            // remove adjoint
            if (q2)
                adj.new_qbs2().replaceAllUsesWith(newGate->getResults().drop_back().back());
            adj.res().replaceAllUsesWith(newGate->getResults().back());
            adj.erase();
        }
    }

    void stripControl(OpBuilder &b, Operation *ctrlOp) {
        ControlOp ctrl = cast<ControlOp>(ctrlOp);
        assert(ctrl.ctrls().getType().isa<QstateType>() && "Register ctrls not yet implemented!");
        IntegerAttr ctrlAttr = ctrl.getAttrOfType<IntegerAttr>("_num_ctrls");

        if (!ctrl.qbs()) { // intermediate control
            // pass through value, ammend num of control qubits attribute (will be used later)
            for (auto op : ctrl.res().getUsers()) {
                int64_t newCtrlCount = 1;
                if (ctrlAttr)
                    newCtrlCount += ctrlAttr.getInt();
                IntegerAttr opAttr = op->getAttrOfType<IntegerAttr>("_num_ctrls");
                if (opAttr)
                    newCtrlCount += opAttr.getInt();
                op->setAttr("_num_ctrls", b.getI64IntegerAttr(newCtrlCount));
            }
            ctrl.new_ctrls().replaceAllUsesWith(ctrl.ctrls());
            ctrl.res().replaceAllUsesWith(ctrl.heldOp());
            ctrl.erase();
        } else { // gate application control
            Operation *gate = ctrl.heldOp().getDefiningOp();
            assert(isa<HOp>(gate) || isa<XOp>(gate) || isa<RzOp>(gate) || isa<ROp>(gate) || isa<CNotOp>(gate) || isa<SwapOp>(gate));

            // Possible gate operations: H, X, Rz, R, CX, SWAP (ctrl taken care of by if)
            // We cannot move the arguments provided to the control to the gate,
            // as it might be in a different region and its value might be used elsewhere.
            // Need to keep the gate information here at the control "call site".
            SmallVector<Value, 4> args;
            SmallVector<Type, 3> resTypes;
            args.append(gate->operand_begin(), gate->operand_end());
            args.append(ctrl.operand_begin()+2, ctrl.operand_end());
            resTypes.append(gate->result_type_begin(), gate->result_type_end()-1);
            resTypes.append(ctrl.result_type_begin()+1, ctrl.result_type_end());

            b.setInsertionPoint(ctrl);
            Operation *newGate = Operation::create(ctrl.getLoc(), gate->getName(), resTypes, args, {});
            b.insert(newGate);

            // add segment sizes
            bool q2 = !!ctrl.qbs2();
            if (isa<CNotOp>(gate) || isa<SwapOp>(gate))
                newGate->setAttr("operand_segment_sizes", b.getI32VectorAttr({q2, 1}));

            // amend number of control qubits attribute
            int64_t newCtrlCount = 1;
            if (ctrlAttr)
                newCtrlCount += ctrlAttr.getInt();
            newGate->setAttr("_num_ctrls", b.getI64IntegerAttr(newCtrlCount));

            // remove control
            ctrl.new_ctrls().replaceAllUsesWith(ctrl.ctrls());
            if (q2)
                ctrl.new_qbs2().replaceAllUsesWith(newGate->getResults().drop_back().back());
            ctrl.res().replaceAllUsesWith(newGate->getResults().back());
            ctrl.erase();
        }
    }

    void walkAdj(OpBuilder &b, Operation *op) {
        // walk nested operations
        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (isa<AdjointOp>(nestedOp)) {
                        stripAdjoint(b, &nestedOp);
                    } else if (isa<scf::ForOp>(nestedOp) || isa<scf::IfOp>(nestedOp)) {
                        walkAdj(b, &nestedOp);
                    }
                }
            }
        }
    }

    void walkCtrl(OpBuilder &b, Operation *op) {
        // walk nested operations
        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (isa<ControlOp>(nestedOp)) {
                        stripControl(b, &nestedOp);
                    } else if (isa<scf::ForOp>(nestedOp) || isa<scf::IfOp>(nestedOp)) {
                        walkCtrl(b, &nestedOp);
                    }
                }
            }
        }
    }

    void updateControlAttr(OpBuilder &b, Operation *gate, int64_t nctrl) {
        // no need to control gates in compute/uncompute sections
        if (gate->getAttr("compute") || gate->getAttr("uncompute"))
            return;

        IntegerAttr attr = gate->getAttrOfType<IntegerAttr>("_num_ctrls");
        if (attr)
            nctrl += attr.getInt();
        gate->setAttr("_num_ctrls", b.getI64IntegerAttr(nctrl));
    }

    void propControls(OpBuilder &b, Operation *op, int64_t nctrl) {
        // walk nested operations
        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (isGate(&nestedOp))
                        updateControlAttr(b, &nestedOp, nctrl);
                    else if (isa<scf::ForOp>(nestedOp) || isa<scf::IfOp>(nestedOp))
                        propControls(b, &nestedOp, nctrl);
                }
            }
        }
    }

    Operation* createControlledCircuit(OpBuilder &b, Operation *circ, Operation *call, int64_t nctrl) {
        // copy circuit with new name to which controls will be propagated
        StringAttr circName = circ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
        std::string newCircName = llvm::join<ArrayRef<StringRef>>(
            {circName.getValue(), std::to_string(nctrl)}, "_C");
        Operation *newCirc = SymbolTable::lookupNearestSymbolFrom(call, newCircName);
        if (!newCirc) {
            newCirc = circ->clone();
            newCirc->setAttr(SymbolTable::getSymbolAttrName(), b.getStringAttr(newCircName));
            b.setInsertionPointAfter(circ);
            b.insert(newCirc);

            // propagate controls
            propControls(b, newCirc, nctrl);
        }

        // update call
        ValueRange operands = call->getOperands();
        if (isa<ApplyCircOp>(call))
            operands = operands.drop_front();
        b.setInsertionPoint(call);
        OperationState callState(call->getLoc(), CallCircOp::getOperationName());
        CallCircOp::build(b, callState, call->getResultTypes(), newCircName, operands);
        Operation *newCallOp = b.createOperation(callState);
        call->replaceAllUsesWith(newCallOp);
        call->erase();

        return newCirc;
    }

    void walkCallTree(OpBuilder &b, Operation *op, int64_t nctrl) {
        nrec++;
        if (auto circ = dyn_cast<CircuitOp>(op)) {
            if (alreadyBuilt.count(circ.getName().str())) {
                return;
            }
        }

        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (auto call = dyn_cast<CallCircOp>(nestedOp)) {
                        Operation *calledCirc = SymbolTable::lookupNearestSymbolFrom(call, call.circref());


                        int64_t new_nctrl = nctrl;
                        if (call.getAttr("compute") || call.getAttr("uncompute"))
                            new_nctrl = 0;
                        if (auto attr = call.getAttrOfType<IntegerAttr>("_num_ctrls"))
                            new_nctrl += attr.getInt();
                        if (new_nctrl)
                            walkCallTree(b, createControlledCircuit(b, calledCirc, call, new_nctrl), new_nctrl);
                        else
                            walkCallTree(b, calledCirc, new_nctrl);
                    } else if (auto apply = dyn_cast<ApplyCircOp>(nestedOp)) {
                        StringRef cref = cast<CircuitValueOp>(apply.circval().getDefiningOp()).circref();
                        Operation *calledCirc = SymbolTable::lookupNearestSymbolFrom(apply, cref);
                        int64_t new_nctrl = nctrl;
                        if (apply.getAttr("compute") || apply.getAttr("uncompute"))
                            new_nctrl = 0;
                        if (auto attr = apply.getAttrOfType<IntegerAttr>("_num_ctrls"))
                            new_nctrl += attr.getInt();
                        if (new_nctrl)
                            walkCallTree(b, createControlledCircuit(b, calledCirc, apply, new_nctrl), new_nctrl);
                        else
                            walkCallTree(b, calledCirc, new_nctrl);
                    } else if (isa<scf::ForOp>(nestedOp) || isa<scf::IfOp>(nestedOp))
                        walkCallTree(b, &nestedOp, nctrl);
                }
            }
        }

        if (auto circ = dyn_cast<CircuitOp>(op))
            alreadyBuilt.insert(circ.getName().str());
    }

    void convertGates(OpBuilder &b, Operation *gate) {
        int64_t nctrl = 0;
        if (auto attr = gate->getAttrOfType<IntegerAttr>("_num_ctrls"))
            nctrl += attr.getInt();

        bool noTarget = false;
        // for now delete all quantum operations, return/yield operation updated with some input
        if (auto h = dyn_cast<HOp>(gate)) {
            if (h.qbs())
                h.res().replaceAllUsesWith(h.qbs());
            else
                noTarget = true;
            assert(h.res().getUses().empty() && "still has uses");
        } else if (auto x = dyn_cast<XOp>(gate)) {
            if (x.qbs())
                x.res().replaceAllUsesWith(x.qbs());
            else
                noTarget = true;
            assert(x.res().getUses().empty() && "still has uses");
        } else if (auto rz = dyn_cast<RzOp>(gate)) {
            if (rz.qbs())
                rz.res().replaceAllUsesWith(rz.qbs());
            else
                noTarget = true;
            assert(rz.res().getUses().empty() && "still has uses");
        } else if (auto r = dyn_cast<ROp>(gate)) {
            if (r.qbs())
                r.res().replaceAllUsesWith(r.qbs());
            else
                noTarget = true;
            assert(r.res().getUses().empty() && "still has uses");
        } else if (auto cx = dyn_cast<CNotOp>(gate)) {
            if (cx.qbs())
                cx.res().replaceAllUsesWith(cx.qbs());
            else
                noTarget = true;
            if (cx.ctrl())
                cx.new_ctrl().replaceAllUsesWith(cx.ctrl());
            assert(cx.res().getUses().empty() && "still has uses");
        } else if (auto sw = dyn_cast<SwapOp>(gate)) {
            if (sw.qbs())
                sw.res().replaceAllUsesWith(sw.qbs());
            else
                noTarget = true;
            if (sw.qbs2())
                sw.new_qbs2().replaceAllUsesWith(sw.qbs2());
            assert(sw.res().getUses().empty() && "still has uses");
        } else if (isa<FreeOp>(gate) || isa<FreeRegOp>(gate)) {
            // just delete (at bottom)
        } else if (isa<scf::YieldOp>(gate)) {
            gate->eraseOperands(0, gate->getNumOperands());
            gate->insertOperands(0, {Rcounter, Tcounter});
            return;
        } else if (auto call = dyn_cast<CallCircOp>(gate)) {
            gate->insertOperands(0, {Rcounter, Tcounter});
            SmallVector<Type, 6> retTypes(2, b.getI64Type());
            for (auto type : gate->getResultTypes())
                retTypes.push_back(type);

            b.setInsertionPoint(gate);
            OperationState callState(gate->getLoc(), CallOp::getOperationName());
            CallOp::build(b, callState, retTypes, call.circref(), gate->getOperands());
            Operation *newCallOp = b.createOperation(callState);

            gate->replaceAllUsesWith(newCallOp->getResults().drop_front(2));
            gate->erase();
            Rcounter = newCallOp->getResult(0);
            Tcounter = newCallOp->getResult(1);
            return;
        } else if (auto apply = dyn_cast<ApplyCircOp>(gate)) {
            gate->insertOperands(1, {Rcounter, Tcounter});
            SmallVector<Type, 6> retTypes(2, b.getI64Type());
            for (auto type : gate->getResultTypes())
                retTypes.push_back(type);

            b.setInsertionPoint(gate);
            OperationState callState(gate->getLoc(), CallOp::getOperationName());
            CallOp::build(b, callState, retTypes, cast<CircuitValueOp>(apply.circval().getDefiningOp()).circref(), gate->getOperands().drop_front());
            Operation *newCallOp = b.createOperation(callState);

            gate->replaceAllUsesWith(newCallOp->getResults().drop_front(2));
            gate->erase();
            Rcounter = newCallOp->getResult(0);
            Tcounter = newCallOp->getResult(1);
            return;
        } else if (isa<ReturnStateOp>(gate)) {
            // keep original return types for now to keep valid intermediate IR (will be removed in last step)
            // do replace those operands with function input arguments though to allow folding of operations
            Value qb, reg;
            for (auto arg : gate->getParentOfType<CircuitOp>().getArguments()) {
                if (arg.getType().isa<QstateType>())
                    qb = arg;
                else if (arg.getType().isa<RstateType>())
                    reg = arg;
            }
            SmallVector<Value, 4> retValues({Rcounter, Tcounter});
            for (auto type : gate->getOperandTypes()) {
                if (type.isa<QstateType>())
                    retValues.push_back(qb);
                else if (type.isa<RstateType>())
                    retValues.push_back(reg);
                else
                    llvm_unreachable("Non qdata type in return op!");
            }

            b.setInsertionPoint(gate);
            // if in main, insert print calls for the final counter values
            if (gate->getParentOp() == main) {
                OperationState printState(gate->getLoc(), vector::PrintOp::getOperationName());
                vector::PrintOp::build(b, printState, Rcounter);
                b.createOperation(printState);
                printState = OperationState(gate->getLoc(), vector::PrintOp::getOperationName());
                vector::PrintOp::build(b, printState, Tcounter);
                b.createOperation(printState);
            }

            OperationState retState(gate->getLoc(), ReturnOp::getOperationName());
            if (gate->getParentOp() == main)
                ReturnOp::build(b, retState, {});
            else
                ReturnOp::build(b, retState, retValues);
            b.createOperation(retState);
            gate->erase();
            return;
        } else if (auto meas = dyn_cast<MeasurementOp>(gate)) {
            assert(meas.qbs().getType().isa<QstateType>() && "Registers not yet implemented!");
            b.setInsertionPoint(meas);
            OperationState constState(meas.getLoc(), ConstantOp::getOperationName());
            ConstantOp::build(b, constState, b.getIntegerAttr(b.getI1Type(), 0));
            Operation *newOp = b.createOperation(constState);
            meas.qbs_out().replaceAllUsesWith(meas.qbs());
            meas.res().replaceAllUsesWith(newOp->getResult(0));
            meas.erase();
            return;
        } else {
            return;
        }

        if ((nctrl || isa<ROp>(gate)) && !noTarget)
            genCounterInc(b, gate, nctrl);
        gate->erase();
    }

    Operation* convertFor(OpBuilder &b, scf::ForOp op, Value &R, Value &T) {
        // setup counters as iteration arguments
        b.setInsertionPoint(op);
        OperationState forState(op.getLoc(), scf::ForOp::getOperationName());
        scf::ForOp::build(b, forState, op.lowerBound(), op.upperBound(), op.step(),
                          {Rcounter, Tcounter});
        Operation *newOp = b.createOperation(forState);

        // originally returned qdata values can be replaced by original iter args
        for (auto arg : op.getIterOperands())
            assert((arg.getType().isa<QstateType>() || arg.getType().isa<RstateType>()) && "Non-qdata in ForOp!");
        op.replaceAllUsesWith(op.getIterOperands());
        // iter block args can also be replaced by their initial values
        auto argIt = op.getIterOperands().begin();
        for (auto arg : op.getBody()->getArguments().drop_front())
            arg.replaceAllUsesWith(*argIt++);

        // steal loop body, remove any previous iter args and insert counters only
        while (op.getNumRegionIterArgs())
            op.getBody()->eraseArgument(1);
        newOp->getRegion(0).takeBody(op.getLoopBody());
        newOp->getRegion(0).front().insertArgument(1, b.getI64Type());
        newOp->getRegion(0).front().insertArgument(2, b.getI64Type());

        // region arguments need to be set as current counters before recursing into body
        // index 0 is the loop iteration variable
        Rcounter = newOp->getRegion(0).getArgument(1);
        Tcounter = newOp->getRegion(0).getArgument(2);

        op.erase();

        R = newOp->getResult(0);
        T = newOp->getResult(1);
        return newOp;
    }

    Operation* convertIf(OpBuilder &b, scf::IfOp op, Value &R, Value &T) {
        b.setInsertionPoint(op);
        OperationState ifState(op.getLoc(), scf::IfOp::getOperationName());
        scf::IfOp::build(b, ifState, {b.getI64Type(), b.getI64Type()}, op.condition(), true);
        Operation *newOp = b.createOperation(ifState);

        // need to replace original qdata return value with some value defined outside the if op
        Operation *term = op.elseRegion().front().getTerminator();
        assert(isa<scf::YieldOp>(term) && "Non-yield terminator!");
        SmallVector<Value, 2> retVals;
        for (auto arg : term->getOperands()) {
            assert((arg.getType().isa<QstateType>() || arg.getType().isa<RstateType>()) && "Non-qdata in YieldOp!");
            Value replValue = arg;
            // a block arg here (getDefiningOp = null) implies value must be defined outside if
            while(replValue.getDefiningOp() && replValue.getDefiningOp()->getParentOp() == op.getOperation()) {
                auto argIt = replValue.getDefiningOp()->getOperands().begin();
                while (replValue.getType() != (*argIt).getType())
                    argIt++;
                replValue = *argIt;
            }
            retVals.push_back(replValue);
        }

        newOp->getRegion(0).takeBody(op.thenRegion());
        newOp->getRegion(1).takeBody(op.elseRegion());
        op.replaceAllUsesWith(retVals);
        op.erase();

        R = newOp->getResult(0);
        T = newOp->getResult(1);
        return newOp;
    }

    void walkGates(OpBuilder &b, Operation *op) {
        // temporarily store current counter states to restore between regions of If operations
        Value R = Rcounter;
        Value T = Tcounter;
        // walk nested operations
        for (auto &region : op->getRegions()) {
            for (auto &block : region) {
                for (auto &nestedOp : llvm::make_early_inc_range(block)) {
                    if (isa<QuantumSSADialect>(nestedOp.getDialect()) || isa<scf::YieldOp>(nestedOp)) {
                        convertGates(b, &nestedOp);
                    } else if (auto forOp = dyn_cast<scf::ForOp>(nestedOp)) {
                        Value newR, newT;
                        walkGates(b, convertFor(b, forOp, newR, newT));
                        Rcounter = newR;
                        Tcounter = newT;
                    } else if (auto ifOp = dyn_cast<scf::IfOp>(nestedOp)) {
                        Value newR, newT;
                        walkGates(b, convertIf(b, ifOp, newR, newT));
                        Rcounter = newR;
                        Tcounter = newT;
                    }
                }
            }
            if (isa<scf::IfOp>(op)) {
                Rcounter = R;
                Tcounter = T;
            }
        }
    }

    void convertCircuit(OpBuilder &b, CircuitOp circ, ModuleOp module) {
        // create function to replace circuit
        b.setInsertionPoint(circ);

        SmallVector<Type, 6> argTypes(2, b.getI64Type());
        SmallVector<Type, 6> resTypes(2, b.getI64Type());
        for (auto type : circ.getArgumentTypes())
            argTypes.push_back(type);
        for (auto type : circ.getCallableResults())
            resTypes.push_back(type);
        FunctionType funtype = circ.getOperation() == main ? circ.getType()
                                                           : b.getFunctionType(argTypes, resTypes);

        OperationState funcState(circ.getLoc(), FuncOp::getOperationName());
        FuncOp::build(b, funcState, circ.getName(), funtype);
        Operation *newfun = b.createOperation(funcState);

        newfun->getRegion(0).takeBody(circ.gates());
        if (circ.getOperation() != main) {
            newfun->getRegion(0).front().insertArgument((unsigned) 0, b.getI64Type());
            newfun->getRegion(0).front().insertArgument((unsigned) 1, b.getI64Type());
            newfun->setAttr("_was_circ", b.getUnitAttr());
        }
        circ.erase();
    }

    void fold(ModuleOp &module) {
        OwningRewritePatternList patterns;
        ControlOp::getCanonicalizationPatterns(patterns, module.getContext());
        AdjointOp::getCanonicalizationPatterns(patterns, module.getContext());
        ApplyCircOp::getCanonicalizationPatterns(patterns, module.getContext());
        CombineStatOp::getCanonicalizationPatterns(patterns, module.getContext());
        CombineDynOp::getCanonicalizationPatterns(patterns, module.getContext());
        CircuitValueOp::getCanonicalizationPatterns(patterns, module.getContext());
        applyPatternsAndFoldGreedily(module, patterns);
    }

    void stripCall(OpBuilder &b, CallOp call) {
        SmallVector<Value, 4> operands;
        SmallVector<Value, 2> qargs;
        SmallVector<Type, 6> resTypes;
        for (auto arg : call.getOperands()) {
            if (!arg.getType().isa<QstateType>() && !arg.getType().isa<RstateType>())
                operands.push_back(arg);
            else
                qargs.push_back(arg);
        }
        for (auto ty : call.getResultTypes())
            if (!ty.isa<QstateType>() && !ty.isa<RstateType>())
                resTypes.push_back(ty);

        b.setInsertionPoint(call);
        OperationState callState(call.getLoc(), CallOp::getOperationName());
        CallOp::build(b, callState, call.getCallee(), resTypes, operands);
        Operation *newOp = b.createOperation(callState);

        auto resIt = newOp->result_begin();
        auto qresIt = qargs.begin();
        for (auto res : call.getResults()) {
            if (res.getType().isa<QstateType>() || res.getType().isa<RstateType>())
                res.replaceAllUsesWith(*qresIt++);
            else
                res.replaceAllUsesWith(*resIt++);
        }
        call.erase();
    }

    void walkCalls(OpBuilder &b, Operation *op) {
        // walk nested operations
        for (auto &region : op->getRegions())
            for (auto &block : region)
                for (auto &nestedOp : llvm::make_early_inc_range(block))
                    if (auto call = dyn_cast<CallOp>(nestedOp))
                        stripCall(b, call);
                    else if (isa<scf::ForOp>(nestedOp) || isa<scf::IfOp>(nestedOp))
                        walkCalls(b, &nestedOp);
    }

    void stripReturns(OpBuilder &b, FuncOp func) {
        // walk nested blocks, check return op terminators for qdata operands
        for (auto &block : func.getBlocks()) {
            Operation *term = block.getTerminator();
            if (isa<ReturnOp>(term)) {
                unsigned i = 0;
                while (i < term->getNumOperands()) {
                    Type ty = term->getOperand(i).getType();
                    if (ty.isa<QstateType>() || ty.isa<RstateType>()) {
                        term->eraseOperand(i);
                        i--;
                    }
                    i++;
                }
            }
        }

        // make final addition of local counter to input arg
        Value R = func.getArgument(0);
        Value T = func.getArgument(1);
        for (auto &block : func.getBlocks()) {
            Operation *term = block.getTerminator();
            if (isa<ReturnOp>(term)) {
                b.setInsertionPoint(term);
                OperationState addState(term->getLoc(), AddIOp::getOperationName());
                AddIOp::build(b, addState, R, term->getOperand(0));
                term->setOperand(0, b.createOperation(addState)->getResult(0));
                addState = OperationState(term->getLoc(), AddIOp::getOperationName());
                AddIOp::build(b, addState, T, term->getOperand(1));
                term->setOperand(1, b.createOperation(addState)->getResult(0));
            }
        }
    }

    void stripFuncQData(OpBuilder &b, FuncOp func) {
        // filter argument use in return ops
        stripReturns(b, func);

        // filter function types
        SmallVector<Type, 6> argTypes, resTypes;
        SmallVector<int, 4> qIndices;
        int i = 0;
        for (auto ty : func.getArgumentTypes()) {
            if (!ty.isa<QstateType>() && !ty.isa<RstateType>())
                argTypes.push_back(ty);
            else
                qIndices.push_back(i);
            i++;
        }
        for (auto ty : func.getCallableResults())
            if (!ty.isa<QstateType>() && !ty.isa<RstateType>())
                resTypes.push_back(ty);

        // filter function args
        int offset = 0;
        for (int i : qIndices) {
            assert(func.getArgument(i-offset).getUses().empty() && "Func arg still has uses!");
            func.eraseArgument(i-offset++);
        }

        func.setType(b.getFunctionType(argTypes, resTypes));
    }

public:
    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder b(module.getContext());
        nrec = 0;

        // assume all quantum code is within circuit ops
        // start stripping all meta operations from the program
        for (auto &block : module.getBodyRegion()) {
            for (auto &op : llvm::make_early_inc_range(block)) {
                if (isa<CircuitOp>(op)) {
                    // adjoints and controls can all be removed in a single pass
                    // as their effect does not require repeated propagation
                    walkAdj(b, &op);
                    walkCtrl(b, &op);
                }
            }
        }

        // we can propagate controls on circuits starting from program entry point
        main = module.lookupSymbol("mlir_main");
        assert(main && "Need circuit entry point!");
        walkCallTree(b, main, 0);
        llvm::errs() << "\n -------- nrec = " << nrec << " -------- \n\n";

        // TODO: remove unused circuit definitions

        // convert gates to resource counter increments, convert circuits to functions
        for (auto &block : module.getBodyRegion()) {
            for (auto &op : llvm::make_early_inc_range(block)) {
                if (auto circ = dyn_cast<CircuitOp>(op)) {
                    initialize(b, circ);
                    walkGates(b, &op);
                    convertCircuit(b, circ, module);
                }
            }
        }

        fold(module);

        // filter qdata argument use in function calls
        for (auto &block : module.getBodyRegion())
            for (auto &op : llvm::make_early_inc_range(block))
                if (auto func = dyn_cast<FuncOp>(op))
                    if (func.getAttr("_was_circ"))
                        walkCalls(b, func);
        // this should remove any last extract/combines
        fold(module);

        // at this point, only allocations should be left
        // they will be folded once qubits and registers are filtered from function calls
        for (auto &block : module.getBodyRegion())
            for (auto &op : llvm::make_early_inc_range(block))
                if (auto func = dyn_cast<FuncOp>(op))
                    if (func.getAttr("_was_circ"))
                        stripFuncQData(b, func);

        fold(module);
    }
};
} // end namespace

std::unique_ptr<Pass> quantum::createResourceCounterPass() {
    return std::make_unique<ResourceCounterPass>();
}
