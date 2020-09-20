#ifndef QUANTUM_INTERFACES
#define QUANTUM_INTERFACES

include "mlir/IR/OpBase.td"

// Define operation interfaces
def RegAccessInterface : OpInterface<"RegAccessInterface"> {
    let description = [{
        This interface allows dialect operations that accept qubit
        registers as operands to also access individual qubits, as
        well as qubit ranges, within those registers. It provides
        a unified interface to simply parsing these specialized
        register accesses.
    }];

    let methods = [
        StaticInterfaceMethod<"Obtain the name of the i^th static range array attribute.",
            "StringRef", "getAccessorAttrName", (ins "unsigned" : $idx)
        >,
        StaticInterfaceMethod<"Obtain the names of all static range array attributes as a vector.",
            "SmallVector<StringRef, 2>", "getAccessorAttrNames"
        >,
        StaticInterfaceMethod<"Get a boolean array that indicates for each ODS operand whether it "
                              "is register-like, and thus has a corresponding range operand and "
                              "attribute.",
            "std::pair<SmallVector<bool, 4>, unsigned>", "getRegLikeArray"
        >,
        StaticInterfaceMethod<"Number of ODS operands.",
            "size_t", "getSegmentSizesArraySize"
        >
    ];

    let verify = [{
        auto concreteOp = cast<ConcreteOp>($_op);
        auto segmentSizeAttr = concreteOp.getAttrOfType<DenseIntElementsAttr>(concreteOp.getOperandSegmentSizeAttr());
        auto isRegLike = concreteOp.getRegLikeArray().first;
        unsigned absoluteIdx = 0, regLikeIdx = 0, encounteredRegLikes = 0;

        for (unsigned i = 0; i < concreteOp.getSegmentSizesArraySize(); i++) {
            unsigned numOperands = (*(segmentSizeAttr.begin() + i)).getZExtValue();
            auto operandRange = concreteOp.getODSOperands(i);
            if (isRegLike[regLikeIdx++]) {
                auto accessorAttr = concreteOp.getAttrOfType<ArrayAttr>(concreteOp.getAccessorAttrName(encounteredRegLikes++));
                for (unsigned j = 0; j < numOperands; j++) {
                    auto argType = operandRange[j].getType();
                    bool accessorPresent;
                    if (accessorAttr.size() && accessorAttr[0].isa<ArrayAttr>())
                        accessorPresent = j < accessorAttr.size() ? accessorAttr[j].cast<ArrayAttr>().size() : false;
                    else
                        accessorPresent = accessorAttr.size();
                    if (!argType.isa<QuregType>() && accessorPresent)
                        return concreteOp.emitError() << "Range specifiers are only allowed on register types!";
                }
                i++;
            }
        }

        return success();
    }];
}

#endif // QUANTUM_INTERFACES
