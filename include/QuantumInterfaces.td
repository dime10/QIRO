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
        StaticInterfaceMethod<"Parse the given number of quantum data (qubits, registers, lists) "
                              "operands with optional accessors. Accessors are only parsed on "
                              "registers and lists.",
            "StringRef", "getAccessorAttrName", (ins "unsigned" : $n)
        >,
        StaticInterfaceMethod<"Parse the given number of quantum data (qubits, registers, lists) "
                              "operands with optional accessors. Accessors are only parsed on "
                              "registers and lists.",
            "std::pair<SmallVector<bool, 4>, unsigned>", "getRegLikeArray"
        >,
        StaticInterfaceMethod<"Parse the given number of quantum data (qubits, registers, lists) "
                              "operands with optional accessors. Accessors are only parsed on "
                              "registers and lists.",
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
                    if (!argType.isa<QuregType>() && !argType.isa<QlistType>() && accessorPresent)
                        concreteOp.emitError() << "Range specifiers are only allowed on register-like types!";
                }
                i++;
            }
        }

        return success();
    }];
}

#endif // QUANTUM_INTERFACES
