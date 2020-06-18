/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "QuantumDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantum;

// Define the Dialect contructor. This is the point of registration of
// all custom types, operations, attributes, etc. for the dialect.
QuantumDialect::QuantumDialect(mlir::MLIRContext *ctx) : mlir::Dialect("q", ctx) {
    // these templated methods are from the mlir::Dialect class
    addOperations<
        #define GET_OP_LIST
        #include "QuantumOps.cpp.inc"
    >();
    
    addTypes<QubitType, QuregType, OpType, COpType, CircType>();  // in mlir::quantum

    // addAttributes<QuantumAttribute>();
    // addInterfaces<QuantumInterface>();
}

namespace mlir {
namespace quantum {
namespace detail {
// This class represents the internal storage of the Quantum 'QuregType'.
struct QuregTypeStorage : public mlir::TypeStorage {
    // The `KeyTy` is a required type that provides an interface for the storage instance.
    // This type will be used when uniquing an instance of the type storage. For our Qureg
    // type, we will unique each instance on its size.
    using KeyTy = unsigned;

    // Size of the qubit register
    unsigned size;

    // A constructor for the type storage instance.
    QuregTypeStorage(unsigned size) { this->size = size; }

    // Define the comparison function for the key type with the current storage instance.
    // This is used when constructing a new instance to ensure that we haven't already
    // uniqued an instance of the given key.
    bool operator==(const KeyTy &key) const { return key == size; }

    // Define a construction method for creating a new instance of this storage.
    // This method takes an instance of a storage allocator, and an instance of a `KeyTy`.
    // The given allocator must be used for *all* necessary dynamic allocations used to
    // create the type storage and its internal.
    static QuregTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        // Allocate the storage instance and construct it.
        return new (allocator.allocate<QuregTypeStorage>()) QuregTypeStorage(key);
    }
};

// This class represents the internal storage of the Quantum 'COpType'.
struct COpTypeStorage : public mlir::TypeStorage {
    using KeyTy = unsigned;

    unsigned nctrl;

    COpTypeStorage(unsigned nctrl) { this->nctrl = nctrl; }

    bool operator==(const KeyTy &key) const { return key == nctrl; }

    static COpTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<COpTypeStorage>()) COpTypeStorage(key);
    }
};
} // end namespace detail
} // end namespace quantum
} // end namespace mlir

/* Methods of complex types must be implemented after their TypeStorage has been defined */

// Qureg
QuregType QuregType::get(mlir::MLIRContext *ctx, unsigned size) {
    assert((size > 0) && "Qureg size must be > 0");

    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, QuantumTypes::Qureg, size);
}

unsigned QuregType::getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->size;
}

// COp
COpType COpType::get(mlir::MLIRContext *ctx, unsigned nctrl) {
    assert((nctrl > 0) && "Qureg size must be > 0");

    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, QuantumTypes::COp, nctrl);
}

unsigned COpType::getNumCtrls() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->nctrl;
}

/* Finally, to be able to read and output .mlir code (roundtrip) from this dialect
   with our custom types, we need to overwrite the printType and parseType hooks. */

// Print an instance of a type registered in the Quantum dialect.
void QuantumDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    // Differentiate between the Quantum types via their kinds and print accordingly.
    switch (type.getKind()) {
    case QuantumTypes::Qubit:
        printer << "qubit";
        break;
    case QuantumTypes::Qureg: {
        QuregType ctype = type.cast<QuregType>();
        printer << "qureg<" << ctype.getNumQubits() << ">";
        break;
    }
    case QuantumTypes::Op:
        printer << "op";
        break;
    case QuantumTypes::COp: {
        COpType ctype = type.cast<COpType>();
        printer << "cop<" << ctype.getNumCtrls() << ">";
        break;
    }
    case QuantumTypes::Circ:
        printer << "circ";
        break;
    default:
        throw "unrecognized type encountered in the printer!";
    }
} 

// Parse an instance of a type registered to the Quantum dialect.
mlir::Type QuantumDialect::parseType(mlir::DialectAsmParser &parser) const {
    // NOTE: All MLIR parser function return a ParseResult. This is a
    // specialization of LogicalResult that auto-converts to a `true` boolean
    // value on failure to allow for chaining, but may be used with explicit
    // `mlir::failed/mlir::succeeded` as desired.

    // Try to parse either the Qubit or Qureg type. On failure, exit this function.
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
        return Type();
    }
    if (keyword == "qubit")
        return QubitType::get(this->getContext());
    if (keyword == "qureg") {
        unsigned size;
        if (parser.parseLess()) {
            return nullptr;
        }
        if (parser.parseInteger<unsigned>(size)) {
            return nullptr;
        }
        if (parser.parseGreater()) {
            return nullptr;
        }
        return QuregType::get(this->getContext(), size);
    }
    if (keyword == "op")
        return OpType::get(this->getContext());
    if (keyword == "cop") {
        unsigned nctrl;
        if (parser.parseLess()) {
            return nullptr;
        }
        if (parser.parseInteger<unsigned>(nctrl)) {
            return nullptr;
        }
        if (parser.parseGreater()) {
            return nullptr;
        }
        return COpType::get(this->getContext(), nctrl);
    }
    if (keyword == "circ")
        return CircType::get(this->getContext());

    parser.emitError(parser.getNameLoc(), "unrecognized quantum type");
    return Type();
}

//===------------------------------------------------------------------------------------------===//
// Custom CircuitOp assembly format
//===------------------------------------------------------------------------------------------===//
static void print(OpAsmPrinter &p, CircuitOp op) {
    p << op.getOperationName();
    if (op.getAttr("name")) {
        p << "(";
        p.printAttributeWithoutType(op.nameAttr());
        p << ")";
    }
    p.printRegion(op.gates(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
    p << " : " << op.getType();
}

static ParseResult parseCircuitOp(OpAsmParser &p, OperationState &result) {
    auto &builder = p.getBuilder();

    // parse optional 'name' attribute as "function argument"
    if (succeeded(p.parseOptionalLParen())) {
        StringAttr nameAttr;
        if (p.parseAttribute(nameAttr, "name", result.attributes))
            return failure();
        if (p.parseRParen())
            return failure();
    }

    // Parse the body region.
    Region *body = result.addRegion();
    if (p.parseRegion(*body, {}, {}))
        return failure();

    CircuitOp::ensureTerminator(*body, builder, result.location);

    // Parse the optional attribute list.
    if (p.parseOptionalAttrDict(result.attributes))
        return failure();

    // Parse return type
    Type type;
    llvm::SMLoc trailingTypeLoc;
    if (p.parseColon() || p.getCurrentLocation(&trailingTypeLoc) || p.parseType(type))
        return failure();

    // Extract the result type from the trailing function type.
    auto funcType = type.dyn_cast<FunctionType>();
    if (funcType) {
        if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1)
            return p.emitError(trailingTypeLoc,
                "expected trailing function type with no argument and one result");
        result.addTypes({funcType.getResult(0)});
    } else {
        result.addTypes({type});
    }

    return success();
}

#define GET_OP_CLASSES
#include "QuantumOps.cpp.inc"
