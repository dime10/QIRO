/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "QuantumDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

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
    
    addTypes<QubitType, QuregType>();  // in mlir::quantum

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
} // end namespace detail
} // end namespace quantum
} // end namespace mlir

/* QuregType method implementations after QuregTypeStorage has been defined */

QuregType QuregType::get(mlir::MLIRContext *ctx, unsigned size) {
    assert(!(size==0) && "Qureg size must be >= 1");

    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, QuantumTypes::Qureg, size);
}

unsigned QuregType::getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->size;
}

/* Finally, to be able to read and output .mlir code (roundtrip) from this dialect
   with our custom types, we need to overwrite the printType and parseType hooks. */

// Print an instance of a type registered in the Quantum dialect.
void QuantumDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    // Differentiate between the Quantum types via their kinds and print accordingly.
    if (type.getKind() == QuantumTypes::Qubit) {
        printer << "qubit";
    }
    else if (type.getKind() == QuantumTypes::Qureg) {
        QuregType type = type.cast<QuregType>();
        printer << "qureg[" << type.getNumQubits() << "]";
    }
    else {
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
    if (parser.parseKeyword("qubit")) {
        if (parser.parseKeyword("qureg")) {
            return Type();
        }

        // parse the register size if successfully matched 'qureg'
        llvm::SMLoc typeLoc = parser.getCurrentLocation();
        if (parser.parseLSquare()) {
            parser.emitError(typeLoc, "qureg type must have a size in square brackets,"
                                      " e.g. qureg[5]");
            return Type();
        }

        typeLoc = parser.getCurrentLocation();
        unsigned size;
        if (parser.parseInteger<unsigned>(size)) {
            parser.emitError(typeLoc, "qureg type must have a size in square brackets,"
                                      " e.g. qureg[5]");
            return Type();
        }
        
        typeLoc = parser.getCurrentLocation();
        if (parser.parseRSquare()) {
            parser.emitError(typeLoc, "qureg type must have a size in square brackets,"
                                      " e.g. qureg[5]");
            return Type();
        }

        return QuregType::get(this->getContext(), size);
    }

    return QubitType::get(this->getContext());
}

#define GET_OP_CLASSES
#include "QuantumOps.cpp.inc"
