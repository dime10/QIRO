/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringSwitch.h"

#include "QuantumSSADialect.h"

using namespace mlir;
using namespace mlir::quantumssa;

#define EMIT_ERROR(p, m) p.emitError(p.getCurrentLocation(), m)

//===------------------------------------------------------------------------------------------===//
// Dialect Definitions
//===------------------------------------------------------------------------------------------===//

// latest changes in MLIR upstream now only require this function for op, type, etc. registration
void QuantumSSADialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "QuantumSSAOps.cpp.inc"
    >();

    addTypes<QstateType, RstateType, OpType, COpType, FunCircType>();
}

namespace mlir {
namespace quantumssa {
namespace detail {
// This class represents the internal storage of the Quantum 'QuregType'.
struct RstateTypeStorage : public TypeStorage {
    // The `KeyTy` is a required type that provides an interface for the storage instance.
    // This type will be used when uniquing an instance of the type storage. For our Qureg
    // type, we will unique each instance on its size.
    using KeyTy = int;

    // Size of the qubit register
    llvm::Optional<int> size;

    // A constructor for the type storage instance.
    RstateTypeStorage(llvm::Optional<int> size) {
        this->size = size;
    }

    // Define the comparison function for the key type with the current storage instance.
    // This is used when constructing a new instance to ensure that we haven't already
    // uniqued an instance of the given key.
    bool operator==(const KeyTy &key) const {
        return size == (key < 0 ? llvm::None : llvm::Optional<int>(key));
    }

    // Define a construction method for creating a new instance of this storage.
    // This method takes an instance of a storage allocator, and an instance of a `KeyTy`.
    // The given allocator must be used for *all* necessary dynamic allocations used to
    // create the type storage and its internal.
    static RstateTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        // Allocate the storage instance and construct it.
        llvm::Optional<int> size = key < 0 ? llvm::None : llvm::Optional<int>(key);
        return new (allocator.allocate<RstateTypeStorage>()) RstateTypeStorage(size);
    }
};

// This class represents the internal storage of the Quantum 'COpType'.
struct COpTypeStorage : public TypeStorage {
    using KeyTy = std::pair<unsigned, Type>;

    unsigned nctrl;
    Type baseType;

    COpTypeStorage(unsigned nctrl, Type baseType) {
        assert(nctrl > 0 && "Number of controls must be > 0");
        if (baseType)
            assert(baseType.isa<OpType>() || baseType.isa<FunCircType>() &&
                   "Base type of controlled op can only be supported quantum operations!");
        this->nctrl = nctrl;
        this->baseType = baseType;
    }

    bool operator==(const KeyTy &key) const { return key.first == nctrl && key.second == baseType; }

    static COpTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<COpTypeStorage>()) COpTypeStorage(key.first, key.second);
    }
};

// This class represents the internal storage of the Quantum 'FunCircType'.
struct FunCircTypeStorage : public TypeStorage {
    using KeyTy = FunctionType;

    FunctionType funtype;

    FunCircTypeStorage(FunctionType funtype) { this->funtype = funtype; }

    static llvm::hash_code hashKey(const KeyTy &funtype) {
        return hash_value(funtype);
    }

    bool operator==(const KeyTy &key) const { return key == this->funtype; }

    static FunCircTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<FunCircTypeStorage>()) FunCircTypeStorage(key);
    }
};
} // end namespace detail
} // end namespace quantumssa
} // end namespace mlir


//===------------------------------------------------------------------------------------------===//
// Method implementations of complex types
//===------------------------------------------------------------------------------------------===//

// Rstate
RstateType RstateType::get(MLIRContext *ctx, llvm::Optional<int> size) {
    // Parameters to the storage class are passed after the custom type kind.
    detail::RstateTypeStorage::KeyTy key = size ? *size : -1;
    return Base::get(ctx, key);
}

llvm::Optional<int> RstateType::getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->size;
}

// COp
COpType COpType::get(MLIRContext *ctx, unsigned nctrl, Type baseType) {
    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, nctrl, baseType);
}

unsigned COpType::getNumCtrls() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->nctrl;
}

Type COpType::getBaseType() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->baseType;
}

// FunCirc
FunCircType FunCircType::get(MLIRContext *ctx, FunctionType funtype) {
    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, funtype);
}

FunctionType FunCircType::getFunType() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->funtype;
}


//===------------------------------------------------------------------------------------------===//
// Dialect types printing and parsing
//===------------------------------------------------------------------------------------------===//

// Print an instance of a type registered in the Quantum dialect.
void QuantumSSADialect::printType(Type type, DialectAsmPrinter &printer) const {
    // Differentiate between the Quantum types and print accordingly.
    llvm::TypeSwitch<Type>(type)
        .Case<QstateType> ([&](QstateType)    { printer << "qstate"; })
        .Case<RstateType> ([&](RstateType t)  { printer << "rstate<";
                                                if (auto numQubits = t.getNumQubits())
                                                    printer << *numQubits;
                                                printer << ">"; })
        .Case<OpType>     ([&](OpType)        { printer << "op"; })
        .Case<COpType>    ([&](COpType t)     { printer << "cop<" << t.getNumCtrls();
                                                if (auto baseType = t.getBaseType())
                                                    printer << ", " << baseType;
                                                printer << ">"; })
        .Case<FunCircType>([&](FunCircType t) { printer << "fcirc<" << t.getFunType() << ">"; })
        .Default([](Type) { llvm_unreachable("unrecognized type encountered in the printer!"); });
}

// Parse an instance of a type registered to the Quantum dialect.
Type QuantumSSADialect::parseType(DialectAsmParser &parser) const {
    // NOTE: All MLIR parser function return a ParseResult. This is a
    // specialization of LogicalResult that auto-converts to a `true` boolean
    // value on failure to allow for chaining, but may be used with explicit
    // `mlir::failed/mlir::succeeded` as desired.
    Builder &builder = parser.getBuilder();

    // Attempt to parse all supported dialect types.
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return EMIT_ERROR(parser, "error parsing type keyword!"), nullptr;

    // Lambdas are needed so as not to call the helper functions due to eager parameter evaluation
    Type result = llvm::StringSwitch<function_ref<Type()>>(keyword)
        .Case("qstate", [&] { return builder.getType<QstateType>(); })
        .Case("rstate", [&] { return parseRstateType(parser); })
        .Case("op",     [&] { return builder.getType<OpType>(); })
        .Case("cop",    [&] { return parseCOpType(parser); })
        .Case("fcirc",  [&] { return parseFunCircType(parser); })
        .Default([&] { return EMIT_ERROR(parser, "unrecognized quantumssa type!"), nullptr; })();

    return result;
}

Type QuantumSSADialect::parseRstateType(DialectAsmParser &parser) {
    StringRef errmsg = "error during 'Rstate' type parsing!";
    llvm::Optional<int> optionalSize;
    int size;

    if (parser.parseLess())
        return EMIT_ERROR(parser, errmsg), nullptr;

    auto res = parser.parseOptionalInteger<int>(size);
    if (res.hasValue() && failed(res.getValue()))
        return EMIT_ERROR(parser, errmsg), nullptr;
    optionalSize = res.hasValue() ? llvm::Optional<int>(size) : llvm::None;

    if (parser.parseGreater())
        return EMIT_ERROR(parser, errmsg), nullptr;

    return parser.getBuilder().getType<RstateType>(optionalSize);
}

Type QuantumSSADialect::parseCOpType(DialectAsmParser &parser) {
    StringRef errmsg = "error during 'COp' type parsing!";
    Type baseType(nullptr);
    int nctrl;

    if (parser.parseLess() || parser.parseInteger<int>(nctrl))
        return EMIT_ERROR(parser, errmsg), nullptr;

    if (succeeded(parser.parseOptionalComma()))
        if (parser.parseType(baseType))
            return EMIT_ERROR(parser, errmsg), nullptr;

    if (parser.parseGreater())
        return EMIT_ERROR(parser, errmsg), nullptr;

    if (baseType && !(baseType.isa<OpType>() || baseType.isa<FunCircType>()))
        return EMIT_ERROR(parser, "base type of controlled op must be either 'Op' or 'Circ' type!"),
               nullptr;

    return parser.getBuilder().getType<COpType>(nctrl, baseType);
}

Type QuantumSSADialect::parseFunCircType(DialectAsmParser &parser) {
    FunctionType funtype;

    if (parser.parseLess() || parser.parseType<FunctionType>(funtype) || parser.parseGreater())
        return EMIT_ERROR(parser, "error during 'FunCirc' type parsing!"), nullptr;

    return parser.getBuilder().getType<FunCircType>(funtype);
}


//===------------------------------------------------------------------------------------------===//
// Auto-generated op definitions
//===------------------------------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QuantumSSAOps.cpp.inc"
