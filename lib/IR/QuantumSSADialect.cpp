/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "QuantumSSADialect.h"

using namespace mlir;
using namespace mlir::quantumssa;


//===------------------------------------------------------------------------------------------===//
// Dialect Definitions
//===------------------------------------------------------------------------------------------===//

// latest changes in MLIR upstream now only require this function for op, type, etc. registration
void QuantumSSADialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "QuantumSSAOps.cpp.inc"
    >();

    addTypes<QstateType, RstateType, LstateType, OpType, COpType, FunCircType>();
}

namespace mlir {
namespace quantumssa {
namespace detail {
// This class represents the internal storage of the Quantum 'QuregType'.
struct RstateTypeStorage : public mlir::TypeStorage {
    // The `KeyTy` is a required type that provides an interface for the storage instance.
    // This type will be used when uniquing an instance of the type storage. For our Qureg
    // type, we will unique each instance on its size.
    using KeyTy = unsigned;

    // Size of the qubit register
    unsigned size;

    // A constructor for the type storage instance.
    RstateTypeStorage(unsigned size) {
        assert(size > 1 && "Register type must have size > 1!");
        this->size = size;
    }

    // Define the comparison function for the key type with the current storage instance.
    // This is used when constructing a new instance to ensure that we haven't already
    // uniqued an instance of the given key.
    bool operator==(const KeyTy &key) const { return key == size; }

    // Define a construction method for creating a new instance of this storage.
    // This method takes an instance of a storage allocator, and an instance of a `KeyTy`.
    // The given allocator must be used for *all* necessary dynamic allocations used to
    // create the type storage and its internal.
    static RstateTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        // Allocate the storage instance and construct it.
        return new (allocator.allocate<RstateTypeStorage>()) RstateTypeStorage(key);
    }
};

// This class represents the internal storage of the Quantum 'COpType'.
struct COpTypeStorage : public mlir::TypeStorage {
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

    static COpTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<COpTypeStorage>()) COpTypeStorage(key.first, key.second);
    }
};

// This class represents the internal storage of the Quantum 'FunCircType'.
struct FunCircTypeStorage : public mlir::TypeStorage {
    using KeyTy = FunctionType;

    FunctionType funtype;

    FunCircTypeStorage(FunctionType funtype) { this->funtype = funtype; }

    static llvm::hash_code hashKey(const KeyTy &funtype) {
        return hash_value(funtype);
    }

    bool operator==(const KeyTy &key) const { return key == this->funtype; }

    static FunCircTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
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
RstateType RstateType::get(mlir::MLIRContext *ctx, unsigned size) {
    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, size);
}

unsigned RstateType::getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->size;
}

// COp
COpType COpType::get(mlir::MLIRContext *ctx, unsigned nctrl, mlir::Type baseType) {
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
FunCircType FunCircType::get(mlir::MLIRContext *ctx, FunctionType funtype) {
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
void QuantumSSADialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    // Differentiate between the Quantum types and print accordingly.
    llvm::TypeSwitch<mlir::Type>(type)
        .Case<QstateType>([&](QstateType)     { printer << "qstate"; })
        .Case<RstateType>([&](RstateType t)   { printer << "rstate<" << t.getNumQubits() << ">"; })
        .Case<LstateType>([&](LstateType)     { printer << "lstate"; })
        .Case<OpType>([&](OpType)             { printer << "op"; })
        .Case<COpType>([&](COpType t)         { printer << "cop<" << t.getNumCtrls();
                                                if (t.getBaseType())
                                                    printer << ", " << t.getBaseType();
                                                printer << ">"; })
        .Case<FunCircType>([&](FunCircType t) { printer << "fcirc<" << t.getFunType() << ">"; })
        .Default([](Type) { llvm_unreachable("unrecognized type encountered in the printer!"); });
}

// Parse an instance of a type registered to the Quantum dialect.
mlir::Type QuantumSSADialect::parseType(mlir::DialectAsmParser &parser) const {
    // NOTE: All MLIR parser function return a ParseResult. This is a
    // specialization of LogicalResult that auto-converts to a `true` boolean
    // value on failure to allow for chaining, but may be used with explicit
    // `mlir::failed/mlir::succeeded` as desired.

    // Try to parse either the Qubit or Qureg type. On failure, exit this function.
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return Type();

    if (keyword == "qstate")
        return QstateType::get(this->getContext());
    if (keyword == "rstate") {
        unsigned size;
        if (parser.parseLess())
            return nullptr;
        if (parser.parseInteger<unsigned>(size))
            return nullptr;
        if (parser.parseGreater())
            return nullptr;
        return RstateType::get(this->getContext(), size);
    }
    if (keyword == "lstate")
        return LstateType::get(this->getContext());
    if (keyword == "op")
        return OpType::get(this->getContext());
    if (keyword == "cop") {
        unsigned nctrl;
        Type baseType;
        if (parser.parseLess())
            return nullptr;
        if (parser.parseInteger<unsigned>(nctrl))
            return nullptr;
        if (succeeded(parser.parseOptionalComma()))
            if (parser.parseType(baseType))
                return nullptr;
        if (parser.parseGreater())
            return nullptr;
        return COpType::get(this->getContext(), nctrl, baseType);
    }
    if (keyword == "fcirc") {
        FunctionType funtype;
        if (parser.parseLess())
            return nullptr;
        if (parser.parseType<FunctionType>(funtype))
            return nullptr;
        if (parser.parseGreater())
            return nullptr;
        return FunCircType::get(this->getContext(), funtype);
    }

    parser.emitError(parser.getNameLoc(), "Unrecognized quantum ssa type!");
    return Type();
}


//===------------------------------------------------------------------------------------------===//
// Custom pretty assembly format for gate ops
//===------------------------------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, HOp op) {
    p << op.getOperationName();
    if (op.qbs()) {
        p << " ";
        p.printOperand(op.qbs());
        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
        p << " : ";
        p.printType(op.qbs().getType());
    } else {
        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
    }
    if (op.res()) {
        p << " -> ";
        p.printType(op.res().getType());
    }
}

static void print(OpAsmPrinter &p, XOp op) {
    p << op.getOperationName();
    if (op.qbs()) {
        p << " ";
        p.printOperand(op.qbs());
        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
        p << " : ";
        p.printType(op.qbs().getType());
    } else {
        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
    }
    if (op.res()) {
        p << " -> ";
        p.printType(op.res().getType());
    }
}

static void print(OpAsmPrinter &p, RzOp op) {
    p << op.getOperationName();
    p << "(";
    p.printAttributeWithoutType(op.phiAttr());
    p << ")";
    if (op.qbs()) {
        p << " ";
        p.printOperand(op.qbs());
        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"phi"});
        p << " : ";
        p.printType(op.qbs().getType());
    } else {
        p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"phi"});
    }
    if (op.res()) {
        p << " -> ";
        p.printType(op.res().getType());
    }
}

static void print(OpAsmPrinter &p, CNotOp op) {
    p << op.getOperationName();
    p << " ";
    p.printOperand(op.ctrl());
    if (op.qbs()) {
        p << ", ";
        p.printOperand(op.qbs());
    }
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
    p << " : ";
    p.printType(op.ctrl().getType());
    if (op.qbs()) {
        p << ", ";
        p.printType(op.qbs().getType());
    }
    if (op.res()) {
        p << " -> ";
        p.printType(op.res().getType());
    }
}

static void print(OpAsmPrinter &p, ControlOp op) {
    p << op.getOperationName();
    p << " ";
    p.printOperand(op.heldOp());
    p << ", ";
    p.printOperand(op.ctrls());
    if (op.qbs()) {
        p << ", ";
        p.printOperand(op.qbs());
    }
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
    p << " : ";
    p.printType(op.heldOp().getType());
    p << ", ";
    p.printType(op.ctrls().getType());
    if (op.qbs()) {
        p << ", ";
        p.printType(op.qbs().getType());
    }
    if (op.res()) {
        p << " -> ";
        p.printType(op.res().getType());
    }
}

static void print(OpAsmPrinter &p, AdjointOp op) {
    p << op.getOperationName();
    p << " ";
    p.printOperand(op.heldOp());
    if (op.qbs()) {
        p << ", ";
        p.printOperand(op.qbs());
    }
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
    p << " : ";
    p.printType(op.heldOp().getType());
    if (op.qbs()) {
        p << ", ";
        p.printType(op.qbs().getType());
    }
    if (op.res()) {
        p << " -> ";
        p.printType(op.res().getType());
    }
}

static ParseResult prettyParseOp(OpAsmParser &p, OperationState &result, bool parametric = false) {
    if (parametric) {
        FloatAttr phiAttr;
        if (p.parseLParen())
            return failure();
        if (p.parseAttribute(phiAttr, "phi", result.attributes))
            return failure();
        if (p.parseRParen())
            return failure();
    }

    SmallVector<Type, 3> allOperandTypes;
    llvm::SMLoc allOperandLoc = p.getCurrentLocation();
    SmallVector<OpAsmParser::OperandType, 3> allOperands;
    if (p.parseOperandList(allOperands))
        return failure();

    if (p.parseOptionalAttrDict(result.attributes))
        return failure();

    if (succeeded(p.parseOptionalColon()))
        if (p.parseTypeList(allOperandTypes))
            return failure();

    // parse optional return type
    if (succeeded(p.parseOptionalArrow())) {
        Type type;
        if (p.parseType(type))
            return failure();
        result.addTypes({type});
    }

    if (p.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
        return failure();

    return success();
}


//===------------------------------------------------------------------------------------------===//
// Additional implementations of OpInterface methods or ExtraClassDeclarations
//===------------------------------------------------------------------------------------------===//

ArrayRef<Type> ApplyFunCircOp::getInputsSafe(Type callee) {
    ArrayRef<Type> inputs;
    if (callee.isa<COpType>())
        callee = callee.cast<COpType>().getBaseType();
    if (callee && callee.isa<FunCircType>())
        inputs = callee.cast<FunCircType>().getFunType().getInputs();
    return inputs;
}

ArrayRef<Type> ApplyFunCircOp::getResultsSafe(Type callee) {
    ArrayRef<Type> results;
    if (callee.isa<COpType>())
        callee = callee.cast<COpType>().getBaseType();
    if (callee && callee.isa<FunCircType>())
        results = callee.cast<FunCircType>().getFunType().getResults();
    return results;
}

#define GET_OP_CLASSES
#include "QuantumSSAOps.cpp.inc"
