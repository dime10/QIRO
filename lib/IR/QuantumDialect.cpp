/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "mlir/IR/DialectImplementation.h"

#include "QuantumDialect.h"

using namespace mlir;
using namespace mlir::quantum;


//===------------------------------------------------------------------------------------------===//
// Dialect Definitions
//===------------------------------------------------------------------------------------------===//

// Define the Dialect contructor. This is the point of registration of
// all custom types, operations, attributes, etc. for the dialect.
QuantumDialect::QuantumDialect(mlir::MLIRContext *ctx) : mlir::Dialect("q", ctx) {
    // these templated methods are from the mlir::Dialect class
    addOperations<
        #define GET_OP_LIST
        #include "QuantumOps.cpp.inc"
    >();

    addTypes<QubitType, QuregType, QlistType, OpType, COpType, CircType>();  // in mlir::quantum

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
    QuregTypeStorage(unsigned size) {
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
    static QuregTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        // Allocate the storage instance and construct it.
        return new (allocator.allocate<QuregTypeStorage>()) QuregTypeStorage(key);
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
            assert(baseType.isa<OpType>() || baseType.isa<CircType>() &&
                   "Base type of controlled op can only be supported quantum operations!");
        this->nctrl = nctrl;
        this->baseType = baseType;
    }

    bool operator==(const KeyTy &key) const { return key.first == nctrl && key.second == baseType; }

    static COpTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<COpTypeStorage>()) COpTypeStorage(key.first, key.second);
    }
};
} // end namespace detail
} // end namespace quantum
} // end namespace mlir


//===------------------------------------------------------------------------------------------===//
// Method implementations of complex types
//===------------------------------------------------------------------------------------------===//

// Qureg
QuregType QuregType::get(mlir::MLIRContext *ctx, unsigned size) {
    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, QuantumTypes::Qureg, size);
}

unsigned QuregType::getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->size;
}

// COp
COpType COpType::get(mlir::MLIRContext *ctx, unsigned nctrl, mlir::Type baseType) {
    // Parameters to the storage class are passed after the custom type kind.
    return Base::get(ctx, QuantumTypes::COp, nctrl, baseType);
}

unsigned COpType::getNumCtrls() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->nctrl;
}

Type COpType::getBaseType() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->baseType;
}


//===------------------------------------------------------------------------------------------===//
// Dialect types printing and parsing
//===------------------------------------------------------------------------------------------===//

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
    case QuantumTypes::Qlist:
        printer << "qlist";
        break;
    case QuantumTypes::Op:
        printer << "op";
        break;
    case QuantumTypes::COp: {
        COpType ctype = type.cast<COpType>();
        printer << "cop<" << ctype.getNumCtrls();
        if (ctype.getBaseType())
            printer << ", " << ctype.getBaseType();
        printer << ">";
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
    if (parser.parseKeyword(&keyword))
        return Type();

    if (keyword == "qubit")
        return QubitType::get(this->getContext());
    if (keyword == "qureg") {
        unsigned size;
        if (parser.parseLess())
            return nullptr;
        if (parser.parseInteger<unsigned>(size))
            return nullptr;
        if (parser.parseGreater())
            return nullptr;
        return QuregType::get(this->getContext(), size);
    }
    if (keyword == "qlist")
        return QlistType::get(this->getContext());
    if (keyword == "op")
        return OpType::get(this->getContext());
    if (keyword == "cop") {
        unsigned nctrl;
        Type baseType = nullptr;
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
    if (keyword == "circ")
        return CircType::get(this->getContext());

    parser.emitError(parser.getNameLoc(), "Unrecognized quantum type!");
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
    p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"name"});
    p << " -> " << op.getType();
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
    if (p.parseArrow() || p.getCurrentLocation(&trailingTypeLoc) || p.parseType(type))
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
        p << " -> ";
        p.printType(op.op().getType());
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
        p << " -> ";
        p.printType(op.op().getType());
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
        p << " -> ";
        p.printType(op.op().getType());
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
    if (op.op()) {
        p << " -> ";
        p.printType(op.op().getType());
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
    if (op.op()) {
        p << " -> ";
        p.printType(op.op().getType());
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
    if (op.op()) {
        p << " -> ";
        p.printType(op.op().getType());
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
// Additional implementations of OpInterface methods
//===------------------------------------------------------------------------------------------===//

// Return the callee, required by the call interface.
CallInterfaceCallable ParametricCircuitOp::getCallableForCallee() {
    return getAttrOfType<SymbolRefAttr>("callee");
}

// Get the arguments to the called function, required by the call interface.
Operation::operand_range ParametricCircuitOp::getArgOperands() {
    return qbs();
}

#define GET_INTERFACE_CLASSES
#include "QuantumInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "QuantumOps.cpp.inc"
