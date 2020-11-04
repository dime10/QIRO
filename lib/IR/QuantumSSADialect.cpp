/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionImplementation.h"
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

    addTypes<QstateType, RstateType, U1Type, U2Type, COpType, CircType>();
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
    using KeyTy = std::pair<int, Type>;

    llvm::Optional<int> nctrl;
    Type baseType;

    COpTypeStorage(llvm::Optional<int> nctrl, Type baseType) {
        assert(!nctrl || *nctrl > 0 && "Number of controls must be > 0");
        if (baseType)
            assert(baseType.isa<U1Type>() || baseType.isa<U2Type>() || baseType.isa<CircType>() &&
                   "Base type of controlled op can only be supported quantum operations!");
        this->nctrl = nctrl;
        this->baseType = baseType;
    }

    bool operator==(const KeyTy &key) const {
        return nctrl == (key.first < 0 ? llvm::None : llvm::Optional<int>(key.first))
            && key.second == baseType;
    }

    static COpTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        llvm::Optional<int> nctrl = key.first < 0 ? llvm::None : llvm::Optional<int>(key.first);
        return new (allocator.allocate<COpTypeStorage>()) COpTypeStorage(nctrl, key.second);
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
COpType COpType::get(MLIRContext *ctx, llvm::Optional<int> nctrl, Type baseType) {
    // Parameters to the storage class are passed after the custom type kind.
    detail::COpTypeStorage::KeyTy key = {nctrl ? *nctrl : -1, baseType};
    return Base::get(ctx, key);
}

llvm::Optional<int> COpType::getNumCtrls() {
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
void QuantumSSADialect::printType(Type type, DialectAsmPrinter &printer) const {
    // Differentiate between the Quantum types and print accordingly.
    llvm::TypeSwitch<Type>(type)
        .Case<QstateType> ([&](QstateType)    { printer << "qstate"; })
        .Case<RstateType> ([&](RstateType t)  { printer << "rstate<";
                                                if (auto numQubits = t.getNumQubits())
                                                    printer << *numQubits;
                                                printer << ">"; })
        .Case<U1Type>   ([&](U1Type)          { printer << "u1"; })
        .Case<U2Type>   ([&](U2Type)          { printer << "u2"; })
        .Case<COpType>    ([&](COpType t)     { printer << "cop<";
                                                if (auto numCtrls = t.getNumCtrls())
                                                    printer << *numCtrls << ", ";
                                                printer << t.getBaseType() << ">"; })
        .Case<CircType>   ([&](CircType)      { printer << "circ"; })
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
        .Case("u1",     [&] { return builder.getType<U1Type>(); })
        .Case("u2",     [&] { return builder.getType<U2Type>(); })
        .Case("cop",    [&] { return parseCOpType(parser); })
        .Case("circ",   [&] { return builder.getType<CircType>(); })
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
    llvm::Optional<int> optionalNctrl;
    int nctrl;

    if (parser.parseLess())
        return EMIT_ERROR(parser, errmsg), nullptr;

    auto res = parser.parseOptionalInteger<int>(nctrl);
    if (res.hasValue() && failed(res.getValue()))
        return EMIT_ERROR(parser, errmsg), nullptr;
    else if (res.hasValue())
        if (parser.parseComma())
            return EMIT_ERROR(parser, errmsg), nullptr;
    optionalNctrl = res.hasValue() ? llvm::Optional<int>(nctrl) : llvm::None;

    if (parser.parseType(baseType))
        return EMIT_ERROR(parser, errmsg), nullptr;

    if (parser.parseGreater())
        return EMIT_ERROR(parser, errmsg), nullptr;

    if (baseType && !(baseType.isa<U1Type>() || baseType.isa<U2Type>() || baseType.isa<CircType>()))
        return EMIT_ERROR(parser, "Base type of COp must be either 'u1', 'u2', or 'circ'!"),
               nullptr;

    return parser.getBuilder().getType<COpType>(optionalNctrl, baseType);
}


//===------------------------------------------------------------------------------------------===//
// Custom parsing for special operations types
//===------------------------------------------------------------------------------------------===//

// custom printing for the function-like CircuitOp
static void print(OpAsmPrinter &p, CircuitOp op) {
    p << CircuitOp::getOperationName() << " ";
    p.printSymbolName(op.getName());

    FunctionType type = op.OpTrait::FunctionLike<CircuitOp>::getType();
    impl::printFunctionSignature(p, op.getOperation(), type.getInputs(),
                                 /*isVariadic=*/false, type.getResults());
    impl::printFunctionAttributes(p, op.getOperation(), type.getNumInputs(), type.getNumResults());

    p.printRegion(op.OpTrait::FunctionLike<CircuitOp>::getBody(),
                  /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
}

// custom parsing for the function-like CircuitOp
static ParseResult parseCircuitOp(OpAsmParser &p, OperationState &result) {
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                            impl::VariadicFlag, std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    if (impl::parseFunctionLikeOp(p, result, /*allowVariadic=*/false, buildFuncType))
        return failure();

    return success();
}


//===------------------------------------------------------------------------------------------===//
// Custom directives for declarative op printing and parsing
//===------------------------------------------------------------------------------------------===//

// custom directive to print one or two optional operands
static void printOneOrTwoQbs(OpAsmPrinter &p, Value qbs2, Value qbs) {
    if (qbs)
        p << ", ";
    if (qbs2) {
        p.printOperand(qbs2);
        p << ", ";
    }
    if (qbs)
        p.printOperand(qbs);
}

// custom directive to parse one or two optional operands
static ParseResult parseOneOrTwoQbs(OpAsmParser &p,
                                    Optional<OpAsmParser::OperandType> &qbs2,
                                    Optional<OpAsmParser::OperandType> &qbs) {
    if (failed(p.parseOptionalComma()))
        return success();

    OpAsmParser::OperandType operand;
    if (p.parseOperand(operand))
        return EMIT_ERROR(p, "expected operand after comma!");
    if (succeeded(p.parseOptionalComma())) {
        qbs2 = operand;
        if (p.parseOperand(operand))
            return EMIT_ERROR(p, "expected operand after comma!");
    }
    qbs = operand;

    return success();
}

// custom directive to print a integer parameter
static void printIntParam(OpAsmPrinter &p, Value dynArg, Type, Attribute staticArg) {
    if (dynArg)
        p.printOperand(dynArg);
    if (staticArg)
        p.printAttributeWithoutType(staticArg);
}

// custom directive to parse a integer parameter
static ParseResult parseIntParam(OpAsmParser &p,
                                 Optional<OpAsmParser::OperandType> &dynArg, Type &dynType,
                                 Attribute &staticArg) {
    OpAsmParser::OperandType operand;
    auto res = p.parseOptionalOperand(operand);
    if (res.hasValue() && failed(res.getValue())) {
        return EMIT_ERROR(p, "unexpected failure for optional operand!");
    } else if (res.hasValue()) {
        dynArg = operand;
        dynType = p.getBuilder().getIndexType();
    } else if (p.parseAttribute(staticArg, p.getBuilder().getI64Type())) {
        return EMIT_ERROR(p, "expected integer parameter!");
    }

    return success();
}

//===------------------------------------------------------------------------------------------===//
// Additional op method definitions
//===------------------------------------------------------------------------------------------===//

// Optional CircuitOp verification methods
LogicalResult CircuitOp::verifyType() {
    auto retType = this->getType().getResults().begin();
    for (auto argType : this->getType().getInputs()) {
        if (argType.isa<QstateType>() || argType.isa<RstateType>()) {
            if (retType == this->getType().getResults().end())
                return this->emitOpError() << "has too few return values! Every QData argument "
                                              "needs to be returned in its updated state.";
            if (*retType != argType)
                return this->emitOpError() << "has mismatched return type! Requires: " << argType
                                           << ". Got: " << *retType << ".";
            retType++;
        }
    }

    return success();
}

LogicalResult CircuitOp::verifyBody() {
    Operation *term = this->gates().front().getTerminator();
    if (cast<ReturnStateOp>(term).retvals().getType() != this->getType().getResults())
        return this->emitOpError() << "must return updated state for all QData arguments!";

    return success();
};

// The below methods are required for the CallableOpInterface
Region *CircuitOp::getCallableRegion() {
  return &this->getRegion();
}

ArrayRef<Type> CircuitOp::getCallableResults() {
    return {};
}

// The below methods are required for the CallOpInterface
CallInterfaceCallable CallCircOp::getCallableForCallee() {
    return this->getAttrOfType<SymbolRefAttr>("circref");
}

OperandRange CallCircOp::getArgOperands() {
    return this->args();
}


//===------------------------------------------------------------------------------------------===//
// Auto-generated op definitions
//===------------------------------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QuantumSSAOps.cpp.inc"
