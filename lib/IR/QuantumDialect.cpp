/* Place additional code not defined by the ODS or RRD systems here,
   e.g. operations, custom types, attributes etc. */

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringSwitch.h"

#include "InliningUtils.h"
#include "QuantumDialect.h"

#include <sstream>

using namespace mlir;
using namespace mlir::quantum;

#define EMIT_ERROR(p, m) p.emitError(p.getCurrentLocation(), m)


//===------------------------------------------------------------------------------------------===//
// Inlining Interface
//===------------------------------------------------------------------------------------------===//

// This class defines the interface for handling inlining with Quantum operations.
// We simply inherit from the base interface class and override the necessary methods.
struct QuantumInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    // Returns true if the given region 'src' can be inlined into the region 'dest' that is attached
    // to an operation registered to the current dialect. 'valueMapping' contains any remapped
    // values from within the 'src' region. This can be used to examine what values will
    // replace entry arguments into the 'src' region for example.
    bool isLegalToInline(Region *dest, Region *src, BlockAndValueMapping &mapping) const final {
        if (src->getParentOp()->getAttr("no_inline"))
            return false;
        if (!dest->getParentOfType<CircuitOp>() ||
                dest->getParentOfType<CircuitOp>().getAttr("no_inline_target"))
            return false;
        return true;
    }

    // Returns true if the given operation 'op', that is registered to this dialect, can be inlined
    // into the given region, false otherwise. 'valueMapping' contains any remapped values from
    // within the 'src' region. This can be used to examine what values may potentially replace the
    // operands to 'op'.
    bool isLegalToInline(Operation *op, Region *region, BlockAndValueMapping &mapping) const final {
        return true;
    }

    // Handle the given inlined terminator by replacing it with a new operation as necessary. This
    // overload is called when the inlined region has more than one block. The 'newDest' block
    // represents the new final branching destination of blocks within this region, i.e. operations
    // that release control to the parent operation will likely now branch to this block. Its block
    // arguments correspond to any values that need to be replaced by terminators within the
    // inlined region.
    void handleTerminator(Operation *op, Block *newDest) const final {
        assert(isa<TerminatorOp>(op) && "encoutered unknown terminator!");
    }

    // This hook is called when a terminator operation has been inlined. The only terminator in the
    // Quantum dialect is the q.term operation. It can simply be ignored, as it produces no results.
    void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final {
        assert(isa<TerminatorOp>(op) && "encoutered unknown terminator!");
    }

    // Attempt to materialize a conversion for a type mismatch between a call from this dialect,
    // and a callable region. This method should generate an operation that takes as 'input' the
    // only operand, and produces a single result of 'resultType'. If a conversion can not be
    // generated, nullptr should be returned.
    // NOTE: This hook may be invoked before the 'isLegal' checks above.
    Operation* materializeCallConversion(OpBuilder &builder, Value input, Type resultType,
                                         Location conversionLoc) const {
        if (!input.getType().isa<QuregType>() || !resultType.isa<QuregType>() ||
                !input.getType().cast<QuregType>().getNumQubits() ||
                resultType.cast<QuregType>().getNumQubits())
            return nullptr;

        OperationState castState(conversionLoc, CastRegOp::getOperationName());
        CastRegOp::build(builder, castState, resultType, input);
        Operation *cast = builder.createOperation(castState);

        return cast;
    }
};


//===------------------------------------------------------------------------------------------===//
// Dialect Definitions
//===------------------------------------------------------------------------------------------===//

// latest changes in MLIR upstream now only require this function for op, type, etc. registration
void QuantumDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "QuantumOps.cpp.inc"
    >();

    addTypes<QubitType, QuregType, U1Type, U2Type, COpType, CircType>();
    addInterfaces<QuantumInlinerInterface>();
}

namespace mlir {
namespace quantum {
namespace detail {
// This class represents the internal storage of the Quantum 'QuregType'.
struct QuregTypeStorage : public TypeStorage {
    // The `KeyTy` is a required type that provides an interface for the storage instance.
    // This type will be used when uniquing an instance of the type storage. For our Qureg
    // type, we will unique each instance on its size.
    using KeyTy = int;

    // Size of the qubit register
    llvm::Optional<int> size;

    // A constructor for the type storage instance.
    QuregTypeStorage(llvm::Optional<int> size) {
        assert(!size || *size > 1 && "Register type must have size > 1!");
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
    static QuregTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
        // Allocate the storage instance and construct it.
        llvm::Optional<int> size = key < 0 ? llvm::None : llvm::Optional<int>(key);
        return new (allocator.allocate<QuregTypeStorage>()) QuregTypeStorage(size);
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
} // end namespace quantum
} // end namespace mlir


//===------------------------------------------------------------------------------------------===//
// Method implementations of complex types
//===------------------------------------------------------------------------------------------===//

// Qureg
QuregType QuregType::get(MLIRContext *ctx, llvm::Optional<int> size) {
    // Parameters to the storage class are passed after the custom type kind.
    detail::QuregTypeStorage::KeyTy key = size ? *size : -1;
    return Base::get(ctx, key);
}

llvm::Optional<int> QuregType::getNumQubits() {
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
void QuantumDialect::printType(Type type, DialectAsmPrinter &printer) const {
    // Differentiate between the Quantum types and print accordingly.
    llvm::TypeSwitch<Type>(type)
        .Case<QubitType>([&](QubitType)   { printer << "qubit"; })
        .Case<QuregType>([&](QuregType t) { printer << "qureg<";
                                            if (auto numQubits = t.getNumQubits())
                                                printer << *numQubits;
                                            printer << ">"; })
        .Case<U1Type>   ([&](U1Type)      { printer << "u1"; })
        .Case<U2Type>   ([&](U2Type)      { printer << "u2"; })
        .Case<COpType>  ([&](COpType t)   { printer << "cop<";
                                            if (auto numCtrls = t.getNumCtrls())
                                                printer << *numCtrls << ", ";
                                            printer << t.getBaseType() << ">"; })
        .Case<CircType> ([&](CircType)    { printer << "circ"; })
        .Default([](Type) { llvm_unreachable("unrecognized type encountered in the printer!"); });
}

// Parse an instance of a type registered to the Quantum dialect.
Type QuantumDialect::parseType(DialectAsmParser &parser) const {
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
        .Case("qubit", [&] { return builder.getType<QubitType>(); })
        .Case("qureg", [&] { return parseQuregType(parser); })
        .Case("u1",    [&] { return builder.getType<U1Type>(); })
        .Case("u2",    [&] { return builder.getType<U2Type>(); })
        .Case("cop",   [&] { return parseCOpType(parser); })
        .Case("circ",  [&] { return builder.getType<CircType>(); })
        .Default([&] { return EMIT_ERROR(parser, "unrecognized quantum type!"), nullptr; })();

    return result;
}

Type QuantumDialect::parseQuregType(DialectAsmParser &parser) {
    StringRef errmsg = "error during 'Qureg' type parsing!";
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

    return parser.getBuilder().getType<QuregType>(optionalSize);
}

Type QuantumDialect::parseCOpType(DialectAsmParser &parser) {
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
                  /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

// custom parsing for the function-like CircuitOp
static ParseResult parseCircuitOp(OpAsmParser &p, OperationState &result) {
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                            impl::VariadicFlag, std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    if (impl::parseFunctionLikeOp(p, result, /*allowVariadic=*/false, buildFuncType))
        return failure();

    for (auto &region : result.regions)
        OpTrait::SingleBlockImplicitTerminator<TerminatorOp>::Impl<CircuitOp>::
            ensureTerminator(*region, p.getBuilder(), result.location);

    return success();
}


//===------------------------------------------------------------------------------------------===//
// Custom directives for declarative op printing and parsing
//===------------------------------------------------------------------------------------------===//

// custom directive to print single register accessor list
static void printSingleAccessorList(OpAsmPrinter &p,
                                    OperandRange range, TypeRange, ArrayAttr staticRange) {
    if (!staticRange.size())
        return;

    p << "[";
    int opIdx = 0;
    const char *sep = "";
    for (auto attr : staticRange) {
        p << sep;
        if (attr.dyn_cast<IntegerAttr>().getInt() == -1)
            p.printOperand(range[opIdx++]);
        else
            p.printAttributeWithoutType(attr);
        sep = ", ";
    }
    p << "]";
}

// custom directive to parse single register accessor list
static ParseResult parseSingleAccessorList(OpAsmParser &p,
                                           SmallVectorImpl<OpAsmParser::OperandType> &range,
                                           SmallVectorImpl<Type> &types, ArrayAttr &staticRange) {
    Builder builder = p.getBuilder();
    if(failed(p.parseOptionalLSquare())) {
        staticRange = builder.getI64ArrayAttr({});
        return success();
    }

    // Maximum 3 register accessors: start, size, step. Either index operands or int attributes.
    SmallVector<int64_t, 3> staticRangeValues;
    do {
        OpAsmParser::OperandType operand;
        auto res = p.parseOptionalOperand(operand);
        if (res.hasValue() && failed(res.getValue())) {
            return EMIT_ERROR(p, "unexpected failure for optional operand");
        } else if (res.hasValue()) {
            range.push_back(operand);
            types.push_back(builder.getIndexType());
            staticRangeValues.push_back(-1);
        } else {
            Attribute attr;
            if (failed(p.parseAttribute(attr, builder.getI64Type())))
                return EMIT_ERROR(p, "expected SSA value or integer");
            staticRangeValues.push_back(attr.cast<IntegerAttr>().getInt());
        }
    } while (succeeded(p.parseOptionalComma()));

    staticRange = builder.getI64ArrayAttr(staticRangeValues);

    return p.parseRSquare();
}

// custom directive to print a quatum argument with register accessors
static void printQDataArg(OpAsmPrinter &p,
                          Value qbs, OperandRange range, TypeRange, ArrayAttr staticRange) {
    p.printOperand(qbs);
    printSingleAccessorList(p, range, nullptr, staticRange);
}

// custom directive to parse a quatum argument with register accessors
static ParseResult parseQDataArg(OpAsmParser &p,
                                 OpAsmParser::OperandType &qbs,
                                 SmallVectorImpl<OpAsmParser::OperandType> &range,
                                 SmallVectorImpl<Type> &types, ArrayAttr &staticRange) {
    if (p.parseOperand(qbs) || parseSingleAccessorList(p, range, types, staticRange))
        return EMIT_ERROR(p, "expected operand with potential accessors!");

    return success();
}

// custom directive to print an optional quatum argument with register accessors
static void printOptionalQDataArg(OpAsmPrinter &p,
                                  Value qbs, OperandRange range, TypeRange, ArrayAttr staticRange) {
    if (qbs) {
        printQDataArg(p, qbs, range, nullptr, staticRange);
    }
}

// custom directive to parse an optional quatum argument with register accessors
static ParseResult parseOptionalQDataArg(OpAsmParser &p,
                                         Optional<OpAsmParser::OperandType> &qbs,
                                         SmallVectorImpl<OpAsmParser::OperandType> &range,
                                         SmallVectorImpl<Type> &types, ArrayAttr &staticRange) {
    OpAsmParser::OperandType operand;
    auto res = p.parseOptionalOperand(operand);
    if (res.hasValue() && failed(res.getValue())) {
        return EMIT_ERROR(p, "unexpected failure for optional operand");
    } else if (res.hasValue()) {
        qbs = operand;
        if (parseSingleAccessorList(p, range, types, staticRange))
            return failure();
    } else {
        staticRange = p.getBuilder().getI64ArrayAttr({});
    }

    return success();
}

// custom directive to print two optional quatum arguments with register accessors
static void printTwoQDataArg(OpAsmPrinter &p,
                             Value qbs1, OperandRange range1, TypeRange, ArrayAttr staticRange1,
                             Value qbs2, OperandRange range2, TypeRange, ArrayAttr staticRange2) {
    printOptionalQDataArg(p, qbs1, range1, nullptr, staticRange1);
    if (qbs1 && qbs2)
        p << ", ";
    printOptionalQDataArg(p, qbs2, range2, nullptr, staticRange2);
}

// custom directive to parse two optional quatum arguments with register accessors
static ParseResult parseTwoQDataArg(OpAsmParser &p,
                                    Optional<OpAsmParser::OperandType> &qbs1,
                                    SmallVectorImpl<OpAsmParser::OperandType> &range1,
                                    SmallVectorImpl<Type> &types1, ArrayAttr &staticRange1,
                                    Optional<OpAsmParser::OperandType> &qbs2,
                                    SmallVectorImpl<OpAsmParser::OperandType> &range2,
                                    SmallVectorImpl<Type> &types2, ArrayAttr &staticRange2) {
    if (parseOptionalQDataArg(p, qbs1, range1, types1, staticRange1))
        return failure();
    p.parseOptionalComma();
    if (parseOptionalQDataArg(p, qbs2, range2, types2, staticRange2))
        return failure();

    return success();
}

// custom directive to print two optional quatum arguments with register accessors
static void printMetaQDataArgs(OpAsmPrinter &p,
                               Value qbs1, OperandRange range1, TypeRange, ArrayAttr staticRange1,
                               Value qbs2, OperandRange range2, TypeRange, ArrayAttr staticRange2) {
    if (qbs1) {
        p << ", ";
        printTwoQDataArg(p, qbs1, range1, nullptr, staticRange1,
                            qbs2, range2, nullptr, staticRange2);
    }
}

// custom directive to parse two optional quatum arguments with register accessors
static ParseResult parseMetaQDataArgs(OpAsmParser &p,
                                      Optional<OpAsmParser::OperandType> &qbs1,
                                      SmallVectorImpl<OpAsmParser::OperandType> &range1,
                                      SmallVectorImpl<Type> &types1, ArrayAttr &staticRange1,
                                      Optional<OpAsmParser::OperandType> &qbs2,
                                      SmallVectorImpl<OpAsmParser::OperandType> &range2,
                                      SmallVectorImpl<Type> &types2, ArrayAttr &staticRange2) {
    p.parseOptionalComma();
    if (parseTwoQDataArg(p, qbs1, range1, types1, staticRange1,
                            qbs2, range2, types2, staticRange2))
        return failure();

    return success();
}

// custom directive to print two optional quatum arguments with register accessors
static void printCtrlArgs(OpAsmPrinter &p,
                          Value ctrls, OperandRange crange, TypeRange, ArrayAttr staticCrange,
                          Value qbs1, OperandRange range1, TypeRange, ArrayAttr staticRange1,
                          Value qbs2, OperandRange range2, TypeRange, ArrayAttr staticRange2) {
    printQDataArg(p, ctrls, crange, nullptr, staticCrange);
    printMetaQDataArgs(p, qbs1, range1, nullptr, staticRange1, qbs2, range2, nullptr, staticRange2);
}

// custom directive to parse two optional quatum arguments with register accessors
static ParseResult parseCtrlArgs(OpAsmParser &p,
                                 OpAsmParser::OperandType &ctrls,
                                 SmallVectorImpl<OpAsmParser::OperandType> &crange,
                                 SmallVectorImpl<Type> &ctypes, ArrayAttr &staticCrange,
                                 Optional<OpAsmParser::OperandType> &qbs1,
                                 SmallVectorImpl<OpAsmParser::OperandType> &range1,
                                 SmallVectorImpl<Type> &types1, ArrayAttr &staticRange1,
                                 Optional<OpAsmParser::OperandType> &qbs2,
                                 SmallVectorImpl<OpAsmParser::OperandType> &range2,
                                 SmallVectorImpl<Type> &types2, ArrayAttr &staticRange2) {
    if (parseQDataArg(p, ctrls, crange, ctypes, staticCrange) ||
            parseMetaQDataArgs(p, qbs1, range1, types1, staticRange1,
                                  qbs2, range2, types2, staticRange2))
        return failure();

    return success();
}

// custom directive to print arbitrary sequence of arguments in circuit calls
static void printArbitraryArgs(OpAsmPrinter &p,
                               OperandRange args, OperandRange ranges, TypeRange,
                               ArrayAttr staticRanges, ArrayAttr sizeParams) {
    const char *sep = "";
    for (size_t sizeIdx = 0, argIdx = 0, rangeIdx = 0; sizeIdx < sizeParams.size(); sizeIdx++) {
        p << sep; sep = ", ";

        if (sizeParams[sizeIdx].cast<IntegerAttr>().getInt() != -1) {
            p.printAttributeWithoutType(sizeParams[sizeIdx]);
            continue;
        }

        ArrayAttr subArray = staticRanges[argIdx].dyn_cast<ArrayAttr>();
        OperandRange subRange = ranges.drop_front(rangeIdx);
        rangeIdx += std::count_if(subArray.begin(), subArray.end(),
            [](Attribute a){ return a.dyn_cast<IntegerAttr>().getInt() == -1; });
        printQDataArg(p, args[argIdx++], subRange, nullptr, subArray);
    }
}

// custom directive to parse arbitrary sequence of arguments in circuit calls
static ParseResult parseArbitraryArgs(OpAsmParser &p,
                                      SmallVectorImpl<OpAsmParser::OperandType> &args,
                                      SmallVectorImpl<OpAsmParser::OperandType> &ranges,
                                      SmallVectorImpl<Type> &rangesTypes, ArrayAttr &staticRanges,
                                      ArrayAttr &sizeParams) {
    Builder builder = p.getBuilder();
    SmallVector<Attribute, 4> staticRangesValues;
    SmallVector<Attribute, 2> sizeParamsValues;

    do {
        Optional<OpAsmParser::OperandType> operand(llvm::None);
        ArrayAttr staticRange;
        IntegerAttr sizeParam;

        if (parseOptionalQDataArg(p, operand, ranges, rangesTypes, staticRange))
            return failure();
        if (operand) {
            args.push_back(*operand);
            staticRangesValues.push_back(staticRange);
            sizeParamsValues.push_back(builder.getI64IntegerAttr(-1));
        } else {
            if (p.parseAttribute<IntegerAttr>(sizeParam))
                return EMIT_ERROR(p, "expected next argument after comma!");
            if (sizeParam.getInt() < 0)
                return EMIT_ERROR(p, "negative size param during parsing!");
            sizeParamsValues.push_back(sizeParam);
        }
    } while (succeeded(p.parseOptionalComma()));

    staticRanges = builder.getArrayAttr(staticRangesValues);
    sizeParams = builder.getArrayAttr(sizeParamsValues);

    return success();
}

// custom directive to print quantum U1 gate type signature
static void printU1TypeSig(OpAsmPrinter &p, Type qbsType, Type opType) {
    if (qbsType) {
        p << ": ";
        p.printType(qbsType);
    }
    if (opType) {
        p << "-> ";
        p.printType(opType);
    }
}

// custom directive to parse quantum U1 gate type signature
static ParseResult parseU1TypeSig(OpAsmParser &p, Type &qbsType, Type &opType) {
    if (succeeded(p.parseOptionalColon()))
        if(p.parseType(qbsType))
            return EMIT_ERROR(p, "expected type after ':' token!");
    if (succeeded(p.parseOptionalArrow()))
        if (p.parseType(opType))
            return EMIT_ERROR(p, "expected type after '->' token!");
    return success();
}

// custom directive to print quantum U2 gate type signature
static void printU2TypeSig(OpAsmPrinter &p, Type qbs1Type, Type qbs2Type, Type opType) {
    if (qbs1Type || qbs2Type)
        p << ": ";
    if (qbs1Type)
        p.printType(qbs1Type);
    if (qbs1Type && qbs2Type)
        p << ", ";
    if (qbs2Type)
        p.printType(qbs2Type);
    if (opType) {
        p << "-> ";
        p.printType(opType);
    }
}

// custom directive to parse quantum U2 gate type signature
static ParseResult parseU2TypeSig(OpAsmParser &p, Type &qbs1Type, Type &qbs2Type, Type &opType) {
    if (succeeded(p.parseOptionalColon()))
        if(p.parseType(qbs1Type) || p.parseComma() || p.parseType(qbs2Type))
            return EMIT_ERROR(p, "expected 2 types after ':' token!");
    if (succeeded(p.parseOptionalArrow()))
        if (p.parseType(opType))
            return EMIT_ERROR(p, "expected type after '->' token!");
    return success();
}

// custom directive to print quantum meta gate type signature
static void printMetaTypeSig(OpAsmPrinter &p, Type qbs1Type, Type qbs2Type, Type opType) {
    if (qbs1Type || qbs2Type)
        p << ", ";
    if (qbs1Type)
        p.printType(qbs1Type);
    if (qbs1Type && qbs2Type)
        p << ", ";
    if (qbs2Type)
        p.printType(qbs2Type);
    if (opType) {
        p << "-> ";
        p.printType(opType);
    }
}

// custom directive to parse quantum meta gate type signature
static ParseResult parseMetaTypeSig(OpAsmParser &p, Type &qbs1Type, Type &qbs2Type, Type &opType) {
    if (succeeded(p.parseOptionalComma())) {
        if(p.parseType(qbs1Type))
            return EMIT_ERROR(p, "expected type after ',' token!");
        if (succeeded(p.parseOptionalComma()))
            if(p.parseType(qbs2Type))
                return EMIT_ERROR(p, "expected type after ',' token!");
    }
    if (succeeded(p.parseOptionalArrow()))
        if (p.parseType(opType))
            return EMIT_ERROR(p, "expected type after '->' token!");
    return success();
}

// custom directive to print quantum meta gate type signature
static void printCtrlTypeSig(OpAsmPrinter &p,
                             Type ctrlsType, Type qbs1Type, Type qbs2Type, Type opType) {
    p.printType(ctrlsType);
    printMetaTypeSig(p, qbs1Type, qbs2Type, opType);
}

// custom directive to parse quantum meta gate type signature
static ParseResult parseCtrlTypeSig(OpAsmParser &p,
                                    Type &ctrlsType, Type &qbs1Type, Type &qbs2Type, Type &opType) {
    if (p.parseType(ctrlsType))
        return EMIT_ERROR(p, "expected type after ',' token!");
    if (parseMetaTypeSig(p, qbs1Type, qbs2Type, opType))
        return failure();

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

// custom directive to print a floating point parameter
static void printFloatParam(OpAsmPrinter &p, Value dynArg, Type dynType, Attribute staticArg) {
    if (dynArg) {
        p.printOperand(dynArg);
        p << ": ";
        p.printType(dynType);
    }
    if (staticArg)
        p.printAttributeWithoutType(staticArg);
}

// custom directive to parse a floating point parameter
static ParseResult parseFloatParam(OpAsmParser &p,
                                   Optional<OpAsmParser::OperandType> &dynArg, Type &dynType,
                                   Attribute &staticArg) {
    OpAsmParser::OperandType operand;
    auto res = p.parseOptionalOperand(operand);
    if (res.hasValue() && failed(res.getValue())) {
        return EMIT_ERROR(p, "unexpected failure for optional operand!");
    } else if (res.hasValue()) {
        dynArg = operand;
        if (p.parseColonType(dynType))
            return EMIT_ERROR(p, "dynamic parem requires type!");
    } else if (p.parseAttribute(staticArg, p.getBuilder().getF64Type())) {
        return EMIT_ERROR(p, "expected floating point parameter!");
    }

    return success();
}


//===------------------------------------------------------------------------------------------===//
// Additional op method definitions
//===------------------------------------------------------------------------------------------===//

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
// Auto-generated op & interface definitions
//===------------------------------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "QuantumOps.cpp.inc"
