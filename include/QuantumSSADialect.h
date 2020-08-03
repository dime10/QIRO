#ifndef MLIR_QUANTUMSSA_DIALECT_H
#define MLIR_QUANTUMSSA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

#include "QuantumDialect.h" // required for the unitary, hermitian, and metaop traits

namespace mlir {
namespace quantumssa {
namespace detail {

struct RstateTypeStorage;
struct COpTypeStorage;
struct FunCircTypeStorage;

} // end namespace detail

// Add autogenerated header files from the ODS system, which include operation declarations
#include "QuantumSSAOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "QuantumSSAOps.h.inc"

// MLIR types are represented by a *unique* value (for efficiency), which means all types must be
// statically assigned a value by entering them in a global registry (DialectSymbolRegistry.def).
// We can use PRIVATE_EXPERIMENTAL_0 (to 9) reserved for prototyping (each with a range of 256).
namespace QuantumTypes {
enum Kinds {
    Qstate = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE,
    Rstate,
    Lstate,
    Op,
    COp,
    FunCirc
};
}

// This class represents the quantum state of one qubit at a single point in time.
class QstateType : public Type::TypeBase<QstateType, mlir::Type> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::Qstate; }

    static QstateType get(mlir::MLIRContext *ctx) { return Base::get(ctx, QuantumTypes::Qstate); }
};

// This class represents the state of a quantum regist, which has a static size.
class RstateType : public Type::TypeBase<RstateType, mlir::Type, detail::RstateTypeStorage> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::Rstate; }

    static RstateType get(mlir::MLIRContext *ctx, unsigned size);

    unsigned RstateType::getNumQubits();
};

// This class represents the state of quantum list, i.e. qubit register without declared size.
class LstateType : public Type::TypeBase<LstateType, mlir::Type> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::Lstate; }

    static LstateType get(mlir::MLIRContext *ctx) { return Base::get(ctx, QuantumTypes::Lstate); }
};

// This class represents a singular quantum operation (such as a gate).
class OpType : public Type::TypeBase<OpType, mlir::Type> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::Op; }

    static OpType get(mlir::MLIRContext *ctx) {
        return Base::get(ctx, QuantumTypes::Op);
    }
};

// This class represents controlled operations, where the underlying operation
// could be a single op or an entire circuit.
class COpType : public Type::TypeBase<COpType, mlir::Type, detail::COpTypeStorage> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::COp; }

    static COpType get(mlir::MLIRContext *ctx, unsigned nctrl, Type baseType);

    unsigned getNumCtrls();

    Type getBaseType();
};

// This class represents a quantum circuit, that is, a collection of quantum ops.
class FunCircType : public Type::TypeBase<FunCircType, mlir::Type, detail::FunCircTypeStorage> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::FunCirc; }

    static FunCircType get(mlir::MLIRContext *ctx, FunctionType funtype);

    FunctionType getFunType();
};
} // namespace quantumssa
} // namespace mlir

#endif // MLIR_QUANTUMSSA_DIALECT_H
