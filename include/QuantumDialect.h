#ifndef MLIR_QUANTUM_DIALECT_H
#define MLIR_QUANTUM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace quantum {
namespace detail {
struct QuregTypeStorage;
struct COpTypeStorage;
} // end namespace detail

// Add autogenerated header files from the ODS system, which include operation declarations
#include "QuantumOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "QuantumOps.h.inc"

// MLIR types are represented by a *unique* value (for efficiency), which means all types must be
// statically assigned a value by entering them in a global registry (DialectSymbolRegistry.def).
// We can use PRIVATE_EXPERIMENTAL_0 (to 9) reserved for prototyping (each with a range of 256).
namespace QuantumTypes {
enum Kinds {
    Qubit = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
    Qureg,
    Op,
    COp,
    Circ
};
}

// This class represents individual qubits. It derives from mlir::detail::StorageUserBase
// (aliased to mlir::Type::TypeBase) as all custom types must. The template parameters
// consist of the concrete type (QubitType), and the base class to use (Type).
class QubitType : public Type::TypeBase<QubitType, mlir::Type> {
public:
    // 'Base' is a type alias for the templated mlir::detail::StorageUserBase class,
    // just as TypeBase is (but inside mlir::detail::StorageUserBase instead of mlir::Type).
    // Thus we expose the contructors from 'TypeBase' here.
    using Base::Base;

    // This is used to support type inquiry through isa, cast, and dyn_cast.
    static bool kindof(unsigned kind) { return kind == QuantumTypes::Qubit; }

    // This is used to get an instance of the 'QubitType'. Given that this is
    // a parameterless type, it just needs the context for uniquing purposes.
    static QubitType get(mlir::MLIRContext *ctx) {
        return Base::get(ctx, QuantumTypes::Qubit);
    }
};

// This class represents a register of qubits, with a minimum of one qubit. For such a
// parametrized type, we also need to the type with a custom storage class (QuregTypeStorage).
class QuregType : public Type::TypeBase<QuregType, Type, detail::QuregTypeStorage> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::Qureg; }

    // This method is a bit different from the simple type above. It takes the parameters
    // required for uniquing which will be passed to the container (storage) class.
    // It will also assert that all of the construction invariants are satisfied by
    // calling 'verifyContructionInvariants'. To gracefully handle errors, use getChecked.
    static QuregType get(mlir::MLIRContext *ctx, unsigned size);

    // Return the register size
    unsigned getNumQubits();
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

    static COpType get(mlir::MLIRContext *ctx, unsigned nctrl);

    unsigned getNumCtrls();
};

// This class represents a quantum circuit, that is, a collection of quantum ops.
class CircType : public Type::TypeBase<CircType, mlir::Type> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == QuantumTypes::Circ; }

    static CircType get(mlir::MLIRContext *ctx) {
        return Base::get(ctx, QuantumTypes::Circ);
    }
};
} // namespace quantum
} // namespace mlir

#endif // MLIR_QUANTUM_DIALECT_H
