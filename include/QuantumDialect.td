#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"

def APFloatAttr : Attr<CPred<"$_self.isa<FloatAttr>()">, "arbitrary float atribute"> {
    let storageType = [{ FloatAttr }];
    let returnType = [{ APFloat }];
}

// Define the basic properties of our dialect
def Quantum_Dialect : Dialect {
    let name = "q";
    let summary = "An MLIR dialect for quantum circuit compilation.";
    let description = [{
        This dialect is an example of a dialect designed to allow for
        the compilation and optimization of quantum programs.
    }];
    let cppNamespace = "quantum";
}

// Base class for all operations in this dialect
class Quantum_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Quantum_Dialect, mnemonic, traits>;
// Base class for all types in this dialect
class Quantum_Type<string cppTypeName, string description> :
        DialectType<Quantum_Dialect, CPred<"$_self.isa<" # cppTypeName # ">()">, description>;
// NOTE: Use DialectType instead of Type, otherwise ODS will not declare parse/print functions.

// Provide definitions for custom data types that support memory semantics
def Qubit_Type : Quantum_Type<"QubitType", "Quantum qubit type">;
def Qureg_Type : Quantum_Type<"QuregType", "Quantum register type">;
def Qlist_Type : Quantum_Type<"QlistType", "Quantum register of undeclared size type">;

// Provide definitions of custom data types that support value semantics
def Qstate_Type : Quantum_Type<"QstateType", "Quatum state on a single qubit at a specific time">;

// Provide custom operation types representing a gate or circuit
def Op_Type : Quantum_Type<"OpType", "Quantum op type">;
def ControlledOp_Type : Quantum_Type<"COpType", "Quantum c-op type">;
def Circuit_Type : Quantum_Type<"CircType", "Quantum circuit type">;
def FunctionCircuit_Type : Quantum_Type<"FunCircType", "Function circuit type">;

// Provide aggregate types that might be used in ops accepting multiple types
def QData_Type : AnyTypeOf<[Qubit_Type, Qureg_Type]>;
def QDataX_Type : AnyTypeOf<[QData_Type, Qlist_Type]>;
def QOp_Type : AnyTypeOf<[Op_Type, ControlledOp_Type, Circuit_Type, FunctionCircuit_Type]>;
def QCirc_Type : AnyTypeOf<[Circuit_Type, ControlledOp_Type]>;
def QFunCirc_Type : AnyTypeOf<[FunctionCircuit_Type, ControlledOp_Type]>;
def QStateOrOp_Type : AnyTypeOf<[Qstate_Type, Op_Type]>;
def QStateOrCOp_Type : AnyTypeOf<[Qstate_Type, ControlledOp_Type]>;
def QStateOrQOp_Type : AnyTypeOf<[Qstate_Type, QOp_Type]>;
def QStateOrBit_Type : AnyTypeOf<[Qstate_Type, Qubit_Type]>;
def QStateOrData_Type : AnyTypeOf<[Qstate_Type, QData_Type]>;

// Define new dialect traits
def Unitary : NativeOpTrait<"UnitaryTrait">;
def Hermitian : NativeOpTrait<"HermitianTrait">;
def Meta : NativeOpTrait<"MetaOpTrait">;
def ImplicitQuantumTerminator : SingleBlockImplicitTerminator<"TerminatorOp">;

#endif // QUANTUM_DIALECT
