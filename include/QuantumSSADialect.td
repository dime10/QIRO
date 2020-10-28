#ifndef QUANTUMSSA_DIALECT
#define QUANTUMSSA_DIALECT

include "mlir/IR/OpBase.td"

def APFloatAttr : Attr<CPred<"$_self.isa<FloatAttr>()">, "arbitrary float atribute"> {
    let storageType = [{ FloatAttr }];
    let returnType = [{ APFloat }];
}

// Define the basic properties of our dialect
def QuantumSSA_Dialect : Dialect {
    let name = "qs";
    let summary = "An MLIR dialect for quantum circuits in SSA form.";
    let description = [{
        This dialect is a quantum program representation with value semantics,
        that is, quantum memory is represent via state values, and each operation
        generates a new state value. This dialect better integrates with existing
        MLIR infrastructure, but it is susceptible to represent unphysical quantum
        programs, such as a violation of the no-cloning theorem.
        However, by using this dialect internally as a target for transformation
        from the memory semantics "Quantum" dialect, where all operations act via
        side-effects, we are guarenteed that the program does not violate the no-
        cloning theorem, as this implicit in the structure of memory semantics.
    }];
    let cppNamespace = "mlir::quantumssa";
    let extraClassDeclaration = [{
        static Type parseRstateType(DialectAsmParser &);
        static Type parseCOpType(DialectAsmParser &);
    }];
}

// Base class for all operations in this dialect
class QuantumSSA_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<QuantumSSA_Dialect, mnemonic, traits>;
// Base class for all types in this dialect
class QuantumSSA_Type<string cppTypeName, string description> :
        DialectType<QuantumSSA_Dialect, CPred<"$_self.isa<" # cppTypeName # ">()">, description>;
// NOTE: Use DialectType instead of Type, otherwise ODS will not declare parse/print functions.

// Provide definitions of custom data types that support value semantics
def Qstate_Type : QuantumSSA_Type<"QstateType", "Qubit state at a specific time">;
def Rstate_Type : QuantumSSA_Type<"RstateType", "(Qubit) Register state at a specific time">;

// Provide custom operation types representing a gate or circuit
def U1_Type : QuantumSSA_Type<"U1Type", "U1 native gate type">;
def U2_Type : QuantumSSA_Type<"U2Type", "U2 native gate type">;
def ControlledOp_Type : QuantumSSA_Type<"COpType", "Quantum c-op type">;
def Circuit_Type : QuantumSSA_Type<"CircType", "Quantum circuit type">;

// Provide aggregate types that might be used in ops accepting multiple types
def QData_Type : AnyTypeOf<[Qstate_Type, Rstate_Type]>;
def QGate_Type : AnyTypeOf<[U1_Type, U2_Type]>;
def QOp_Type : AnyTypeOf<[QGate_Type, ControlledOp_Type, Circuit_Type]>;
def QCirc_Type : AnyTypeOf<[Circuit_Type, ControlledOp_Type]>;

// Define new dialect traits
def Unitary : NativeOpTrait<"UnitaryTrait">;
def Hermitian : NativeOpTrait<"HermitianTrait">;
def Meta : NativeOpTrait<"MetaOpTrait">;
def ImplicitQuantumTerminator : SingleBlockImplicitTerminator<"TerminatorOp">;

#endif // QUANTUMSSA_DIALECT
