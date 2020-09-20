#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"

def APFloatAttr : Attr<CPred<"$_self.isa<FloatAttr>()">, "arbitrary float atribute"> {
    let storageType = [{ FloatAttr }];
    let returnType = [{ APFloat }];
}

class 2dArrayAttrBase<Attr element, string description> : ArrayAttrBase<
        And<[
            CPred<"$_self.isa<ArrayAttr>()">,
            Concat<"llvm::all_of($_self.cast<ArrayAttr>(), [](Attribute attr) { return ",
                And<[
                    CPred<"attr.isa<ArrayAttr>()">,
                    Concat<"llvm::all_of(attr.cast<ArrayAttr>(), [](Attribute attr) { return ",
                        SubstLeaves<"$_self", "attr", element.predicate>, "; })">
                ]>,  "; })">
        ]>, description> {
    let constBuilderCall = "$_builder.getArrayAttr($0)";
    Attr elementAttr = element;
}
def 2dI64ArrayAttr : 2dArrayAttrBase<I64Attr, "2D 64-bit integer array attribute">;

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

// Provide custom operation types representing a gate or circuit
def Op_Type : Quantum_Type<"OpType", "Quantum op type">;
def ControlledOp_Type : Quantum_Type<"COpType", "Quantum c-op type">;
def Circuit_Type : Quantum_Type<"CircType", "Quantum circuit type">;

// Provide aggregate types that might be used in ops accepting multiple types
def QData_Type : AnyTypeOf<[Qubit_Type, Qureg_Type]>;
def QOp_Type : AnyTypeOf<[Op_Type, ControlledOp_Type, Circuit_Type]>;
def QCirc_Type : AnyTypeOf<[Circuit_Type, ControlledOp_Type]>;

// Define new dialect traits
def Unitary : NativeOpTrait<"UnitaryTrait">;
def Hermitian : NativeOpTrait<"HermitianTrait">;
def Meta : NativeOpTrait<"MetaOpTrait">;
def ImplicitQuantumTerminator : SingleBlockImplicitTerminator<"TerminatorOp">;

#endif // QUANTUM_DIALECT
