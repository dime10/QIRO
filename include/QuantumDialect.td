#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"

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

// Provide definitions for the custom Quantum types to ODS. Syntax:
//   def <TypeName> : Type<<predicate identifying type>, <description>>;
//   `$_self` will be replaced with the entity this predicate is attached to.
def Quantum_QubitType : Type<CPred<"$_self.isa<QubitType>()">, "Quantum qubit type">;
def Quantum_QuregType : Type<CPred<"$_self.isa<QuregType>()">, "Quantum register type">;

// Provide a definition of the types that are used within the Toy dialect.
def Quantum_Type : AnyTypeOf<[Quantum_QubitType, Quantum_QuregType]>;

#endif // QUANTUM_DIALECT
