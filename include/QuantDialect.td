#ifndef QUANT_DIALECT
#define QUANT_DIALECT

include "mlir/IR/OpBase.td"

// Define the basic properties of our dialect
def Quant_Dialect : Dialect {
    let name = "quant";
    let summary = "An MLIR dialect for quantum circuit compilation.";
    let description = [{
        This dialect is an example of a dialect designed to allow for 
        the compilation and optimization of quantum programs.
    }];
    let cppNamespace = "quant";
}

// Base class for all operations in this dialect
class Quant_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Quant_Dialect, mnemonic, traits>;

// Provide definitions for the custom Quant types to ODS. Syntax:
//   def <TypeName> : Type<<predicate identifying type>, <description>>;
//   `$_self` will be replaced with the entity this predicate is attached to.
def Quant_QubitType :
    Type<CPred<"$_self.isa<QubitType>()">, "Quant qubit type">;
def Quant_QuregType :
    Type<CPred<"$_self.isa<QuregType>()">, "Quant register type">;

// Provide a definition of the types that are used within the Toy dialect.
def Quant_Type : AnyTypeOf<[Quant_QubitType, Quant_QuregType]>;

#endif // QUANT_DIALECT
