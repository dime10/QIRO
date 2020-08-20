## Document high-level design decisions here.
(The chosen design option is highlighted.)\
**Work in progress, very rough!**

&nbsp;

## Handling of quantum meta operations
### Option 1: Attributes
Every operation takes optional attributes that determine whether and how a meta operation is applied to the current op. For example:
- Control: optional control qubits argument
- Hermitian conjugate (inverse): boolean parameter
### `Option 2: First-class operations`
Every operation can optionally return an SSA value of type 'Op' provided no target qubits argument was given. This represents the operation in a "hold" state that can be applied to qubits later on. Thus, an operation becomes a first-class value that can be passed/returned to/from other "meta" operations. Example:
- Control: meta op taking ***Op*** as argument and (optionally) returning ***C-Op***
- Hermitian conjugate: also takes some ***Op*** and (optionally) returns the inverse ***Op†***
Meta operations can directly be used for op application by specifying the target qubits as an argument.\
A special apply op is necessary for ops that cannot be applied via optional arguments (such as circuits).
### Reasoning:
First-class operations allow for more flexibility in defining meta operations by allowing it to be represented as a full operation with custom semantics, verification, assembly forms, etc., compared to their representation as simple arguments/attributes in every op. Maintaining and adding new meta operations is significantly simplified by collecting the relevant code in a single place and not requiring the modification of existing ops for the definition of new meta operations.

## Semantics of quantum gate operations
### Option 1: Value semantics
Qubits are not explicitly represented in the IR, however their quantum *state* at time ***i*** is.
Thus, every operation acts on a qubit state represented by an SSA value, and similarly returns an
SSA value representing the *modified* quantum state at time ***i***+1.\
To recover qubit lines and the gates applied to them, one needs to trace the use-def chains of the corresponding SSA values. The graph created this way is acyclical by construction and thus represents the circuit in DAG form.\
Gate effects are explicit in this model and don't involve side-effects.
### `Option 2: Memory semantics`
Qubits are explicitly represented in the IR by SSA values.
Thus, every operation acts on a *qubit*, and doesn't return anything.\
A DAG is (potentially?) recoverable by parsing the whole program using the lexographical order of operations.\
Gates act entirely via side-effects on the quantum state.
### Reasoning:
Using explicit qubits simplifies certain aspects such as ensuring no qubit *state* value is used more than once. However, it might complicate reasoning about the program when classical control flow is involved.

## Handling of user-defined operations
### `Option 1: Built-in functions`
Use the 'function' construct provided by MLIR (~ named op) to represent user-defined quantum operations, which can be made parametric by passing (arbitrary) arguments to the function. The operation is considered executed when an appropriate application mechanism is invoked (see the *Handling of quantum meta operations*).\
Functions have the limitation that all values referenced within must be passed to it. All arguments to ops within the body must (likely?) be made SSA values (no attributes). Functions do not directly support the extended "hold" concept since optional arguments and return types with custom verification are not supported, but they can be integrated into the first-class operations paradigm with the help of an additional custom op that specializes the function arguments and produces a concrete circuit to be manipulated later.
### Option 2: Op-supported definition & application
Placeholder
### Reasoning:
The circuit type & operation already provide support for named user-defined operations that act on a fixed set of qubits and can be manipulated (repeatedly applied, controlled, inverted) throughout the program. For user-defined operations that need to be applied to multiple (different) qubit sets or be made parametric (in the number of qubits e.g.), the function construct already provides most of what is needed to represent these. In particular, it provides block arguments that can be passed to the operation at a later point. This can (likely?) be done with custom operations as well, but outside the ODS framework. Thus, the currently simplest approach is to use this "hybrid" approach op-supported user operation construction (via circuits) and function-supported parametric user operation definition.
