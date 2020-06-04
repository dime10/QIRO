## Document high-level design decisions here.
(The chosen design option is highlighted.)

&nbsp;

## Handling of quantum meta operations
### Option 1: Attributes
Every operation takes optional attributes that determine whether and how a meta operation is applied to the current op. For example:
- Control: optional control qubits argument
- Hermitian conjugate (inverse): boolean parameter
### `Option 2: First-class operations`
Every operation returns an SSA value of type 'Op' that represents the operation being applied. Thus, an operation becomes a first-class value that can be passed/returned to/from other "meta" operations. Example:
- Control: meta op taking ***Op*** as argument and returning ***C-Op***
- Hermitian conjugate: also takes some ***Op*** and returns the inverse ***Op†***
### Reasoning:
First-class operations allow for more flexibility in defining meta operations by allowing it to be represented as a full operation with custom semantics, verification, assembly forms, etc., compared to their representation as simple arguments/attributes in every op. Maintaining and adding new meta operations is significantly simplified by collecting the relevant code in a single place and not requiring the modification of existing ops for the definition of new meta operations.

## Semantics of quantum gate operations
### Option 1: Value semantics
Qubits are not explicitely represented in the IR, however their quantum *state* at time ***i*** is.
Thus, every operation acts on a qubit state represented by an SSA value, and similarly returns an
SSA value representing the *modified* quantum state at time ***i***+1.\
To recover qubit lines and the gates applied to them, one needs to trace the use-def chains of the corresponding SSA values. The graph created this way is acyclical by construction and thus represents the circuit in DAG form.\
Gate effects are explicit in this model and don't involve side-effects.
### `Option 2: Memory semantics`
Qubits are explicitely represented in the IR by SSA values.
Thus, every operation acts on a *qubit*, and doesn't return anything.\
A DAG is (potentially?) recoverable by parsing the whole program using the lexographical order of operations.\
Gates act entirely via side-effects on the quantum state.
### Reasoning:
Using explicit qubits simplifies certain aspects such as ensuring no qubit *state* value is used more than once. However, it might complicate reasoning about the program when classical control flow is involved.
