#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "QuantumDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Allow for an implicit terminator in blocks
def ImplicitQuantumTerminator : SingleBlockImplicitTerminator<"BarrierOp">;

def AllocOp : Quantum_Op<"alloc"> {
    let summary = "Allocate a qubit.";
    let description = [{
        The allocation command registers a new qubit with the system, which
        is guaranteed to be in the |0> state.

        This operation takes no inputs and returns a value of type 'qubit'.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        ```
    }];

    let results = (outs
        Qubit_Type : $qb
    );

    let assemblyFormat = [{
        attr-dict `:` type($qb)
    }];
}

def AllocRegOp : Quantum_Op<"allocreg"> {
    let summary = "Allocate a qubit register.";
    let description = [{
        This allocation command registers a new qubit register with the system,
        which is guaranteed to be in the |0..> state.

        This operation takes an integer attribute as input specifying the size,
        and returns a value of type 'qureg<size>'.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        ```
    }];

    let arguments = (ins
        Confined<APIntAttr, [IntMinValue<2>]> : $size
    );

    let results = (outs
        Qureg_Type : $reg
    );

    let assemblyFormat = [{
        `(`$size`)` attr-dict `:` type($reg)
    }];
}

def ExtractOp : Quantum_Op<"extract", [NoSideEffect]> {
    let summary = "Extract single qubit from register.";
    let description = [{
        This op allows to get out individual qubits from a register.
        The register is passed as an argument, and the index as an attribute.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        %1 = "q.extract"(%0) {idx=2} : (!q.qureg<4>) -> (!q.qubit)
        // OR in custom assembly format
        %2 = q.extract %0[2] : !q.qureg<4> -> !q.qubit
    }];

    let arguments = (ins
        Qureg_Type : $reg,
        Confined<APIntAttr, [IntNonNegative]> : $idx
    );

    let results = (outs
        Qubit_Type : $qb
    );

    let verifier = [{
        llvm::APInt idx = this->idx();
        unsigned regsize = this->reg().getType().cast<QuregType>().getNumQubits();

        if (idx.uge(regsize))
            return this->emitError() << "index out of bounds, got "
                                     << "idx=" << *(idx.getRawData())
                                     << ", regsize=" << regsize;

        return success();
    }];

    let assemblyFormat = [{
        $reg `[` $idx `]` attr-dict `:` functional-type(operands, results)
    }];
}

def SliceRegOp : Quantum_Op<"slice", [NoSideEffect]> {
    let summary = "Extract range of qubits from register.";
    let description = [{
        This op allows to extract a continous range of qubits from a register.
        The register is passed as an argument, and the indeces as attributes.
        The endpoint is included for the left index but NOT the right index.

        Example:

        ```mlir
        %0 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        %1 = "q.extract"(%0) {a=1, b=4} : (!q.qureg<4>) -> (!q.qureg<3>)
        // OR in custom assembly format
        %2 = q.extract %0[1,4] : !q.qureg<4> -> !q.qureg<3>
    }];

    let arguments = (ins
        Qureg_Type : $reg,
        Confined<APIntAttr, [IntNonNegative]> : $a,
        Confined<APIntAttr, [IntNonNegative]> : $b
    );

    let results = (outs
        Qureg_Type : $newreg
    );

    let verifier = [{
        llvm::APInt a = this->a();
        llvm::APInt b = this->b();
        unsigned regsize = this->reg().getType().cast<QuregType>().getNumQubits();
        unsigned newregsize = this->newreg().getType().cast<QuregType>().getNumQubits();

        if (a.uge(regsize) || b.ugt(regsize))
            return this->emitError() << "one or both indeces out of bounds, got "
                                     << "a=" << *(a.getRawData())
                                     << ", b=" << *(b.getRawData())
                                     << ", regsize=" << regsize;
        if ((b-a).slt(2))
            return this->emitError() << "slice must include at least 2 elements, got "
                                     << (long long) *((b-a).getRawData());
        if ((b-a) != newregsize)
            return this->emitError() << "output register has wrong size, got " << newregsize
                                     << ", requires " << *((b-a).getRawData());

        return success();
    }];

    let assemblyFormat = [{
        $reg `[` $a `,` $b `]` attr-dict `:` functional-type(operands, results)
    }];
}

def GenRegOp : Quantum_Op<"genreg", [NoSideEffect]> {
    let summary = "Generate a register from multiple qubits/quregs.";
    let description = [{
        This op allows to combine previously allocated qubits/quregs
        into a new register for the purpose of passing it to other operations.
        Note that there is no checks against qubit overlap, that is putting
        the same qubit into a register multiple times. This resposibility
        is left to the code generating the IR, and thus the op should be
        considered "internal".

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        %1 = "q.allocreg"() {size=2} : () -> (!q.qureg<2>)
        %2 = "q.genreg"(%0, %1) : (!q.qubit, !q.qureg<2>) -> (!q.qureg<3>)
        ```
    }];

    let arguments = (ins
        QData_Type : $input1,
        QData_Type : $input2,
        Variadic<QData_Type> : $extrainputs
    );

    let results = (outs
        Qureg_Type : $reg
    );

    let verifier = [{
        // obtain total number of qubits from input
        unsigned totalqbs = 0;

        auto qbtype = this->input1().getType();
        if (qbtype.isa<QubitType>()) {
            totalqbs++;
        }
        else if (qbtype.isa<QuregType>()) {
            totalqbs += qbtype.cast<QuregType>().getNumQubits();
        }

        qbtype = this->input2().getType();
        if (qbtype.isa<QubitType>()) {
            totalqbs++;
        }
        else if (qbtype.isa<QuregType>()) {
            totalqbs += qbtype.cast<QuregType>().getNumQubits();
        }

        auto xtypes = this->extrainputs().getTypes();
        for (auto xtype : xtypes) {
            if (xtype.isa<QubitType>()) {
                totalqbs++;
            }
            else if (xtype.isa<QuregType>()) {
                totalqbs += xtype.cast<QuregType>().getNumQubits();
            }
        }

        // assert that this matches the output type
        unsigned declsize = this->reg().getType().cast<QuregType>().getNumQubits();
        if (declsize != totalqbs) {
            return this->emitError() << "Number of qubits in the output doesn't match inputs!";
        }

        return success();
    }];

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];
}

def HOp : Quantum_Op<"H", [Unitary, Hermitian]> {
    let summary = "Hadamard gate.";
    let description = [{
        The Hadamard gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,  1],
             [1, -1]] / sqrt(2)

        This operation takes a qubit as input and returns an 'op' type.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        %1 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        // Apply a Hadamard gate to qubit %0
        $op1 = "q.H"(%0) : (!q.qubit) -> (!q.op)
        // Apply a Hadamard gate to each qubit in register %1
        $op2 = "q.H"(%1) : (!q.qureg<4>) -> (!q.op)
        ```
    }];

    let arguments = (ins
        QData_Type : $qbs
    );

    let results = (outs
        Op_Type : $op
    );

    let assemblyFormat = [{
        $qbs attr-dict `:` functional-type(operands, results)
    }];
}

def XOp : Quantum_Op<"X", [Unitary, Hermitian]> {
    let summary = "NOT gate.";
    let description = [{
        The NOT gate is a single Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 2x2 matrix representation in the computational basis is given by:
            [[0, 1],
             [1, 0]]

        This operation takes a qubit as input and returns an 'op' type.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        $op = "q.X"(%0) : (!q.qubit) -> (!q.op)
        ```
    }];

    let arguments = (ins
        QData_Type : $qbs
    );

    let results = (outs
        Op_Type : $op
    );

    let assemblyFormat = [{
        $qbs attr-dict `:` functional-type(operands, results)
    }];
}

def RzOp : Quantum_Op<"RZ", [Unitary]> {
    let summary = "Z-Rotation gate.";
    let description = [{
        The Rz gate is a single Qubit unitary.
        It rotates a qubit state by an angle φ along the z-axis.
        Its 2x2 matrix representation in the computational basis is given by:
            [[1,   0 ],
             [0, e^iφ]]

        This operation takes a qubit as input and returns an 'op' type.
        The parameter φ is passed as an attribute.
        It acts on the qubits via side-effect by changing the quantum state.

        Example:

        ```mlir
        $op = "q.RZ"(%0) {phi=0.1} : (!q.qubit) -> (!q.op)
        // OR in custom assembly format
        $op = q.RZ(0.1) %0 : (!q.qubit) -> !q.op
        ```
    }];

    let arguments = (ins
        QData_Type : $qbs,
        APFloatAttr : $phi
    );

    let results = (outs
        Op_Type : $op
    );

    let assemblyFormat = [{
        `(` $phi `)` $qbs attr-dict `:` functional-type(operands, results)
    }];
}

def CNotOp : Quantum_Op<"CX", [Unitary, Hermitian]> {
    let summary = "CNOT gate.";
    let description = [{
        The CNOT gate is a two Qubit (hermitian) unitary.
        Due to its hermitian property it is also its own inverse.
        Its 4x4 matrix representation in the computational basis is given by:
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]

        As with other basic operations, the target is allowed to be either a qubit or
        a register (in which case the gate is applied to every qubit in the register).
        The control however must be a singular qubit. The return type is a 'cop'.
        It acts on the qubits via side-effect by changing the quantum state.
        Note that the control qubit MUST NOT appear as one of target qubits (not checked atm).

        Example:

        ```mlir
        $op = "q.CX"(%0, %1) : (!q.qubit, !q.qubit) -> (!q.cop<1>)
        ```
    }];

    let arguments = (ins
        QData_Type : $trgts,
        Qubit_Type : $ctrl
    );

    let results = (outs
        ControlledOp_Type : $op
    );

    let assemblyFormat = [{
        $trgts `,` $ctrl attr-dict `:` functional-type(operands, results)
    }];
}

def BarrierOp : Quantum_Op<"bar", [Terminator]> {
    let summary = "Marks the end of a circuit (block).";
    let description = [{
        This is an internal operation solely used to mark the end
        of an mlir "block" which is used to hold all operations in
        the circuit. A "terminator" op is required for all blocks
        that do not transfer control flow.
    }];

    let assemblyFormat = [{
       attr-dict `:` `(` `)`
    }];
}

def CircuitOp : Quantum_Op<"circ", [Unitary, ImplicitQuantumTerminator]> {
    let summary = "Create a basic quantum circuit.";
    let description = [{
        A basic quantum circuit is a simple block regrouping multiple gates.
        No control flow is allowed within such a block. However, these operations
        can then be repeated, inverted or controlled as as single unit.

        Note that the gates are grouped within a block, which must be terminated
        with the special 'Barrier' operation.

        Example:

        ```mlir
        %qbs = "q.allocreg"() {size=3} : () -> (!q.qureg<3>)
        %c = "q.circ"() ({
        ^bb0:
            %op = "q.H"(%qbs) : (!q.qureg<3>) -> (!q.op)
            "q.bar"() : () -> ()
        }) : () -> (!q.circ)
        ```
    }];

    let results = (outs
        Circuit_Type : $circ
    );

    let regions = (region SizedRegion<1> : $gates);
}

def ControlOp : Quantum_Op<"c"> {
    let summary = "Control meta-op.";
    let description = [{
        This meta operation transforms any quantum op into an equivalent
        controlled operation. Thus, the op is only executed if the state
        of the control qubit (or register) is |1..>.

        Accepted inputs are an individual quantum operations, a quantum
        circuit, or an already controlled op. The result is a controlled op.

        Example:

        ```mlir
        %0 = "q.alloc"() : () -> (!q.qubit)
        %op = "q.H"(%0) : (!q.qubit) -> (!q.op)
        $cop = "q.c"(%op) : (!q.op) -> (!q.cop)
    }];

    let arguments = (ins
        QOp_Type : $op,
        QData_Type: $ctrls,
        Variadic<QData_Type> : $extractrls
    );

    let results = (outs
        ControlledOp_Type : $cop
    );

    let verifier = [{
        // obtain total number of control qubits from input
        unsigned totalctrls = 0;

        auto optype = this->op().getType();
        if (optype.isa<COpType>()) {
            totalctrls += optype.cast<COpType>().getNumCtrls();
        }

        auto qbtype = this->ctrls().getType();
        if (qbtype.isa<QubitType>()) {
            totalctrls++;
        }
        else if (qbtype.isa<QuregType>()) {
            totalctrls += qbtype.cast<QuregType>().getNumQubits();
        }

        auto xtypes = this->extractrls().getTypes();
        for (auto xtype : xtypes) {
            if (xtype.isa<QubitType>()) {
                totalctrls++;
            }
            else if (xtype.isa<QuregType>()) {
                totalctrls += xtype.cast<QuregType>().getNumQubits();
            }
        }

        // assert that this matches the output type
        unsigned declctrls = this->cop().getType().cast<COpType>().getNumCtrls();
        if (declctrls != totalctrls) {
            return this->emitError() << "Number of output controls doesn't match inputs!";
        }

        return success();
    }];

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];
}

def AdjointOp : Quantum_Op<"adj", [SameOperandsAndResultType]> {
    let summary = "(Hermitian) adjoint meta-op.";
    let description = [{
        This meta operation transforms any quantum op into its complex conjugate
        transpose version. Since all quantum gates are unitary, the resulting op
        is equal to the inverse operation.

        Accepted inputs are any quantum operations, such as a gate, circuit, or
        controlled op. The result is an op.

        Example:

        ```mlir
        %h = "q.H"(%0) : (!q.qubit) -> (!q.op)
        $hdg = "q.adj"(h) : (!q.op) -> (!q.op)
    }];

    let arguments = (ins
        QOp_Type : $op
    );

    let results = (outs
        QOp_Type : $opadj
    );

    let assemblyFormat = [{
        $op attr-dict `:` functional-type(operands, results)
    }];
}

#endif // QUANTUM_OPS
