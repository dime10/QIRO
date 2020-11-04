//===------------------------------------------------------------------------------------------===//
// Test module for the conversion pass between the Quantum and QuantumSSA dialect
//===------------------------------------------------------------------------------------------===//

%s = constant 8 : index
%fp = constant 1.0 : f64

// test register operations (extracting qubit, slicing register, combining to register)
%0 = "q.alloc"() : () -> !q.qubit
%1 = q.alloc -> !q.qubit
%99 = q.alloc -> !q.qubit

%2 = q.allocreg(4) -> !q.qureg<4>
%3 = q.allocreg(%s) -> !q.qureg<>

// test basic gates, including their custom assembly formats
q.H %0 : !q.qubit
q.H %2 : !q.qureg<4>
%op0 = q.H -> !q.u1
q.H %2[0] : !q.qureg<4>

q.X %0 : !q.qubit
q.X %2 : !q.qureg<4>
%op2 = q.X -> !q.u1
q.X %2[%s] : !q.qureg<4>

q.RZ(0.1) %0 : !q.qubit
q.RZ(%fp: f64) %2 : !q.qureg<4>
%op4 = q.RZ(0.1) -> !q.u1
q.RZ(%fp: f64) %3[0] : !q.qureg<>

q.R(%fp: f64) %0 : !q.qubit
q.R(0.1) %2 : !q.qureg<4>
%op5 = q.R(%fp: f64) -> !q.u1
q.R(0.1) %3[%s] : !q.qureg<>

q.CX %0, %1 : !q.qubit, !q.qubit
q.CX %0, %2 : !q.qubit, !q.qureg<4>
%op6 = q.CX -> !q.u2
q.CX %0, %2[0] : !q.qubit, !q.qureg<4>
q.CX %2[%s], %3[0] : !q.qureg<4>, !q.qureg<>

// create a small test circuit
q.circ @circ0(%q: !q.qubit, %p: !q.qubit) {
    q.H %q : !q.qubit
    q.CX %q, %p : !q.qubit, !q.qubit
}
q.call @circ0(%0, %1) : !q.qubit, !q.qubit

// test a parametric circuit
q.circ @newfun(%q : !q.qubit, %q2 : !q.qubit, %r : !q.qureg<>, %n : index, %extra : !q.qubit) {
    %c1 = constant 1 : index
    %cond = cmpi "eq", %n, %c1 : index

    affine.for %i = 0 to %n {
        q.H %q : !q.qubit
        q.CX %q, %r[%i] : !q.qubit, !q.qureg<>

        scf.if %cond {
            %cx = q.CX -> !q.u2
            q.ctrl %cx, %r, %q, %q2 : !q.u2, !q.qureg<>, !q.qubit, !q.qubit
            q.H %r[0] : !q.qureg<>
        }
    }

    scf.if %cond {
        %cx = q.CX -> !q.u2
        q.ctrl %cx, %r, %q, %q2 : !q.u2, !q.qureg<>, !q.qubit, !q.qubit
    } else {
        q.SWAP %q, %q2 : !q.qubit, !q.qubit
    }
}
%n = constant 5 : index
%circ = q.getval @newfun -> !q.circ
q.apply %circ(%0, %1, %2, 5, %99) : !q.circ(!q.qubit, !q.qubit, !q.qureg<4>, !q.qubit)
q.apply %circ(%0, %1, %2, %n, %99) : !q.circ(!q.qubit, !q.qubit, !q.qureg<4>, index, !q.qubit)

// test control meta operation, including on: ops, cops, and circs
q.ctrl %op0, %0, %1 : !q.u1, !q.qubit, !q.qubit
q.ctrl %op0, %0, %2 : !q.u1, !q.qubit, !q.qureg<4>
q.ctrl %op0, %2, %1 : !q.u1, !q.qureg<4>, !q.qubit
q.ctrl %op6, %0, %1, %2 : !q.u2, !q.qubit, !q.qubit, !q.qureg<4>
%cop0 = q.ctrl %op0, %0 : !q.u1, !q.qubit -> !q.cop<1, !q.u1>
%cop2 = q.ctrl %op6, %0 : !q.u2, !q.qubit -> !q.cop<1, !q.u2>
%cop4 =  q.ctrl %cop0, %2 : !q.cop<1, !q.u1>, !q.qureg<4> -> !q.cop<5, !q.u1>
%cop6 =  q.ctrl %circ, %0 : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
q.apply %cop6(%0, %1, %2, 5, %99) : !q.cop<1, !q.circ>(!q.qubit, !q.qubit, !q.qureg<4>, !q.qubit)

// test adjoint meta operation
q.adj %op0, %0 : !q.u1, !q.qubit
q.adj %op0, %2 : !q.u1, !q.qureg<4>
q.adj %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit
%aop0 = q.adj %op0 : !q.u1 -> !q.u1
%aop2 = q.adj %circ : !q.circ -> !q.circ
%aop4 = q.adj %cop0 : !q.cop<1, !q.u1> -> !q.cop<1, !q.u1>
q.apply %aop2(%0, %1, %2, %n, %99) : !q.circ(!q.qubit, !q.qubit, !q.qureg<4>, index, !q.qubit)

// test global affine loop
affine.for %i = 0 to 4 {
    q.H %0 : !q.qubit
    q.CX %0, %2[%i] : !q.qubit, !q.qureg<4>
}

// single register multiple access
q.SWAP %2[0], %2[1] : !q.qureg<4>, !q.qureg<4>

// measurement
q.meas %0 : !q.qubit -> i1
q.meas %2 : !q.qureg<4> -> memref<4xi1>
q.meas %2[0] : !q.qureg<4> -> i1

q.freereg %3 : !q.qureg<>
q.freereg %2 : !q.qureg<4>
q.free %99 : !q.qubit
q.free %1 : !q.qubit
q.free %0 : !q.qubit
