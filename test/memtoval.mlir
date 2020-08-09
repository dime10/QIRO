// test module for all quantum operations, commented ops are supposed to fail
module {
// test register operations (extracting qubit, slicing register, combining to register)
    %0 = "q.alloc"() : () -> !q.qubit
    %1 = q.alloc -> !q.qubit

    %2 = q.allocreg(4) -> !q.qureg<4>

    // test basic gates, including their custom assembly formats
    q.H %0 : !q.qubit
    q.H %2 : !q.qureg<4>
    %op0 = q.H -> !q.op

    q.X %0 : !q.qubit
    q.X %2 : !q.qureg<4>
    %op2 = q.X -> !q.op

    q.RZ(0.1) %0 : !q.qubit
    q.RZ(0.1) %2 : !q.qureg<4>
    %op4 = q.RZ(0.1) -> !q.op

    q.CX %0, %1 : !q.qubit, !q.qubit
    q.CX %0, %2 : !q.qubit, !q.qureg<4>
    %op6 = q.CX %0 : !q.qubit -> !q.cop<1>

    // create a small test circuit
    %circ0 = q.circ {
        q.H %0 : !q.qubit
        q.CX %0, %1 : !q.qubit, !q.qubit
    } -> !q.circ
    q.apply %circ0 : !q.circ

    // test a circuit defined via a function
    func @newfun(%n : index, %q : !q.qubit, %r : !q.qureg<4>, %extra : !q.qubit) {
        q.H %q : !q.qubit
        q.CX %q, %r : !q.qubit, !q.qureg<4>
        %cx = q.CX %q : !q.qubit -> !q.cop<1>
        q.c %cx, %r : !q.cop<1>, !q.qureg<4> -> !q.cop<5>
        q.term
    }
    %n = constant 5 : index
    call @newfun(%n, %0, %2, %1) : (index, !q.qubit, !q.qureg<4>, !q.qubit) -> ()

    %circ = q.parcirc @newfun(3, %0, %2, %1) : (!q.qubit, !q.qureg<4>, !q.qubit) -> !q.circ
    q.apply %circ : !q.circ

    // test control meta operation, including on: ops, cops, and circs
    q.c %op0, %0, %1 : !q.op, !q.qubit, !q.qubit
    q.c %op0, %0, %2 : !q.op, !q.qubit, !q.qureg<4>
    q.c %op0, %2, %1 : !q.op, !q.qureg<4>, !q.qubit
    q.c %op6, %0, %1 : !q.cop<1>, !q.qubit, !q.qubit
    %cop0 = q.c %op0, %0 : !q.op, !q.qubit -> !q.cop<1, !q.op>
    %cop2 = q.c %op6, %0 : !q.cop<1>, !q.qubit -> !q.cop<2>
    %cop4 =  q.c %cop0, %2 : !q.cop<1, !q.op>, !q.qureg<4> -> !q.cop<5, !q.op>
    %cop6 =  q.c %circ, %0 : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
    q.apply %cop6 : !q.cop<1, !q.circ>

    // test adjoint meta operation
    q.adj %op0, %0 : !q.op, !q.qubit
    q.adj %op0, %2 : !q.op, !q.qureg<4>
    q.adj %cop0, %0 : !q.cop<1, !q.op>, !q.qubit
    %aop0 = q.adj %op0 : !q.op -> !q.op
    %aop2 = q.adj %circ : !q.circ -> !q.circ
    %aop4 = q.adj %cop0 : !q.cop<1, !q.op> -> !q.cop<1, !q.op>
    q.apply %aop2 : !q.circ
}
