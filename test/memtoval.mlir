// test module for all quantum operations, commented ops are supposed to fail
module {
// test register operations (extracting qubit, slicing register, combining to register)
    %0 = "q.alloc"() : () -> !q.qubit
    %1 = q.alloc -> !q.qubit
    %2 = q.alloc -> !q.qubit

    //%2 =  q.allocreg(4) -> !q.qureg<4>

    // test basic gates, including their custom assembly formats
    q.H %0 : !q.qubit
    %op0 = q.H -> !q.op

    q.X %0 : !q.qubit
    %op2 = q.X -> !q.op

    q.RZ(0.1) %0 : !q.qubit
    %op4 = q.RZ(0.1) -> !q.op

    q.CX %1, %0 : !q.qubit, !q.qubit
    //q.CX %2, %0 : !q.qureg<4>, !q.qubit
    %op6 = q.CX %0 : !q.qubit -> !q.cop<1>

    // create a small test circuit
    %circ0 = q.circ {
        q.H %0 : !q.qubit
        q.CX %1, %0 : !q.qubit, !q.qubit
    } -> !q.circ
    q.apply %circ0 : !q.circ

    func @newfun(%n : index, %q0 : !q.qubit, %q1 : !q.qubit, %q2 : !q.qubit) {
        q.H %q0 : !q.qubit
        q.CX %q1, %q0 : !q.qubit, !q.qubit
        %cx = q.CX %q0 : !q.qubit -> !q.cop<1>
        q.c %cx, %q0 : !q.cop<1>, !q.qubit -> !q.cop<2>
        q.term
    }
    %n = constant 5 : index
    call @newfun(%n, %0, %1, %2) : (index, !q.qubit, !q.qubit, !q.qubit) -> ()

    %circ1 = q.parcirc @newfun(3, %0, %1, %2) : (!q.qubit, !q.qubit, !q.qubit) -> !q.circ
    q.apply %circ1 : !q.circ

    // test control meta operation, including on: ops, cops, and circs
    q.c %op0, %1, %0 : !q.op, !q.qubit, !q.qubit
    q.c %op6, %1, %0 : !q.cop<1>, !q.qubit, !q.qubit
    %cop0 = q.c %op0, %0 : !q.op, !q.qubit -> !q.cop<1, !q.op>

    // test adjoint meta operation
    q.adj %op0, %0 : !q.op, !q.qubit
    q.adj %cop0, %0 : !q.cop<1, !q.op>, !q.qubit
    //%acirc0 = q.adj %circ0 : !q.circ -> !q.circ
}
