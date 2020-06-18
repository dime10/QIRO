// test module for all quantum operations, commented ops are supposed to fail
module {
    // test register operations (extracting qubit, slicing register, combining to register)
    %0 = "q.alloc"() : () -> !q.qubit
    %1 =  q.alloc : !q.qubit

    %2 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
    %3 =  q.allocreg(4) : !q.qureg<4>

    %4 = "q.extract"(%2) {idx=2} : (!q.qureg<4>) -> !q.qubit
    %5 =  q.extract %2[2] : (!q.qureg<4>) -> !q.qubit
    //q.extract %2[4] : (!q.qureg<4>) -> !q.qubit

    %6 = "q.slice"(%2) {a=1, b=4} : (!q.qureg<4>) -> !q.qureg<3>
    %7 =  q.slice %2[1, 4] : (!q.qureg<4>) -> !q.qureg<3>
    //q.slice %2[1,2] : (!q.qureg<4>) -> !q.qureg<1>
    //q.slice %2[3,5] : (!q.qureg<4>) -> !q.qureg<2>
    //q.slice %2[1,3] : (!q.qureg<4>) -> !q.qureg<1>

    %8 = "q.genreg"(%0, %2) : (!q.qubit, !q.qureg<4>) -> !q.qureg<5>
    %9 =  q.genreg %0, %2 : (!q.qubit, !q.qureg<4>) -> !q.qureg<5>
    %10 = "q.genreg"(%0, %2, %1) : (!q.qubit, !q.qureg<4>, !q.qubit) -> !q.qureg<6>
    %11 =  q.genreg %0, %2, %1 : (!q.qubit, !q.qureg<4>, !q.qubit) -> !q.qureg<6>
    //q.genreg %0, %2, %1 : (!q.qubit, !q.qureg<4>, !q.qubit) -> !q.qureg<8>

    // test basic gates, including their custom assembly formats
    "q.H"(%0) : (!q.qubit) -> ()
    q.H %0 : (!q.qubit) -> ()
    "q.H"(%2) : (!q.qureg<4>) -> ()
    q.H %2 : (!q.qureg<4>) -> ()
    %op0 = "q.H"() : () -> !q.op
    %op1 = q.H : () -> !q.op
    //"q.H"(%0) : (!q.qubit) -> !q.op
    //"q.H"() : () -> ()

    "q.X"(%0) : (!q.qubit) -> ()
    q.X %0 : (!q.qubit) -> ()
    "q.X"(%2) : (!q.qureg<4>) -> ()
    q.X %2 : (!q.qureg<4>) -> ()
    %op2 = "q.X"() : () -> !q.op
    %op3 = q.X : () -> !q.op

    "q.RZ"(%0) {phi=0.1} : (!q.qubit) -> ()
    q.RZ(0.1) %0 : (!q.qubit) -> ()
    "q.RZ"(%2) {phi=0.1} : (!q.qureg<4>) -> ()
    q.RZ(0.1) %2 : (!q.qureg<4>) -> ()
    %op4 = "q.RZ"() {phi=0.1} : () -> !q.op
    %op5 = q.RZ(0.1) : () -> !q.op

    "q.CX"(%1, %0) : (!q.qubit, !q.qubit) -> ()
    q.CX %1, %0 : (!q.qubit, !q.qubit) -> ()
    "q.CX"(%2, %0) : (!q.qureg<4>, !q.qubit) -> ()
    q.CX %2, %0 : (!q.qureg<4>, !q.qubit) -> ()
    %op6 = "q.CX"(%0) : (!q.qubit) -> !q.cop<1>
    %op7 = q.CX %0 : (!q.qubit) -> !q.cop<1>

    // create a small test circuit
    %circ0 = "q.circ"() ({
        q.H %0 : (!q.qubit) -> ()
        q.CX %1, %0 : (!q.qubit, !q.qubit) -> ()
        q.term
    }) {name="test"} : () -> !q.circ

    %circ1 = q.circ("test") {
        q.H %0 : (!q.qubit) -> ()
        q.CX %1, %0 : (!q.qubit, !q.qubit) -> ()
    } : !q.circ

    %circ2 = q.circ {
        q.H %0 : (!q.qubit) -> ()
        q.CX %1, %0 : (!q.qubit, !q.qubit) -> ()
    } : !q.circ

    // test a parametrized circuit
    func @fun(%dummy : index, %qbs : !q.qureg<4>) {
        q.H %qbs : (!q.qureg<4>) -> ()
        q.term
    }
    %param = constant 4 : index

    %pc0 = "q.parcirc"(%param, %2) {callee=@fun} : (index, !q.qureg<4>) -> !q.circ
    %pc1 = q.parcirc @fun(%param, %2) : (index, !q.qureg<4>) -> !q.circ

    // test control meta operation, including on: ops, cops, and circs//, test variadic input
    "q.c"(%op0, %1, %0) : (!q.op, !q.qubit, !q.qubit) -> ()
    q.c %op0, %1, %0 : (!q.op, !q.qubit, !q.qubit) -> ()
    "q.c"(%op0, %1, %2) : (!q.op, !q.qubit, !q.qureg<4>) -> ()
    q.c %op0, %1, %2 : (!q.op, !q.qubit, !q.qureg<4>) -> ()
    "q.c"(%op0, %2, %0) : (!q.op, !q.qureg<4>, !q.qubit) -> ()
    q.c %op0, %2, %0 : (!q.op, !q.qureg<4>, !q.qubit) -> ()
    "q.c"(%op6, %1, %0) : (!q.cop<1>, !q.qubit, !q.qubit) -> ()
    q.c %op6, %1, %0 : (!q.cop<1>, !q.qubit, !q.qubit) -> ()
    //q.c %circ0, %1, %0 : (!q.circ, !q.qubit, !q.qubit) -> () // can't apply circ by passing qubits

    %cop0 = "q.c"(%op0, %0) : (!q.op, !q.qubit) -> !q.cop<1>
    %cop1 = q.c %op0, %0 : (!q.op, !q.qubit) -> !q.cop<1>
    %ccop0 = "q.c"(%cop0, %2) : (!q.cop<1>, !q.qureg<4>) -> !q.cop<5>
    %ccop1 =  q.c %cop0, %2 : (!q.cop<1>, !q.qureg<4>) -> !q.cop<5>
    %ccirc0 = "q.c"(%circ0, %0) : (!q.circ, !q.qubit) -> !q.cop<1>
    %ccirc1 =  q.c %circ0, %0 : (!q.circ, !q.qubit) -> !q.cop<1>
    //q.c %op0, %2 : (!q.op, !q.qureg<4>) -> !q.cop<5>
    //q.c %ccop0, %2 : (!q.cop<5>, !q.qureg<4>) -> !q.cop<3>

    // test adjoint meta operation
    "q.adj"(%op0, %0) : (!q.op, !q.qubit) -> ()
    q.adj %op0, %0 : (!q.op, !q.qubit) -> ()
    "q.adj"(%cop0, %0) : (!q.cop<1>, !q.qubit) -> ()
    q.adj %cop0, %0 : (!q.cop<1>, !q.qubit) -> ()
    %aop0 = "q.adj"(%op0) : (!q.op) -> !q.op
    %aop1 = q.adj %op0 : (!q.op) -> !q.op
    %acirc0 = "q.adj"(%circ0) : (!q.circ) -> !q.circ
    %acirc1 = q.adj %circ0 : (!q.circ) -> !q.circ
    %acop0 = "q.adj"(%cop0) : (!q.cop<1>) -> !q.cop<1>
    %acop1 = q.adj %cop0 : (!q.cop<1>) -> !q.cop<1>
    //q.adj %op0 : (!q.op) -> !q.cop<1>             // input and output must be the same type
    //q.adj %circ0, %0 : (!q.circ0, !q.qubit) -> () // circuits cannot be applied by specifying qubits

    // test circuit application
    "q.apply"(%circ0) : (!q.circ) -> ()
    q.apply %circ0 : (!q.circ) -> ()
    "q.apply"(%ccirc0) : (!q.cop<1>) -> ()
    q.apply %ccirc0 : (!q.cop<1>) -> ()
    "q.apply"(%acirc0) : (!q.circ) -> ()
    q.apply %acirc0 : (!q.circ) -> ()
}
