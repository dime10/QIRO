// test module for all quantum operations, commented ops are supposed to fail
module {
    // test basic operations, including their custom assembly format
    %0 = "q.alloc"() : () -> !q.qubit
    %1 =  q.alloc : !q.qubit

    %2 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
    %3 =  q.allocreg(4) : !q.qureg<4>

    %op0 = "q.H"(%0) : (!q.qubit) -> !q.op
    %op1 =  q.H %0 : (!q.qubit) -> !q.op
    %op2 = "q.H"(%2) : (!q.qureg<4>) -> !q.op
    %op3 =  q.H %2 : (!q.qureg<4>) -> !q.op

    %op4 = "q.X"(%0) : (!q.qubit) -> !q.op
    %op5 =  q.X %0 : (!q.qubit) -> !q.op
    %op6 = "q.X"(%2) : (!q.qureg<4>) -> !q.op
    %op7 =  q.X %2 : (!q.qureg<4>) -> !q.op

    %op8 = "q.RZ"(%0) {phi=0.1} : (!q.qubit) -> !q.op
    %op9 =  q.RZ(0.1) %0 : (!q.qubit) -> !q.op
    %op10 = "q.RZ"(%2) {phi=0.1} : (!q.qureg<4>) -> !q.op
    %op11 =  q.RZ(0.1) %2 : (!q.qureg<4>) -> !q.op

    %op12 = "q.CX"(%0, %1) : (!q.qubit, !q.qubit) -> !q.cop<1>
    %op13 =  q.CX %0, %1 : (!q.qubit, !q.qubit) -> !q.cop<1>
    %op14 = "q.CX"(%2, %1) : (!q.qureg<4>, !q.qubit) -> !q.cop<1>
    %op15 =  q.CX %2, %1 : (!q.qureg<4>, !q.qubit) -> !q.cop<1>

    // test register operations (extracting qubit, slicing register, combining to register)
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

    // create a small test circuit
    %qb0 = q.alloc : !q.qubit
    %qb1 = q.alloc : !q.qubit

    %c0 = "q.circ"() ({
        q.H %qb0 : (!q.qubit) -> !q.op
        q.CX %qb1, %qb0 : (!q.qubit, !q.qubit) -> !q.cop<1>
        q.term
    }) : () -> !q.circ

    %c1 = q.circ {
        q.H %qb0 : (!q.qubit) -> !q.op
        q.CX %qb1, %qb0 : (!q.qubit, !q.qubit) -> !q.cop<1>
    } : !q.circ

    // test control meta operation, including on: ops, cops, and circs, test variadic input
    %h = q.H %0 : (!q.qubit) -> !q.op
    %ch0 = "q.c"(%h, %1) : (!q.op, !q.qubit) -> !q.cop<1>
    %ch1 =  q.c %h, %1 : (!q.op, !q.qubit) -> !q.cop<1>
    %ch2 = "q.c"(%h, %2) : (!q.op, !q.qureg<4>) -> !q.cop<4>
    %ch3 =  q.c %h, %2 : (!q.op, !q.qureg<4>) -> !q.cop<4>
    %ch4 = "q.c"(%h, %1, %2) : (!q.op, !q.qubit, !q.qureg<4>) -> !q.cop<5>
    %ch5 =  q.c %h, %1, %2 : (!q.op, !q.qubit, !q.qureg<4>) -> !q.cop<5>
    //%ch6 = q.c %h, %1, %2 : (!q.op, !q.qubit, !q.qureg<4>) -> !q.cop<9>
    %cch0 = "q.c"(%ch0, %2) : (!q.cop<1>, !q.qureg<4>) -> !q.cop<5>
    %cch1 =  q.c %ch0, %2 : (!q.cop<1>, !q.qureg<4>) -> !q.cop<5>
    %cc0 = "q.c"(%c0, %0) : (!q.circ, !q.qubit) -> !q.cop<1>
    %cc1 =  q.c %c0, %0 : (!q.circ, !q.qubit) -> !q.cop<1>

    // test adjoint meta operation
    %hdg0 = "q.adj"(%h) : (!q.op) -> !q.op
    %hdg1 =  q.adj %h : (!q.op) -> !q.op
    %hdg2 = "q.adj"(%ch0) : (!q.cop<1>) -> !q.cop<1>
    %hdg3 =  q.adj %ch0 : (!q.cop<1>) -> !q.cop<1>
    %hdg4 = "q.adj"(%c0) : (!q.circ) -> !q.circ
    %hdg5 =  q.adj %c0 : (!q.circ) -> !q.circ
    //%hdg6 = q.adj %h : (!q.op) -> !q.cop<1>
}
