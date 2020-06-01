module {
    // test basic operations
    %0 = "q.alloc"() : () -> (!q.qubit)
    %1 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
    %op0 = "q.H"(%0) : (!q.qubit) -> (!q.op)
    %op1 = "q.H"(%1) : (!q.qureg<4>) -> (!q.op)

    // test register operations (extracting qubit, pretty format, combining to register)
    %2 = "q.extract"(%1) {idx=2} : (!q.qureg<4>) -> (!q.qubit)
    %3 = q.extract %1[2] : !q.qureg<4> -> !q.qubit
    %4 = "q.slice"(%1) {a=1, b=4} : (!q.qureg<4>) -> (!q.qureg<3>)
    %5 = q.slice %1[1,3] : !q.qureg<4> -> !q.qureg<2>
    %6 = "q.genreg"(%0, %1) : (!q.qubit, !q.qureg<4>) -> (!q.qureg<5>)

    // create a small test circuit without any arguments
    %c1 = "q.circ"() ({
        ^bb0:
            %qb = "q.alloc"() : () -> (!q.qubit)
            %op = "q.H"(%qb) : (!q.qubit) -> (!q.op)
            "q.bar"() : () -> ()
    }) : () -> (!q.circ)

    // create a small testcircuit, which is repeated 6 times
    %qbs = "q.allocreg"() {size=3} : () -> (!q.qureg<3>)
    %c2 = "q.circ"() ({
        ^bb0:
            %op = "q.H"(%qbs) : (!q.qureg<3>) -> (!q.op)
            "q.bar"() : () -> ()
    }) {repeat=6} : () -> (!q.circ)

    // test control meta operation
    %qb = "q.alloc"() : () -> (!q.qubit)
    %h = "q.H"(%qb) : (!q.qubit) -> (!q.op)
    %ch0 = "q.c"(%h, %0) : (!q.op, !q.qubit) -> (!q.cop<1>)
    %ch1 = "q.c"(%h, %1) : (!q.op, !q.qureg<4>) -> (!q.cop<4>)
    %ch2 = "q.c"(%h, %0, %1) : (!q.op, !q.qubit, !q.qureg<4>) -> (!q.cop<5>)
    %ch3 = "q.c"(%h, %1, %0) : (!q.op, !q.qureg<4>, !q.qubit) -> (!q.cop<5>)
    %cch = "q.c"(%ch0, %1) : (!q.cop<1>, !q.qureg<4>) -> (!q.cop<5>)
    %cc = "q.c"(%c1, %0) : (!q.circ, !q.qubit) -> (!q.cop<1>)
}
