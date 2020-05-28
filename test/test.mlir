module {
    // test basic operations
    %0 = "q.alloc"() : () -> (!q.qubit)
    %1 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
    "q.H"(%0) : (!q.qubit) -> ()
    "q.H"(%1) : (!q.qureg<4>) -> ()
    "q.H"(%1, %0) : (!q.qureg<4>, !q.qubit) -> ()
    "q.H"(%0, %1) : (!q.qubit, !q.qureg<4>) -> ()
    // test register operations (extracting qubit, pretty format, combining to register)
    %2 = "q.extract"(%1) {idx=2} : (!q.qureg<4>) -> (!q.qubit)
    %3 = q.extract %1[2] : !q.qureg<4> -> !q.qubit
    %4 = "q.slice"(%1) {a=1, b=4} : (!q.qureg<4>) -> (!q.qureg<3>)
    %5 = q.slice %1[1,3] : !q.qureg<4> -> !q.qureg<2>
    %6 = "q.genreg"(%0, %1) : (!q.qubit, !q.qureg<4>) -> (!q.qureg<5>)

    // create a small test circuit without any arguments
    "q.circ"() ({
        ^bb0:
            %qb = "q.alloc"() : () -> (!q.qubit)
            "q.H"(%qb) : (!q.qubit) -> ()
            "q.bar"() : () -> ()
    }) : () -> ()

    // create a small testcircuit, this time with control qubits
    "q.circ"(%1) ({
        ^bb0:
            %qbs = "q.allocreg"() {size=3} : () -> (!q.qureg<3>)
            "q.H"(%qbs) : (!q.qureg<3>) -> ()
            "q.bar"() : () -> ()
    }) : (!q.qureg<4>) -> ()

    // create a small testcircuit, which is repeated 6 times
    "q.circ"(%1) ({
        ^bb0:
            %qbs = "q.allocreg"() {size=3} : () -> (!q.qureg<3>)
            "q.H"(%qbs) : (!q.qureg<3>) -> ()
            "q.bar"() : () -> ()
    }) {repeat=6} : (!q.qureg<4>) -> ()
}
