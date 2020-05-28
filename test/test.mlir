module {
    // test basic operations
    %0 = "q.alloc"() : () -> (!q.qubit)
    %1 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
    "q.H"(%0) : (!q.qubit) -> ()
    "q.H"(%1) : (!q.qureg<4>) -> ()
    "q.H"(%1, %0) : (!q.qureg<4>, !q.qubit) -> ()
    "q.H"(%0, %1) : (!q.qubit, !q.qureg<4>) -> ()
}
