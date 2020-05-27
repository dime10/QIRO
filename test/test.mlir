module {
    func @bar() {
        %0 = "q.alloc"() : () -> (!q.qubit)
        %1 = "q.allocreg"() {size=4} : () -> (!q.qureg<4>)
        "q.H"(%0) : (!q.qubit) -> ()
        return
    }
}
