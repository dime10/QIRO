module {
    func @bar() {
        %0 = "q.allocate"() : () -> (!q.qubit)
        "q.H"(%0) : (!q.qubit) -> ()
        return
    }
}
