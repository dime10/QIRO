// test module for all quantum operations, commented ops are supposed to fail
module {
    // test register operations (extracting qubit, slicing register, combining to register)
    %0 = "q.alloc"() : () -> !q.qubit
    %1 =  q.alloc -> !q.qubit

    %2 = "q.allocreg"() {size=4} : () -> !q.qureg<4>
    %3 =  q.allocreg(4) -> !q.qureg<4>

    %4 = "q.extract"(%2) {idxAttr=2 : index} : (!q.qureg<4>) -> !q.qubit
    %5 =  q.extract %2[[2]] : !q.qureg<4> -> !q.qubit
    //q.extract %2[[4]] : !q.qureg<4> -> !q.qubit // index out of bounds
    func @testextract(%l : !q.qlist, %r : !q.qureg<4>, %c : index) {
        %20 = "q.extract"(%l) {idxAttr=0 : index} : (!q.qlist) -> !q.qubit
        %21 = q.extract %l[[0]] : !q.qlist -> !q.qubit
        %22 = "q.extract"(%l, %c) : (!q.qlist, index) -> !q.qubit
        %23 = q.extract %l[%c] : !q.qlist, index -> !q.qubit
        //q.extract %r[] : !q.qureg<4> -> !q.qubit              // no index supplied
        //q.extract %r[[4] %c] : !q.qureg<4>, index -> !q.qubit // index supplied twice
        //q.extract %r[%c] : !q.qureg<4>, index -> !q.qubit     // arg index supplied with qureg
        q.term
    }

    %6 = "q.slice"(%2) {a=1, b=4} : (!q.qureg<4>) -> !q.qureg<3>
    %7 =  q.slice %2[1, 4] : !q.qureg<4> -> !q.qureg<3>
    //q.slice %2[1,2] : !q.qureg<4> -> !q.qureg<2> // range too small (min 2)
    //q.slice %2[3,5] : !q.qureg<4> -> !q.qureg<2> // range out of bounds
    //q.slice %2[1,3] : !q.qureg<4> -> !q.qureg<3> // range - type mismatch

    %8 = "q.genreg"(%0, %2) : (!q.qubit, !q.qureg<4>) -> !q.qureg<5>
    %9 =  q.genreg %0, %2 : !q.qubit, !q.qureg<4> -> !q.qureg<5>
    %10 = "q.genreg"(%0, %2, %1) : (!q.qubit, !q.qureg<4>, !q.qubit) -> !q.qureg<6>
    %11 =  q.genreg %0, %2, %1 : !q.qubit, !q.qureg<4>, !q.qubit -> !q.qureg<6>
    //q.genreg %0, %2, %1 : !q.qubit, !q.qureg<4>, !q.qubit -> !q.qureg<8> // qubit num mismatch
    func @testgenreg(%l : !q.qlist, %r : !q.qureg<4>) {
        %20 = "q.genreg"(%l) : (!q.qlist) -> !q.qureg<7>
        %21 = q.genreg %l : !q.qlist -> !q.qureg<7>
        //q.genreg %l, %r : !q.qlist, !q.qureg<4> -> !q.qureg<7> // qlist: only 1 operand allowed
        q.term
    }

    // test basic gates, including their custom assembly formats
    "q.H"(%0) : (!q.qubit) -> ()
    q.H %0 : !q.qubit
    "q.H"(%2) : (!q.qureg<4>) -> ()
    q.H %2 : !q.qureg<4>
    %op0 = "q.H"() : () -> !q.op
    %op1 = q.H -> !q.op
    //q.H %0 : !q.qubit -> !q.op // cannot pass target and get op result
    //q.H                        // must pass either target or op result

    "q.X"(%0) : (!q.qubit) -> ()
    q.X %0 : !q.qubit
    "q.X"(%2) : (!q.qureg<4>) -> ()
    q.X %2 : !q.qureg<4>
    %op2 = "q.X"() : () -> !q.op
    %op3 = q.X -> !q.op

    "q.RZ"(%0) {phi=0.1} : (!q.qubit) -> ()
    q.RZ(0.1) %0 : !q.qubit
    "q.RZ"(%2) {phi=0.1} : (!q.qureg<4>) -> ()
    q.RZ(0.1) %2 : !q.qureg<4>
    %op4 = "q.RZ"() {phi=0.1} : () -> !q.op
    %op5 = q.RZ(0.1) -> !q.op

    "q.CX"(%1, %0) : (!q.qubit, !q.qubit) -> ()
    q.CX %1, %0 : !q.qubit, !q.qubit
    "q.CX"(%2, %0) : (!q.qureg<4>, !q.qubit) -> ()
    q.CX %2, %0 : !q.qureg<4>, !q.qubit
    %op6 = "q.CX"(%0) : (!q.qubit) -> !q.cop<1>
    %op7 = q.CX %0 : !q.qubit -> !q.cop<1>

    // create a small test circuit
    %circ0 = "q.circ"() ({
        q.H %0 : !q.qubit
        q.CX %1, %0 : !q.qubit, !q.qubit
        q.term
    }) {name="test"} : () -> !q.circ

    %circ1 = q.circ("test") {
        q.H %0 : !q.qubit
        q.CX %1, %0 : !q.qubit, !q.qubit
    } -> !q.circ

    %circ2 = q.circ {
        q.H %0 : !q.qubit
        q.CX %1, %0 : !q.qubit, !q.qubit
    } -> !q.circ

    // test a parametrized circuit
    func @fun(%dummy : index, %qbs : !q.qlist) {
        %reg = q.genreg %qbs : !q.qlist -> !q.qureg<4>
        q.H %reg : !q.qureg<4>
        q.term
    }
    func @fun2(%dummy : index, %qb : !q.qubit, %qbs : !q.qlist) {
        %reg = q.genreg %qbs : !q.qlist -> !q.qureg<4>
        q.RZ(0.1) %qb : !q.qubit
        q.H %reg : !q.qureg<4>
        q.term
    }
    %pc0 = "q.parcirc"(%0) {callee=@fun, n=1} : (!q.qubit) -> !q.circ
    %pc1 = q.parcirc @fun(1, %0) : (!q.qubit) -> !q.circ
    %pc2 = "q.parcirc"(%2) {callee=@fun, n=4} : (!q.qureg<4>) -> !q.circ
    %pc3 = q.parcirc @fun(4, %2) : (!q.qureg<4>) -> !q.circ
    %pc4 = "q.parcirc"(%0, %2) {callee=@fun2, n=5} : (!q.qubit, !q.qureg<4>) -> !q.circ
    %pc5 = q.parcirc @fun2(5, %0, %2) : (!q.qubit, !q.qureg<4>) -> !q.circ
    //q.parcirc @fun3(4, %2) : (!q.qureg<4>) -> !q.circ               // function doesn't exist
    //q.parcirc @fun(4, %0, %2) : (!q.qubit, !q.qureg<4>) -> !q.circ  // too many arguments to func
    //q.parcirc @fun2(2, %2, %0) : (!q.qureg<4>, !q.qubit) -> !q.circ // arguments don't match sig
    //q.parcirc @fun(5, %2) : (!q.qureg<4>) -> !q.circ                // less than 'n' qubits given

    // test control meta operation, including on: ops, cops, and circs
    "q.c"(%op0, %1, %0) : (!q.op, !q.qubit, !q.qubit) -> ()
    q.c %op0, %1, %0 : !q.op, !q.qubit, !q.qubit
    "q.c"(%op0, %1, %2) : (!q.op, !q.qubit, !q.qureg<4>) -> ()
    q.c %op0, %1, %2 : !q.op, !q.qubit, !q.qureg<4>
    "q.c"(%op0, %2, %0) : (!q.op, !q.qureg<4>, !q.qubit) -> ()
    q.c %op0, %2, %0 : !q.op, !q.qureg<4>, !q.qubit
    "q.c"(%op6, %1, %0) : (!q.cop<1>, !q.qubit, !q.qubit) -> ()
    q.c %op6, %1, %0 : !q.cop<1>, !q.qubit, !q.qubit
    //q.c %circ0, %1, %0 : !q.circ, !q.qubit, !q.qubit // can't apply circ by passing qubits

    %cop0 = "q.c"(%op0, %0) : (!q.op, !q.qubit) -> !q.cop<1>
    %cop1 = q.c %op0, %0 : !q.op, !q.qubit -> !q.cop<1>
    %ccop0 = "q.c"(%cop0, %2) : (!q.cop<1>, !q.qureg<4>) -> !q.cop<5>
    %ccop1 =  q.c %cop0, %2 : !q.cop<1>, !q.qureg<4> -> !q.cop<5>
    %ccirc0 = "q.c"(%circ0, %0) : (!q.circ, !q.qubit) -> !q.cop<1>
    %ccirc1 =  q.c %circ0, %0 : !q.circ, !q.qubit -> !q.cop<1>
    //q.c %op0, %2 : !q.op, !q.qureg<4> -> !q.cop<5>       // num controls mismatch
    //q.c %ccop0, %2 : !q.cop<5>, !q.qureg<4> -> !q.cop<3> // num combined controls mismatch

    // test adjoint meta operation
    "q.adj"(%op0, %0) : (!q.op, !q.qubit) -> ()
    q.adj %op0, %0 : !q.op, !q.qubit
    "q.adj"(%cop0, %0) : (!q.cop<1>, !q.qubit) -> ()
    q.adj %cop0, %0 : !q.cop<1>, !q.qubit
    %aop0 = "q.adj"(%op0) : (!q.op) -> !q.op
    %aop1 = q.adj %op0 : !q.op -> !q.op
    %acirc0 = "q.adj"(%circ0) : (!q.circ) -> !q.circ
    %acirc1 = q.adj %circ0 : !q.circ -> !q.circ
    %acop0 = "q.adj"(%cop0) : (!q.cop<1>) -> !q.cop<1>
    %acop1 = q.adj %cop0 : !q.cop<1> -> !q.cop<1>
    //q.adj %op0 : !q.op -> !q.cop<1>      // input and output must be the same type
    //q.adj %circ0, %0 : !q.circ, !q.qubit // circ can't be applied by specifying qubits

    // test circuit application
    "q.apply"(%circ0) : (!q.circ) -> ()
    q.apply %circ0 : !q.circ
    "q.apply"(%ccirc0) : (!q.cop<1>) -> ()
    q.apply %ccirc0 : !q.cop<1>
    "q.apply"(%acirc0) : (!q.circ) -> ()
    q.apply %acirc0 : !q.circ
}
