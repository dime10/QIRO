//===------------------------------------------------------------------------------------------===//
// Test module for operations & types of the Quantum dialect, commented ops are supposed to fail
//===------------------------------------------------------------------------------------------===//

module {
    // constants for later use
    %a = constant 1 : index
    %b = constant 3 : index
    %s = constant 2 : index
    %fp = constant 1.0 : f64
    %size = constant 8 : index


    // test allocation ops
    %0 = "q.alloc"() : () -> !q.qubit
    %1 =  q.alloc -> !q.qubit

    %2 = "q.allocreg"() {static_size=4} : () -> !q.qureg<4>
    %3 =  q.allocreg(4) -> !q.qureg<4>
    %4 = "q.allocreg"(%size) : (index) -> !q.qureg<>
    %5 =  q.allocreg(%size) : index -> !q.qureg<>


    // test basic gates, including their custom assembly formats
    "q.H"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, static_range = []} : (!q.qubit) -> ()
    q.H %0 : !q.qubit
    "q.H"(%2) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, static_range = []} : (!q.qureg<4>) -> ()
    q.H %2 : !q.qureg<4>
    %op0 = "q.H"() {operand_segment_sizes = dense<0> : vector<2xi32>, static_range = []} : () -> !q.op
    %op1 = q.H -> !q.op
    //q.H %0 : !q.qubit -> !q.op // cannot pass target and get op result
    //q.H                        // must pass either target or op result
    q.H %2[2] : !q.qureg<4>
    q.H %2[1, 3] : !q.qureg<4>
    q.H %2[0, 2, 2] : !q.qureg<4>
    q.H %2[%a] : !q.qureg<4>
    q.H %2[%a, %b] : !q.qureg<4>
    q.H %2[%a, %b, %s] : !q.qureg<4>
    q.H %2[%a,  3, %s] : !q.qureg<4>
    q.H %2[1,  %b,  2] : !q.qureg<4>
    //q.H %0[2] : !q.qubit    // illegal on anything but qureg
    //q.H %0[%s] : !q.qubit   // illegal on anything but qureg


    "q.X"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, static_range = []} : (!q.qubit) -> ()
    q.X %0 : !q.qubit
    "q.X"(%2) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, static_range = []} : (!q.qureg<4>) -> ()
    q.X %2 : !q.qureg<4>
    %op2 = "q.X"() {operand_segment_sizes = dense<0> : vector<2xi32>, static_range = []} : () -> !q.op
    %op3 = q.X -> !q.op

    q.X %2[2] : !q.qureg<4>
    q.X %2[%a,%b] : !q.qureg<4>
    q.X %2[1,%b,2] : !q.qureg<4>


    "q.RZ"(%0) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qubit) -> ()
    q.RZ(0.1) %0 : !q.qubit
    "q.RZ"(%2) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qureg<4>) -> ()
    q.RZ(0.1) %2 : !q.qureg<4>
    "q.RZ"(%fp, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (f64, !q.qubit) -> ()
    q.RZ(%fp) %0 : f64, !q.qubit
    %op4 = "q.RZ"() {static_phi = 0.1, operand_segment_sizes = dense<0> : vector<3xi32>, static_range = []} : () -> !q.op
    %op5 = q.RZ(0.1) -> !q.op
    %op6 = "q.RZ"(%fp) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (f64) -> !q.op
    %op7 = q.RZ(%fp) : f64 -> !q.op
    //q.RZ() %0 : !q.qubit                           // need to provide angle
    //q.RZ(%fp) %0 {static_phi=0.1} : f64, !q.qubit  // cannot provide angle twice

    q.RZ(0.1) %2[2] : !q.qureg<4>
    q.RZ(%fp) %2[2] : f64, !q.qureg<4>
    q.RZ(0.1) %2[%a,%b] : !q.qureg<4>
    q.RZ(%fp) %2[%a,%b] : f64, !q.qureg<4>
    q.RZ(0.1) %2[1,%b,2] : !q.qureg<4>
    q.RZ(%fp) %2[1,%b,2] : f64, !q.qureg<4>


    "q.R"(%0) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qubit) -> ()
    q.R(0.1) %0 : !q.qubit
    "q.R"(%2) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qureg<4>) -> ()
    q.R(0.1) %2 : !q.qureg<4>
    "q.R"(%fp, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (f64, !q.qubit) -> ()
    q.R(%fp) %0 : f64, !q.qubit
    %op8 = "q.R"() {static_phi = 0.1, operand_segment_sizes = dense<0> : vector<3xi32>, static_range = []} : () -> !q.op
    %op9 = q.R(0.1) -> !q.op
    %op10 = "q.R"(%fp) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (f64) -> !q.op
    %op11 = q.R(%fp) : f64 -> !q.op

    q.R(0.1) %2[2] : !q.qureg<4>
    q.R(%fp) %2[2] : f64, !q.qureg<4>
    q.R(0.1) %2[%a,%b] : !q.qureg<4>
    q.R(%fp) %2[%a,%b] : f64, !q.qureg<4>
    q.R(0.1) %2[1,%b,2] : !q.qureg<4>
    q.R(%fp) %2[1,%b,2] : f64, !q.qureg<4>


    "q.CX"(%0, %1) {operand_segment_sizes = dense<[1, 0, 1, 0]> : vector<4xi32>, static_crange = [], static_qrange = []} : (!q.qubit, !q.qubit) -> ()
    q.CX %0, %1 : !q.qubit, !q.qubit
    "q.CX"(%0, %2) {operand_segment_sizes = dense<[1, 0, 1, 0]> : vector<4xi32>, static_crange = [], static_qrange = []} : (!q.qubit, !q.qureg<4>) -> ()
    q.CX %0, %2 : !q.qubit, !q.qureg<4>
    %op12 = "q.CX"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, static_crange = [], static_qrange = []} : (!q.qubit) -> !q.cop<1>
    %op13 = q.CX %0 : !q.qubit -> !q.cop<1>
    //q.CX %0, %0 : !q.qubit, !q.qubit // cannot use same qubit as control and target

    q.CX %0, %2[2] : !q.qubit, !q.qureg<4>
    q.CX %0, %2[%a,%b] : !q.qubit, !q.qureg<4>
    q.CX %0, %2[1,%b,2] : !q.qubit, !q.qureg<4>
    q.CX %2[0], %2[1,3] : !q.qureg<4>, !q.qureg<4>
    //q.CX %0, %0[2] : !q.qubit, !q.qubit  // illegal on anything but qureg
    //q.CX %0, %0[%s] : !q.qubit, !q.qubit // illegal on anything but qureg
    //q.CX %0[2], %0 : !q.qubit, !q.qubit  // illegal on anything but qureg
    //q.CX %0[%s], %0 : !q.qubit, !q.qubit // illegal on anything but qureg


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


    // test register generation, commented lines trip the qubit/qureg invalidation verifiers
    %10 = q.alloc -> !q.qubit
    %11 = q.alloc -> !q.qubit
    %12 = q.allocreg(4) -> !q.qureg<4>
    %13 = q.allocreg(4) -> !q.qureg<4>
    %14 = "q.genreg"(%10, %12) : (!q.qubit, !q.qureg<4>) -> !q.qureg<5>
    %15 =  q.genreg %11, %13 : !q.qubit, !q.qureg<4> -> !q.qureg<5>
    //q.genreg %0, %2, %1 : !q.qubit, !q.qureg<4>, !q.qubit -> !q.qureg<8> // qubit num mismatch
    //q.genreg %1, %3 : !q.qubit, !q.qureg<4> -> !q.qureg<5>
    //q.genreg %10, %3 : !q.qubit, !q.qureg<4> -> !q.qureg<5>
    func @invalid(%q : !q.qubit, %r : !q.qureg<4>, %l1 : !q.qureg<>, %l2 : !q.qureg<>) -> !q.qureg<> {
        q.genreg %q, %r : !q.qubit, !q.qureg<4> -> !q.qureg<5>
        //q.X %q : !q.qubit
        //q.H %r : !q.qureg<4>
        //q.genlist %l1, %l2 : !q.qureg, !q.qureg -> !q.qureg
        return %l1 : !q.qureg<>
    }


    // test a parametrized circuit
    func @fun(%dummy : index, %qbs : !q.qureg<>) {
        q.H %qbs : !q.qureg<>
        q.term
    }
    func @fun2(%dummy : index, %qb : !q.qubit, %qbs : !q.qureg<>) {
        q.RZ(0.1) %qb : !q.qubit
        q.H %qbs : !q.qureg<>
        q.term
    }
    func @fun3(%dummy : index, %qbs0 : !q.qureg<>, %qb : !q.qubit, %qbs1 : !q.qureg<>) {
        q.term
    }

    %pc0 = "q.parcirc"(%2) {callee=@fun, n=4, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, static_ranges = []} : (!q.qureg<4>) -> !q.circ
    %pc1 = q.parcirc @fun(4, %2) : !q.qureg<4> -> !q.circ
    %pc2 = "q.parcirc"(%0, %2) {callee=@fun2, n=5, operand_segment_sizes = dense<[2, 0]> : vector<2xi32>, static_ranges = []} : (!q.qubit, !q.qureg<4>) -> !q.circ
    %pc3 = q.parcirc @fun2(5, %0, %2) : !q.qubit, !q.qureg<4> -> !q.circ
    //q.parcirc @fun3(4, %2) : !q.qureg<4> -> !q.circ               // function doesn't exist
    //q.parcirc @fun(4, %0, %2) : !q.qubit, !q.qureg<4> -> !q.circ  // too many arguments to func
    //q.parcirc @fun2(2, %2, %0) : !q.qureg<4>, !q.qubit -> !q.circ // arguments don't match sig
    //q.parcirc @fun(5, %2) : !q.qureg<4> -> !q.circ                // less than 'n' qubits given
    //func @fun4(%n : index) {q.term}
    //q.parcirc @fun3(0,) -> !q.circ                                // circuits needs >= 1 qubit
    //func @fun4(%q : !q.qubit, %p : !q.qubit) {q.term}
    //q.parcirc @fun4(1, %0) : !q.qubit -> !q.circ                  // function needs size param

    q.parcirc @fun(2, %2[2]) : !q.qureg<4> -> !q.circ
    q.parcirc @fun(2, %2[%a,%b]) : !q.qureg<4> -> !q.circ
    q.parcirc @fun(2, %2[1,%b,2]) : !q.qureg<4> -> !q.circ
    q.parcirc @fun3(2, %2[1,%b,2], %0, %2[2]) : !q.qureg<4>, !q.qubit, !q.qureg<4> -> !q.circ
    q.parcirc @fun3(2, %2, %0, %2) : !q.qureg<4>, !q.qubit, !q.qureg<4> -> !q.circ
    //q.parcirc @fun3(2, %2, %0[2], %2) : !q.qureg<4>, !q.qubit, !q.qureg<4> -> !q.circ  // illegal on anything but qureg
    //q.parcirc @fun3(2, %2, %0[%s], %2) : !q.qureg<4>, !q.qubit, !q.qureg<4> -> !q.circ //illegal on anything but qureg


    // test control meta operation, including on: ops, cops, and circs
    "q.c"(%op0, %0, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.op, !q.qubit, !q.qubit) -> ()
    q.c %op0, %0, %1 : !q.op, !q.qubit, !q.qubit
    "q.c"(%op0, %0, %2) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.op, !q.qubit, !q.qureg<4>) -> ()
    q.c %op0, %0, %2 : !q.op, !q.qubit, !q.qureg<4>
    "q.c"(%op0, %2, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.op, !q.qureg<4>, !q.qubit) -> ()
    q.c %op0, %2, %1 : !q.op, !q.qureg<4>, !q.qubit
    "q.c"(%op12, %0, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.cop<1>, !q.qubit, !q.qubit) -> ()
    q.c %op12, %0, %1 : !q.cop<1>, !q.qubit, !q.qubit
    %cop0 = "q.c"(%op0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.op, !q.qubit) -> !q.cop<1, !q.op>
    %cop1 = q.c %op0, %0 : !q.op, !q.qubit -> !q.cop<1, !q.op>
    %cop2 = "q.c"(%op12, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.cop<1>, !q.qubit) -> !q.cop<2>
    %cop3 = q.c %op12, %0 : !q.cop<1>, !q.qubit -> !q.cop<2>
    %ccop0 = "q.c"(%cop0, %2) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.cop<1, !q.op>, !q.qureg<4>) -> !q.cop<5, !q.op>
    %ccop1 =  q.c %cop0, %2 : !q.cop<1, !q.op>, !q.qureg<4> -> !q.cop<5, !q.op>
    %ccirc0 = "q.c"(%circ0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.circ, !q.qubit) -> !q.cop<1, !q.circ>
    %ccirc1 =  q.c %circ0, %0 : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
    //q.c %circ0, %0, %1 : !q.circ, !q.qubit, !q.qubit   // can't apply circ by passing qubits
    //q.c %op0, %2, %2 : !q.op, !q.qureg<4>, !q.qureg<4> // can't use same qubits for ctrl and trgt
    //q.c %op0, %2 : !q.op, !q.qureg<4> -> !q.cop<5>                   // # controls mismatch
    //q.c %ccop0, %2 : !q.cop<5, !q.op>, !q.qureg<4> -> !q.cop<3>      // # total controls mismatch
    //q.c %op0, %0 : !q.op, !q.qubit -> !q.cop<1, !q.circ>             // wrong base type
    //q.c %op0, %0 : !q.op, !q.qubit -> !q.cop<1>                      // must provide base type
    //q.c %op6, %0 : !q.cop<1>, !q.qubit -> !q.cop<2, !q.cop<1>>       // can't use cop as base type
    //q.c %cop0, %0 : !q.cop<1, !q.op>, !q.qubit -> !q.cop<2, !q.circ> // must preserve base type
    //q.c %cop0, %0 : !q.cop<1, !q.op>, !q.qubit -> !q.cop<2>          // must preserve base type

    q.c %op0, %0, %2[2] : !q.op, !q.qubit, !q.qureg<4>
    q.c %op0, %0, %2[%a,%b] : !q.op, !q.qubit, !q.qureg<4>
    q.c %op0, %0, %2[1,%b,2] : !q.op, !q.qubit, !q.qureg<4>
    q.c %op0, %2[0], %2[1,3] : !q.op, !q.qureg<4>, !q.qureg<4>
    //q.c %op0, %0[1], %2[1,%b,2] : !q.op, !q.qubit, !q.qureg<4>  // illegal on anything but qureg
    //q.c %op0, %2[1,%b,2], %0[%s] : !q.op, !q.qureg<4>, !q.qubit // illegal on anything but qureg


    // test adjoint meta operation
    "q.adj"(%op0, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (!q.op, !q.qubit) -> ()
    q.adj %op0, %0 : !q.op, !q.qubit
    "q.adj"(%cop0, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (!q.cop<1, !q.op>, !q.qubit) -> ()
    q.adj %cop0, %0 : !q.cop<1, !q.op>, !q.qubit
    %aop0 = "q.adj"(%op0) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (!q.op) -> !q.op
    %aop1 = q.adj %op0 : !q.op -> !q.op
    %acirc0 = "q.adj"(%circ0) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (!q.circ) -> !q.circ
    %acirc1 = q.adj %circ0 : !q.circ -> !q.circ
    %acop0 = "q.adj"(%cop0) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (!q.cop<1, !q.op>) -> !q.cop<1, !q.op>
    %acop1 = q.adj %cop0 : !q.cop<1, !q.op> -> !q.cop<1, !q.op>
    //q.adj %op0 : !q.op -> !q.cop<1>      // input and output must be the same type
    //q.adj %circ0, %0 : !q.circ, !q.qubit // circ can't be applied by specifying qubits

    q.adj %op0, %2[2] : !q.op, !q.qureg<4>
    q.adj %op0, %2[%a,%b] : !q.op, !q.qureg<4>
    q.adj %op0, %2[1,%b,2] : !q.op, !q.qureg<4>
    //q.adj %op0, %0[2] : !q.op, !q.qubit  // illegal on anything but qureg
    //q.adj %op0, %0[%s] : !q.op, !q.qubit // illegal on anything but qureg


    // test circuit application
    "q.apply"(%circ0) : (!q.circ) -> ()
    q.apply %circ0 : !q.circ
    "q.apply"(%ccirc0) : (!q.cop<1, !q.circ>) -> ()
    q.apply %ccirc0 : !q.cop<1, !q.circ>
    "q.apply"(%acirc0) : (!q.circ) -> ()
    q.apply %acirc0 : !q.circ
    //q.apply %op6 : !q.cop<1>         // can only be applied to circuit(-derived) types
    //q.apply %cop0 : !q.cop<1, !q.op> // can only be applied to circuit(-derived) types
}
