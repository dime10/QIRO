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
    %op0 = "q.H"() {operand_segment_sizes = dense<0> : vector<2xi32>, static_range = []} : () -> !q.u1
    %op1 = q.H -> !q.u1
    //q.H %0 : !q.qubit -> !q.u1 // cannot pass target and get op result
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
    %op2 = "q.X"() {operand_segment_sizes = dense<0> : vector<2xi32>, static_range = []} : () -> !q.u1
    %op3 = q.X -> !q.u1

    q.X %2[2] : !q.qureg<4>
    q.X %2[%a,%b] : !q.qureg<4>
    q.X %2[1,%b,2] : !q.qureg<4>


    "q.RZ"(%0) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qubit) -> ()
    q.RZ(0.1) %0 : !q.qubit
    "q.RZ"(%2) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qureg<4>) -> ()
    q.RZ(0.1) %2 : !q.qureg<4>
    "q.RZ"(%fp, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (f64, !q.qubit) -> ()
    q.RZ(%fp) %0 : f64, !q.qubit
    %op4 = "q.RZ"() {static_phi = 0.1, operand_segment_sizes = dense<0> : vector<3xi32>, static_range = []} : () -> !q.u1
    %op5 = q.RZ(0.1) -> !q.u1
    %op6 = "q.RZ"(%fp) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (f64) -> !q.u1
    %op7 = q.RZ(%fp) : f64 -> !q.u1
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
    %op8 = "q.R"() {static_phi = 0.1, operand_segment_sizes = dense<0> : vector<3xi32>, static_range = []} : () -> !q.u1
    %op9 = q.R(0.1) -> !q.u1
    %op10 = "q.R"(%fp) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (f64) -> !q.u1
    %op11 = q.R(%fp) : f64 -> !q.u1

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
    //q.CX %0, %0 : !q.qubit, !q.qubit // cannot use same qubit as control and target
    %op12 = "q.CX"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, static_crange = [], static_qrange = []} : (!q.qubit) -> !q.u2
    %op13 = q.CX %0 : !q.qubit -> !q.u2

    q.CX %0, %2[2] : !q.qubit, !q.qureg<4>
    q.CX %0, %2[%a,%b] : !q.qubit, !q.qureg<4>
    q.CX %0, %2[1,%b,2] : !q.qubit, !q.qureg<4>
    q.CX %2[0], %2[1,3] : !q.qureg<4>, !q.qureg<4>
    //q.CX %0, %0[2] : !q.qubit, !q.qubit  // illegal on anything but qureg
    //q.CX %0, %0[%s] : !q.qubit, !q.qubit // illegal on anything but qureg
    //q.CX %0[2], %0 : !q.qubit, !q.qubit  // illegal on anything but qureg
    //q.CX %0[%s], %0 : !q.qubit, !q.qubit // illegal on anything but qureg


    // create a small test circuit
    "q.circ"() ({
    ^bb0(%t0: !q.qubit, %t1: !q.qubit):
        q.H %t0 : !q.qubit
        q.CX %t1, %t0 : !q.qubit, !q.qubit
        q.term
    }) {sym_name = "test0", type = (!q.qubit, !q.qubit) -> ()} : () -> ()

    q.circ @test1(%t0: !q.qubit, %t1: !q.qubit) {
        q.H %t0 : !q.qubit
        q.CX %t1, %t0 : !q.qubit, !q.qubit
    }

    // test a parametrized circuit
    q.circ @entangle(%n: index, %r: !q.qureg<>) {
        q.H %r[0] : !q.qureg<>
        affine.for %i = 1 to %n {
            q.CX %r[0], %r[%i] : !q.qureg<>, !q.qureg<>
        }
    }

    //q.circ @fail0(%q : !q.qubit) -> !q.qubit {
    //    q.H %q : !q.qubit
    //}


    // create some circuit values
    %circ0 = "q.getval"() {circref = @test0} : () -> !q.circ
    %circ1 = q.getval @test1 -> !q.circ
    //q.getval @fail1 -> !q.circ // symbol does not exist
    //func @fail2(%i : index)
    //q.getval @fail2 -> !q.circ // symbol does not reference a circuit


    // test call to quantum circuit
    "q.call"(%0, %1) {circref = @test0, operand_segment_sizes = dense<[2, 0]> : vector<2xi32>, static_ranges = [[], []], size_params = [-1, -1]} : (!q.qubit, !q.qubit) -> ()
    q.call @test0(%0, %1) : !q.qubit, !q.qubit
    //q.call @fail1(%0, %1) : !q.qubit, !q.qubit                  // symbol does not exist
    //q.call @fail2(%a) : index                                   // symbol doesn't point to circuit
    //q.call @test0(%0, %1, %2) : !q.qubit, !q.qubit, !q.qureg<4> // too many arguments
    //q.call @test0(%0, %2) : !q.qubit, !q.qureg<4>               // arguments don't match signature
    //q.call @test0(%0, %1[0]) : !q.qubit, !q.qubit               // only allowed on qureg

    %c4 = constant 4 : index
    q.call @entangle(4, %2) : !q.qureg<4>
    q.call @entangle(%c4, %2) : index, !q.qureg<4>
    q.call @entangle(2, %2[0,3]) : !q.qureg<4>


    // test register generation, commented lines trip the qubit/qureg invalidation verifiers
    %10 = q.alloc -> !q.qubit
    %11 = q.alloc -> !q.qubit
    %12 = q.allocreg(4) -> !q.qureg<4>
    %13 = q.allocreg(4) -> !q.qureg<4>
    %14 = "q.genreg"(%10, %12) : (!q.qubit, !q.qureg<4>) -> !q.qureg<5>
    %15 =  q.genreg %11, %13 : !q.qubit, !q.qureg<4> -> !q.qureg<5>
    //q.genreg %0, %2, %1 : !q.qubit, !q.qureg<4>, !q.qubit -> !q.qureg<8> // qubit num mismatch
    //q.genreg %1, %3 : !q.qubit, !q.qureg<4> -> !q.qureg<5>   // %1 is used further below
    //q.genreg %10, %3 : !q.qubit, !q.qureg<4> -> !q.qureg<5>  // %10 was used up above


    // test control meta operation, including on: ops, cops, and circs
    "q.c"(%op0, %0, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.u1, !q.qubit, !q.qubit) -> ()
    q.c %op0, %0, %1 : !q.u1, !q.qubit, !q.qubit
    "q.c"(%op0, %0, %2) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.u1, !q.qubit, !q.qureg<4>) -> ()
    q.c %op0, %0, %2 : !q.u1, !q.qubit, !q.qureg<4>
    "q.c"(%op0, %2, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.u1, !q.qureg<4>, !q.qubit) -> ()
    q.c %op0, %2, %1 : !q.u1, !q.qureg<4>, !q.qubit
    "q.c"(%op12, %0, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.u2, !q.qubit, !q.qubit) -> ()
    q.c %op12, %0, %1 : !q.u2, !q.qubit, !q.qubit
    %cop0 = "q.c"(%op0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.u1, !q.qubit) -> !q.cop<1, !q.u1>
    %cop1 = q.c %op0, %0 : !q.u1, !q.qubit -> !q.cop<1, !q.u1>
    %cop2 = "q.c"(%op12, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.u2, !q.qubit) -> !q.cop<1, !q.u2>
    %cop3 = q.c %op12, %0 : !q.u2, !q.qubit -> !q.cop<1, !q.u2>
    %ccop0 = "q.c"(%cop0, %2) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.cop<1, !q.u1>, !q.qureg<4>) -> !q.cop<5, !q.u1>
    %ccop1 =  q.c %cop0, %2 : !q.cop<1, !q.u1>, !q.qureg<4> -> !q.cop<5, !q.u1>
    %ccirc0 = "q.c"(%circ0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_crange = [], static_qrange = []} : (!q.circ, !q.qubit) -> !q.cop<1, !q.circ>
    %ccirc1 =  q.c %circ0, %0 : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
    //q.c %circ0, %0, %1 : !q.circ, !q.qubit, !q.qubit   // can't apply circ by passing qubits
    //q.c %op0, %2, %2 : !q.u1, !q.qureg<4>, !q.qureg<4> // can't use same qubits for ctrl and trgt
    //q.c %op0, %2 : !q.u1, !q.qureg<4> -> !q.cop<5>                   // # controls mismatch
    //q.c %ccop0, %2 : !q.cop<5, !q.u1>, !q.qureg<4> -> !q.cop<3>      // # total controls mismatch
    //q.c %op0, %0 : !q.u1, !q.qubit -> !q.cop<1, !q.circ>             // wrong base type
    //q.c %op0, %0 : !q.u1, !q.qubit -> !q.cop<1>                      // must provide base type
    //q.c %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit -> !q.cop<1, !q.cop<1, !q.u1>> // can't use cop as base type
    //q.c %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit -> !q.cop<2, !q.circ> // must preserve base type
    //q.c %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit -> !q.cop<2>          // must preserve base type

    q.c %op0, %0, %2[2] : !q.u1, !q.qubit, !q.qureg<4>
    q.c %op0, %0, %2[%a,%b] : !q.u1, !q.qubit, !q.qureg<4>
    q.c %op0, %0, %2[1,%b,2] : !q.u1, !q.qubit, !q.qureg<4>
    q.c %op0, %2[0], %2[1,3] : !q.u1, !q.qureg<4>, !q.qureg<4>
    //q.c %op0, %0[1], %2[1,%b,2] : !q.u1, !q.qubit, !q.qureg<4>  // illegal on anything but qureg
    //q.c %op0, %2[1,%b,2], %0[%s] : !q.u1, !q.qureg<4>, !q.qubit // illegal on anything but qureg


    // test adjoint meta operation
    "q.adj"(%op0, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (!q.u1, !q.qubit) -> ()
    q.adj %op0, %0 : !q.u1, !q.qubit
    "q.adj"(%cop0, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (!q.cop<1, !q.u1>, !q.qubit) -> ()
    q.adj %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit
    %aop0 = "q.adj"(%op0) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (!q.u1) -> !q.u1
    %aop1 = q.adj %op0 : !q.u1 -> !q.u1
    %acirc0 = "q.adj"(%circ0) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (!q.circ) -> !q.circ
    %acirc1 = q.adj %circ0 : !q.circ -> !q.circ
    %acop0 = "q.adj"(%cop0) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (!q.cop<1, !q.u1>) -> !q.cop<1, !q.u1>
    %acop1 = q.adj %cop0 : !q.cop<1, !q.u1> -> !q.cop<1, !q.u1>
    //q.adj %op0 : !q.u1 -> !q.cop<1, !q.u1> // input and output must be the same type
    //q.adj %circ0, %0 : !q.circ, !q.qubit   // circ can't be applied by specifying qubits

    q.adj %op0, %2[2] : !q.u1, !q.qureg<4>
    q.adj %op0, %2[%a,%b] : !q.u1, !q.qureg<4>
    q.adj %op0, %2[1,%b,2] : !q.u1, !q.qureg<4>
    //q.adj %op0, %0[2] : !q.u1, !q.qubit  // illegal on anything but qureg
    //q.adj %op0, %0[%s] : !q.u1, !q.qubit // illegal on anything but qureg


    // test circuit application
    "q.apply"(%circ0, %0, %1) {operand_segment_sizes=dense<[1, 2, 0]> : vector<3xi32>, static_ranges = [[], []], size_params = [-1, -1]} : (!q.circ, !q.qubit, !q.qubit) -> ()
    q.apply %circ0(%0, %1) : !q.circ(!q.qubit, !q.qubit)
    "q.apply"(%ccirc0, %0, %1) {operand_segment_sizes=dense<[1, 2, 0]> : vector<3xi32>, static_ranges = [[], []], size_params = [-1, -1]} : (!q.cop<1, !q.circ>, !q.qubit, !q.qubit) -> ()
    q.apply %ccirc0(%0, %1) : !q.cop<1, !q.circ>(!q.qubit, !q.qubit)
    "q.apply"(%acirc0, %0, %1) {operand_segment_sizes=dense<[1, 2, 0]> : vector<3xi32>, static_ranges = [[], []], size_params = [-1, -1]} : (!q.circ, !q.qubit, !q.qubit) -> ()
    q.apply %acirc0(%0, %1) : !q.circ(!q.qubit, !q.qubit)
    //q.apply %op0 : !q.u1             // can only be applied to circuit(-derived) types
    //q.apply %cop0 : !q.cop<1, !q.u1> // can only be applied to circuit(-derived) types
}
