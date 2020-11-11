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
    %5 =  q.allocreg(%size) -> !q.qureg<>

    // test deallocation ops
    "q.free"(%0) : (!q.qubit) -> ()
    q.free %1 : !q.qubit

    "q.freereg"(%2) : (!q.qureg<4>) -> ()
    q.freereg %3 : !q.qureg<4>


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
    q.RZ(%fp: f64) %0 : !q.qubit
    %op4 = "q.RZ"() {static_phi = 0.1, operand_segment_sizes = dense<0> : vector<3xi32>, static_range = []} : () -> !q.u1
    %op5 = q.RZ(0.1) -> !q.u1
    %op6 = "q.RZ"(%fp) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (f64) -> !q.u1
    %op7 = q.RZ(%fp: f64) -> !q.u1
    //q.RZ() %0 : !q.qubit                           // need to provide angle
    //q.RZ(%fp: f64) %0 {static_phi=0.1} : !q.qubit  // cannot provide angle twice

    q.RZ(0.1) %2[2] : !q.qureg<4>
    q.RZ(%fp: f64) %2[2] : !q.qureg<4>
    q.RZ(0.1) %2[%a,%b] : !q.qureg<4>
    q.RZ(%fp: f64) %2[%a,%b] : !q.qureg<4>
    q.RZ(0.1) %2[1,%b,2] : !q.qureg<4>
    q.RZ(%fp: f64) %2[1,%b,2] : !q.qureg<4>


    "q.R"(%0) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qubit) -> ()
    q.R(0.1) %0 : !q.qubit
    "q.R"(%2) {static_phi = 0.1, operand_segment_sizes = dense<[0, 1, 0]> : vector<3xi32>, static_range = []} : (!q.qureg<4>) -> ()
    q.R(0.1) %2 : !q.qureg<4>
    "q.R"(%fp, %0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>, static_range = []} : (f64, !q.qubit) -> ()
    q.R(%fp: f64) %0 : !q.qubit
    %op8 = "q.R"() {static_phi = 0.1, operand_segment_sizes = dense<0> : vector<3xi32>, static_range = []} : () -> !q.u1
    %op9 = q.R(0.1) -> !q.u1
    %op10 = "q.R"(%fp) {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, static_range = []} : (f64) -> !q.u1
    %op11 = q.R(%fp: f64) -> !q.u1

    q.R(0.1) %2[2] : !q.qureg<4>
    q.R(%fp: f64) %2[2] : !q.qureg<4>
    q.R(0.1) %2[%a,%b] : !q.qureg<4>
    q.R(%fp: f64) %2[%a,%b] : !q.qureg<4>
    q.R(0.1) %2[1,%b,2] : !q.qureg<4>
    q.R(%fp: f64) %2[1,%b,2] : !q.qureg<4>


    "q.CX"(%0, %1) {operand_segment_sizes = dense<[1, 0, 1, 0]> : vector<4xi32>, static_crange = [], static_qrange = []} : (!q.qubit, !q.qubit) -> ()
    q.CX %0, %1 : !q.qubit, !q.qubit
    "q.CX"(%0, %2) {operand_segment_sizes = dense<[1, 0, 1, 0]> : vector<4xi32>, static_crange = [], static_qrange = []} : (!q.qubit, !q.qureg<4>) -> ()
    q.CX %0, %2 : !q.qubit, !q.qureg<4>
    %op12 = "q.CX"() {operand_segment_sizes = dense<0> : vector<4xi32>, static_crange = [], static_qrange = []} : () -> !q.u2
    %op13 = q.CX -> !q.u2

    q.CX %0, %2[2] : !q.qubit, !q.qureg<4>
    q.CX %0, %2[%a,%b] : !q.qubit, !q.qureg<4>
    q.CX %0, %2[1,%b,2] : !q.qubit, !q.qureg<4>
    q.CX %2[0], %2[1,3] : !q.qureg<4>, !q.qureg<4>
    //q.CX %0, %0[2] : !q.qubit, !q.qubit  // illegal on anything but qureg
    //q.CX %0, %0[%s] : !q.qubit, !q.qubit // illegal on anything but qureg
    //q.CX %0[2], %0 : !q.qubit, !q.qubit  // illegal on anything but qureg
    //q.CX %0[%s], %0 : !q.qubit, !q.qubit // illegal on anything but qureg


    "q.SWAP"(%0, %1) {operand_segment_sizes = dense<[1, 0, 1, 0]> : vector<4xi32>, static_range = [], static_range2 = []} : (!q.qubit, !q.qubit) -> ()
    q.SWAP %0, %1 : !q.qubit, !q.qubit
    %op14 = "q.SWAP"() {operand_segment_sizes = dense<0> : vector<4xi32>, static_range = [], static_range2 = []} : () -> !q.u2
    %op15 = q.SWAP -> !q.u2

    q.SWAP %0, %2[2] : !q.qubit, !q.qureg<4>
    q.SWAP %2[%b], %0 : !q.qureg<4>, !q.qubit
    q.SWAP %2[0], %2[%a] : !q.qureg<4>, !q.qureg<4>
    //q.SWAP %0, %2 : !q.qubit, !q.qureg<4>
    //q.SWAP %2[%a,%b], %0 : !q.qureg<4>, !q.qubit

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
        %c1 = constant 1 : index
        q.H %r[0] : !q.qureg<>
        scf.for %i = %c1 to %n step %c1 {
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
    "q.ctrl"(%op0, %0, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u1, !q.qubit, !q.qubit) -> ()
    q.ctrl %op0, %0, %1 : !q.u1, !q.qubit, !q.qubit
    "q.ctrl"(%op0, %0, %2) {operand_segment_sizes = dense<[1, 1, 0, 1, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u1, !q.qubit, !q.qureg<4>) -> ()
    q.ctrl %op0, %0, %2 : !q.u1, !q.qubit, !q.qureg<4>
    "q.ctrl"(%op0, %2, %1) {operand_segment_sizes = dense<[1, 1, 0, 1, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u1, !q.qureg<4>, !q.qubit) -> ()
    q.ctrl %op0, %2, %1 : !q.u1, !q.qureg<4>, !q.qubit
    "q.ctrl"(%op12, %0, %1, %2) {operand_segment_sizes = dense<[1, 1, 0, 1, 0, 1, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u2, !q.qubit, !q.qubit, !q.qureg<4>) -> ()
    q.ctrl %op12, %0, %1, %2 : !q.u2, !q.qubit, !q.qubit, !q.qureg<4>
    %cop0 = "q.ctrl"(%op0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u1, !q.qubit) -> !q.cop<1, !q.u1>
    %cop1 = q.ctrl %op0, %0 : !q.u1, !q.qubit -> !q.cop<1, !q.u1>
    %cop2 = "q.ctrl"(%op12, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u2, !q.qubit) -> !q.cop<1, !q.u2>
    %cop3 = q.ctrl %op12, %0 : !q.u2, !q.qubit -> !q.cop<1, !q.u2>
    %ccop0 = "q.ctrl"(%cop0, %2) {operand_segment_sizes = dense<[1, 1, 0, 0, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.cop<1, !q.u1>, !q.qureg<4>) -> !q.cop<5, !q.u1>
    %ccop1 =  q.ctrl %cop0, %2 : !q.cop<1, !q.u1>, !q.qureg<4> -> !q.cop<5, !q.u1>
    %ccirc0 = "q.ctrl"(%circ0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0, 0, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.circ, !q.qubit) -> !q.cop<1, !q.circ>
    %ccirc1 =  q.ctrl %circ0, %0 : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
    //q.ctrl %circ0, %0, %1 : !q.circ, !q.qubit, !q.qubit   // can't apply circ by passing qubits
    //q.ctrl %op0, %2, %2 : !q.u1, !q.qureg<4>, !q.qureg<4> // can't use same qubits for ctrl and trgt
    //q.ctrl %op0, %2 : !q.u1, !q.qureg<4> -> !q.cop<5>               // # controls mismatch
    //q.ctrl %ccop0, %2 : !q.cop<5, !q.u1>, !q.qureg<4> -> !q.cop<3>  // # total controls mismatch
    //q.ctrl %op0, %0 : !q.u1, !q.qubit -> !q.cop<1, !q.circ>         // wrong base type
    //q.ctrl %op0, %0 : !q.u1, !q.qubit -> !q.u2                      // must provide base type
    //q.ctrl %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit -> !q.cop<1, !q.cop<1, !q.u1>> // can't use cop as base type
    //q.ctrl %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit -> !q.cop<2, !q.circ>          // must preserve base type
    //"q.ctrl"(%op12, %0, %2) {operand_segment_sizes = dense<[1, 1, 0, 0, 0, 1, 0]> : vector<7xi32>, static_crange = [], static_range = [], static_range2 = []} : (!q.u2, !q.qubit, !q.qureg<4>) -> ()
    //q.ctrl %op0, %0, %1, %2 : !q.u1, !q.qubit, !q.qubit, !q.qureg<4> // can't second argument for u1
    //q.ctrl %op12, %0, %1 : !q.u2, !q.qubit, !q.qubit // need second argument for u2

    q.ctrl %op0, %0, %2[2] : !q.u1, !q.qubit, !q.qureg<4>
    q.ctrl %op0, %0, %2[%a,%b] : !q.u1, !q.qubit, !q.qureg<4>
    q.ctrl %op0, %0, %2[1,%b,2] : !q.u1, !q.qubit, !q.qureg<4>
    q.ctrl %op0, %2[0], %2[1,3] : !q.u1, !q.qureg<4>, !q.qureg<4>
    //q.ctrl %op0, %0[1], %2[1,%b,2] : !q.u1, !q.qubit, !q.qureg<4>  // illegal on anything but qureg
    //q.ctrl %op0, %2[1,%b,2], %0[%s] : !q.u1, !q.qureg<4>, !q.qubit // illegal on anything but qureg


    // test adjoint meta operation
    "q.adj"(%op0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_range = [], static_range2 = []} : (!q.u1, !q.qubit) -> ()
    q.adj %op0, %0 : !q.u1, !q.qubit
    "q.adj"(%cop0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0, 0]> : vector<5xi32>, static_range = [], static_range2 = []} : (!q.cop<1, !q.u1>, !q.qubit) -> ()
    q.adj %cop0, %0 : !q.cop<1, !q.u1>, !q.qubit
    %aop0 = "q.adj"(%op0) {operand_segment_sizes = dense<[1, 0, 0, 0, 0]> : vector<5xi32>, static_range = [], static_range2 = []} : (!q.u1) -> !q.u1
    %aop1 = q.adj %op0 : !q.u1 -> !q.u1
    %acirc0 = "q.adj"(%circ0) {operand_segment_sizes = dense<[1, 0, 0, 0, 0]> : vector<5xi32>, static_range = [], static_range2 = []} : (!q.circ) -> !q.circ
    %acirc1 = q.adj %circ0 : !q.circ -> !q.circ
    %acop0 = "q.adj"(%cop0) {operand_segment_sizes = dense<[1, 0, 0, 0, 0]> : vector<5xi32>, static_range = [], static_range2 = []} : (!q.cop<1, !q.u1>) -> !q.cop<1, !q.u1>
    %acop1 = q.adj %cop0 : !q.cop<1, !q.u1> -> !q.cop<1, !q.u1>
    //q.adj %op0 : !q.u1 -> !q.cop<1, !q.u1> // input and output must be the same type
    //q.adj %circ0, %0 : !q.circ, !q.qubit   // circ can't be applied by specifying qubits
    //"q.adj"(%op12, %2) {operand_segment_sizes = dense<[1, 0, 0, 1, 0]> : vector<5xi32>, static_range = [], static_range2 = []} : (!q.u2, !q.qureg<4>) -> ()
    //q.adj %op0, %1, %2 : !q.u1, !q.qubit, !q.qureg<4> // can't second argument for u1
    //q.adj %op12, %1 : !q.u2, !q.qubit // need second argument for u2

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

    %m0 = "q.meas"(%0) {static_range=[]} : (!q.qubit) -> i1
    %m1 =  q.meas %0 : !q.qubit -> i1
    %m2 = "q.meas"(%2) {static_range=[]} : (!q.qureg<4>) -> memref<4xi1>
    %m3 =  q.meas %2 : !q.qureg<4> -> memref<4xi1>

    %m4 = q.meas %2[2] : !q.qureg<4> -> i1
    %m5 = q.meas %2[%a,%b] : !q.qureg<4> -> memref<?xi1>
    %m6 = q.meas %2[1,%b,2] : !q.qureg<4> -> memref<?xi1>

    //q.meas %0 : !q.qubit -> memref<1xi1>       // need i1
    //q.meas %2[2] : !q.qureg<4> -> memref<1xi1> // need i1
    //q.meas %2 : !q.qureg<4> -> i1              // need memref
    //q.meas %2 : !q.qureg<4> -> memref<?xi1>    // static dim size
    //q.meas %2 : !q.qureg<4> -> memref<3xi1>    // wrong dim size
    //q.meas %2 : !q.qureg<4> -> memref<2x2xi1>  // only 1 dimension allowed
}
