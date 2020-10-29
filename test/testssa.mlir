//===------------------------------------------------------------------------------------------===//
// Test module for the value semantics the QuantumSSA dialect
//===------------------------------------------------------------------------------------------===//

module {
    %fp = constant 1.0 : f64
    %size = constant 8 : index

    %q0_0 = qs.alloc -> !qs.qstate
    %q1_0 = qs.alloc -> !qs.qstate
    %r0_0 = qs.allocreg(4) -> !qs.rstate<4>
    %r1_0 = qs.allocreg(%size) -> !qs.rstate<>

    qs.free %q0_0 : !qs.qstate
    qs.freereg %r0_0 : !qs.rstate<4>

    %h = qs.H -> !qs.u1
    %q0_1 = qs.H %q0_0 : !qs.qstate -> !qs.qstate
    %q0_2 = qs.H %q0_0 : !qs.qstate -> !qs.qstate       // state reuse is illegal but not enforced
    %r0_1 = qs.H %r0_0 : !qs.rstate<4> -> !qs.rstate<4>
    //qs.H %q0_0 : !qs.qstate                           // must return new state
    //qs.H %q0_0 : !qs.qstate -> !qs.u1                 // must return state not op
    //%q0_3 = qs.H %r0_1 : !qs.rstate<4> -> !qs.qstate  // cannot mix qubit & register states

    %q0_3 = qs.X %q0_2 : !qs.qstate -> !qs.qstate
    %q0_4 = qs.RZ(%fp) %q0_3 : f64, !qs.qstate -> !qs.qstate
    %q0_5 = qs.R(%fp) %q0_4 : f64, !qs.qstate -> !qs.qstate
    %q1_1, %q0_6 = qs.CX %q1_0, %q0_5 : !qs.qstate, !qs.qstate -> !qs.qstate, !qs.qstate
    %q1_2, %r0_2 = qs.CX %q1_1, %r0_1 : !qs.qstate, !qs.rstate<4> -> !qs.qstate, !qs.rstate<4>
    %q1_3, %q0_7 = qs.SWAP %q1_2, %q0_6 : !qs.qstate, !qs.qstate -> !qs.qstate, !qs.qstate

    %q1_4, %ch = qs.ctrl %h, %q1_3 : !qs.u1, !qs.qstate -> !qs.qstate, !qs.cop<1, !qs.u1>
    %q1_5, %q0_10 = qs.ctrl %h, %q1_4, %q0_7 : !qs.u1, !qs.qstate, !qs.qstate -> !qs.qstate, !qs.qstate
    %r0_3, %q0_11 = qs.ctrl %h, %r0_2, %q0_10 : !qs.u1, !qs.rstate<4>, !qs.qstate -> !qs.rstate<4>, !qs.qstate
    %q1_6, %r0_4 = qs.ctrl %h, %q1_5, %r0_3 : !qs.u1, !qs.qstate, !qs.rstate<4> -> !qs.qstate, !qs.rstate<4>

    %ah = qs.adj %h : !qs.u1 -> !qs.u1
    %q0_12 = qs.adj %h, %q0_11 : !qs.u1, !qs.qstate -> !qs.qstate
    %r0_5 = qs.adj %h, %r0_4 : !qs.u1, !qs.rstate<4> -> !qs.rstate<4>

    // test new return statement
    qs.circ @retTest(%arg : !qs.qstate) -> !qs.qstate {
        qs.return %arg : !qs.qstate
    }

    // test new function circuit type
    %circ = qs.getval @retTest -> !qs.circ

    // apply meta op the function circuit
    %q1_7, %ccirc = qs.ctrl %circ, %q1_6 : !qs.circ, !qs.qstate -> !qs.qstate, !qs.cop<1, !qs.circ>
    %acirc = qs.adj %circ : !qs.circ -> !qs.circ

    // execute the function circuits
    %q0_20 = qs.call @retTest(%q0_12) : !qs.qstate -> !qs.qstate
    %q0_21 = qs.apply %circ(%q0_20) : !qs.circ(!qs.qstate -> !qs.qstate)
    %q0_22 = qs.apply %ccirc(%q0_21) : !qs.cop<1, !qs.circ>(!qs.qstate -> !qs.qstate)
    %q0_23 = qs.apply %acirc(%q0_22) : !qs.circ(!qs.qstate -> !qs.qstate)
    //%q0_24 = qs.apply %ch(%q0_23) : !qs.cop<1, !qs.u1>>(!qs.qstate) // only works on circ types

    // test measurements
    %q0_24, %m0 = qs.meas %q0_23 : !qs.qstate -> !qs.qstate, i1
    %r0_6, %m1 = qs.meas %r0_5 : !qs.rstate<4> -> !qs.rstate<4>, memref<4xi1>
    //qs.meas %q0_24 : !qs.qstate -> !qs.qstate, memref<1xi1>
    //qs.meas %r0_5 : !qs.rstate<4> -> !qs.rstate<4>, i1
}
