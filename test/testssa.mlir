// new module to test value semantics
module {
    %q0_0 = qs.alloc -> !qs.qstate
    %r0_0 = qs.allocreg(4) -> !qs.rstate<4>

    %h = qs.H -> !qs.op
    %q0_1 = qs.H %q0_0 : !qs.qstate -> !qs.qstate
    %q0_2 = qs.H %q0_0 : !qs.qstate -> !qs.qstate       // state reuse is illegal but not enforced
    %r0_1 = qs.H %r0_0 : !qs.rstate<4> -> !qs.rstate<4>
    //qs.H %q0_0 : !qs.qstate                           // must return new state
    //qs.H %q0_0 : !qs.qstate -> !qs.op                 // must return state not op
    //%q0_3 = qs.H %r0_1 : !qs.rstate<4> -> !qs.qstate  // cannot mix qubit & register states

    %q0_3 = qs.X %q0_2 : !qs.qstate -> !qs.qstate
    %q0_4 = qs.RZ(0.1) %q0_3 : !qs.qstate -> !qs.qstate
    %q0_5 = qs.CX %q0_4, %q0_0 : !qs.qstate, !qs.qstate -> !qs.qstate
    %r0_2 = qs.CX %r0_1, %q0_0 : !qs.rstate<4>, !qs.qstate -> !qs.rstate<4>
    //%q0_6 = qs.CX %q0_5, %q0_5 : !qs.qstate, !qs.qstate -> !qs.qstate // same trgt & ctrl

    %ch = qs.c %h, %q0_0 : !qs.op, !qs.qstate -> !qs.cop<1, !qs.op>
    %q0_6 = qs.c %h, %q0_5, %q0_0 : !qs.op, !qs.qstate, !qs.qstate -> !qs.qstate
    %q0_7 = qs.c %h, %q0_6, %r0_0 : !qs.op, !qs.qstate, !qs.rstate<4> -> !qs.qstate
    %r0_3 = qs.c %h, %r0_2, %q0_0 : !qs.op, !qs.rstate<4>, !qs.qstate -> !qs.rstate<4>

    %ah = qs.adj %h : !qs.op -> !qs.op
    %q0_8 = qs.adj %h, %q0_7 : !qs.op, !qs.qstate -> !qs.qstate
    %r0_4 = qs.adj %h, %r0_3 : !qs.op, !qs.rstate<4> -> !qs.rstate<4>

    // test new return statement
    func @retTest(%arg : !qs.qstate) -> !qs.qstate {
        qs.return %arg : !qs.qstate
    }

    // test new function circuit type
    %circ = qs.funcirc @retTest -> !qs.fcirc<(!qs.qstate) -> !qs.qstate>
    //qs.funcirc @retTest -> !qs.fcirc<(!qs.qstate) -> ()> // return & function types don't match

    // apply meta op the function circuit
    %ccirc = qs.c %circ, %q0_0 : !qs.fcirc<(!qs.qstate) -> !qs.qstate>, !qs.qstate
                               -> !qs.cop<1, !qs.fcirc<(!qs.qstate) -> !qs.qstate>>
    %acirc = qs.adj %circ : !qs.fcirc<(!qs.qstate) -> !qs.qstate>
                          -> !qs.fcirc<(!qs.qstate) -> !qs.qstate>

    // execute the function circuits
    %q0_9 = call @retTest(%q0_8) : (!qs.qstate) -> !qs.qstate
    %q0_10 = qs.applyfc %circ(%q0_9) : !qs.fcirc<(!qs.qstate) -> !qs.qstate>
    %q0_11 = qs.applyfc %ccirc(%q0_10) : !qs.cop<1, !qs.fcirc<(!qs.qstate) -> !qs.qstate>>
    %q0_12 = qs.applyfc %acirc(%q0_11) : !qs.fcirc<(!qs.qstate) -> !qs.qstate>
    //%q0_13 = qs.applyfc %ch(%q0_12) : !qs.cop<1, !qs.op>> // only works on fcirc(-derived) types
}
