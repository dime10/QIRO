// Extract/Combine consolidations
qs.circ @consolidate(%r0 : !qs.rstate<2>) -> !qs.rstate<2> {

    %q0_0, %rem0 = qs.extract %r0[0] : !qs.rstate<2> -> !qs.qstate, !qs.rstate<1>
    %q0_1 = qs.H %q0_0 : !qs.qstate -> !qs.qstate
    %r1 = qs.scombine %rem0[0], %q0_1 : !qs.rstate<1>, !qs.qstate -> !qs.rstate<2>

    %q1_0, %rem1 = qs.extract %r1[1] : !qs.rstate<2> -> !qs.qstate, !qs.rstate<1>
    %q1_1 = qs.H %q1_0 : !qs.qstate -> !qs.qstate
    %r2 = qs.scombine %rem1[1], %q1_1 : !qs.rstate<1>, !qs.qstate -> !qs.rstate<2>

    %q0_2, %rem2 = qs.extract %r2[0] : !qs.rstate<2> -> !qs.qstate, !qs.rstate<1>
    %q0_3 = qs.H %q0_2 : !qs.qstate -> !qs.qstate
    %r3 = qs.scombine %rem2[0], %q0_3 : !qs.rstate<1>, !qs.qstate -> !qs.rstate<2>

    qs.return %r3 : !qs.rstate<2>
}
