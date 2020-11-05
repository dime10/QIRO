qs.circ @test(%x : i32, %r : !qs.rstate<>, %n : index, %q : !qs.qstate) -> (!qs.rstate<>, !qs.qstate) {
    qs.return %r, %q : !qs.rstate<>, !qs.qstate
}

%x = constant 5 : i32
%n = constant 8 : index
%r = qs.allocreg(%n) -> !qs.rstate<>
%q = qs.alloc -> !qs.qstate

%r2, %q2 = qs.call @test(%x, %r, %n, %q) : i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate

%test = qs.getval @test -> !qs.circ
%testAdj = qs.adj %test : !qs.circ -> !qs.circ
%r3, %q3 = qs.apply %testAdj(%x, %r2, %n, %q2) : !qs.circ(i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate)

%r4, %q4 = qs.apply %testAdj(%x, %r3, %n, %q3) : !qs.circ(i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate)
%r5, %q5 = qs.call @test(%x, %r4, %n, %q4) : i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate

// cancel with constant folding
%y = constant 5 : i32
%r6, %q6 = qs.call @test(%y, %r5, %n, %q5) : i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate
%r7, %q7 = qs.apply %testAdj(%x, %r6, %n, %q6) : !qs.circ(i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate)

// don't cancel
%z = constant 6 : i32
%r8, %q8 = qs.call @test(%z, %r7, %n, %q7) : i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate
%r9, %q9 = qs.apply %testAdj(%x, %r8, %n, %q8) : !qs.circ(i32, !qs.rstate<>, index, !qs.qstate -> !qs.rstate<>, !qs.qstate)
