%q = qs.alloc -> !qs.qstate
%r = qs.allocreg(8) -> !qs.rstate<8>
%phi = constant 0.5 : f64

%q1 = qs.H %q : !qs.qstate -> !qs.qstate
%H = qs.H -> !qs.u1
%q2 = qs.adj %H, %q1 : !qs.u1, !qs.qstate -> !qs.qstate

%q3 = qs.R(%phi) %q2 : f64, !qs.qstate -> !qs.qstate
%R = qs.R(%phi) : f64 -> !qs.u1
%q4 = qs.adj %R, %q3 : !qs.u1, !qs.qstate -> !qs.qstate

%q5, %r1 = qs.CX %q4, %r : !qs.qstate, !qs.rstate<8> -> !qs.qstate, !qs.rstate<8>
%cx = qs.CX -> !qs.u2
%q6, %r2 = qs.adj %cx, %q5, %r1 : !qs.u2, !qs.qstate, !qs.rstate<8> -> !qs.qstate, !qs.rstate<8>

%q7, %r3 = qs.ctrl %H, %q6, %r2 : !qs.u1, !qs.qstate, !qs.rstate<8> -> !qs.qstate, !qs.rstate<8>
%q8, %cH = qs.ctrl %H, %q7 : !qs.u1, !qs.qstate -> !qs.qstate, !qs.cop<1, !qs.u1>
%r4 = qs.adj %cH, %r3 : !qs.cop<1, !qs.u1>, !qs.rstate<8> -> !qs.rstate<8>

%q9, %cH2 = qs.ctrl %H, %q8 : !qs.u1, !qs.qstate -> !qs.qstate, !qs.cop<1, !qs.u1>
%r5 = qs.adj %cH2, %r4 : !qs.cop<1, !qs.u1>, !qs.rstate<8> -> !qs.rstate<8>
%q10, %r6 = qs.ctrl %H, %q9, %r5 : !qs.u1, !qs.qstate, !qs.rstate<8> -> !qs.qstate, !qs.rstate<8>

%r7 = qs.adj %H, %r6 : !qs.u1, !qs.rstate<8> -> !qs.rstate<8>
%Hadj = qs.adj %H : !qs.u1 -> !qs.u1
%r8 = qs.adj %Hadj, %r7 : !qs.u1, !qs.rstate<8> -> !qs.rstate<8>

%Hadj2 = qs.adj %H : !qs.u1 -> !qs.u1
%r9 = qs.adj %Hadj2, %r8 : !qs.u1, !qs.rstate<8> -> !qs.rstate<8>
%r10 = qs.adj %H, %r9 : !qs.u1, !qs.rstate<8> -> !qs.rstate<8>
