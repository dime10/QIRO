%q = qs.alloc -> !qs.qstate
%r = qs.allocreg(8) -> !qs.rstate<8>
%phi = constant 0.5 : f64
%theta = constant 0.3 : f64

// cancel hermitian gates
%q1 = qs.H %q : !qs.qstate -> !qs.qstate
%q2 = qs.H %q1 : !qs.qstate -> !qs.qstate

%q3, %r1 = qs.CX %q2, %r : !qs.qstate, !qs.rstate<8> -> !qs.qstate, !qs.rstate<8>
%q4, %r2 = qs.CX %q3, %r1  : !qs.qstate, !qs.rstate<8> -> !qs.qstate, !qs.rstate<8>

// don't cancel different hermitian gates
%q5 = qs.H %q4 : !qs.qstate -> !qs.qstate
%q6 = qs.X %q5 : !qs.qstate -> !qs.qstate

// merge rotations
%r3 = qs.R(%phi) %r2 : f64, !qs.rstate<8> -> !qs.rstate<8>
%r4 = qs.R(%theta) %r3 : f64, !qs.rstate<8> -> !qs.rstate<8>
