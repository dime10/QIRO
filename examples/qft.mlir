//===------------------------------------------------------------------------------------------===//
// QFT
//===------------------------------------------------------------------------------------------===//

func @calc_qft_angle(%j: index) -> f64 {
    %pi = constant 3.141592653589793238 : f64
    %c1 = constant 1 : index
    %0 = addi %c1, %j : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i32
    %3 = uitofp %2 : i32 to f64
    %4 = divf %pi, %3 : f64
    return %4 : f64
}

// quantum fourier transform on register r
q.circ @QFT(%r: !q.qureg<>, %n : index) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    scf.for %i = %c0 to %n step %c1 {
        %0 = addi %i, %c1 : index
        %k = subi %n, %0 : index
        q.H %r[%k] : !q.qureg<>
        scf.for %j = %c0 to %k step %c1 {
            %phi = call @calc_qft_angle(%j) : (index) -> f64
            %R = q.R(%phi: f64) -> !q.u1
            %1 = addi %j, %c1 : index
            %h = subi %k, %1 : index
            q.ctrl %R, %r[%h], %r[%k] : !q.u1, !q.qureg<>, !q.qureg<>
        }
    }

    %nd2 = divi_unsigned %n, %c2 : index
    scf.for %i = %c0 to %nd2 step %c1 {
        %0 = addi %i, %c1 : index
        %j = subi %n, %0 : index
        q.SWAP %r[%i], %r[%j] : !q.qureg<>, !q.qureg<>
    }
}

q.circ @run_qft(%n : index) {
    %r = q.allocreg(%n) -> !q.qureg<>
    q.call @QFT(%r, %n) : !q.qureg<>, index
}

q.circ @main() attributes {no_inline_target} {
    %n = constant 5 : index
    q.call @run_qft(%n) : index
}
