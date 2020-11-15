func @mod(%a: i64, %N: i64) -> i64 {
    %0 = divi_unsigned %a, %N : i64
    %1 = muli %N, %0 : i64
    %2 = subi %a, %1 : i64
    return %2 : i64
}

func @mod_exp(%b: i64, %e: i64, %N: i64) -> i64 {
    %c0 = constant 0 : i64
    %c1 = constant 1 : i64
    %c2 = constant 2 : i64
    %cond = cmpi "eq", %N, %c1 : i64
    cond_br %cond, ^ret(%c0 : i64), ^reduce

    ^reduce:
        %res = constant 1 : i64
        %base = call @mod(%b, %N) : (i64, i64) -> i64
        %cond2 = cmpi "ugt", %e, %c0 : i64
        cond_br %cond2, ^while(%base, %e, %res : i64, i64, i64), ^ret(%res : i64)

    ^while(%base_0: i64, %exp_0: i64, %res_0: i64):
        %0 = call @mod(%exp_0, %c2) : (i64, i64) -> i64
        %cond3 = cmpi "eq", %0, %c1 : i64
        %res_1 = scf.if %cond3 -> i64 {
            %1 = muli %res_0, %base_0 : i64
            %2 = call @mod(%1, %N) : (i64, i64) -> i64
            scf.yield %2 : i64
        } else {
            scf.yield %res_0 : i64
        }

        %exp_1 = shift_right_unsigned %exp_0, %c1 : i64

        %3 = muli %base_0, %base_0 : i64
        %base_1 = call @mod(%3, %N) : (i64, i64) -> i64

        %cond4 = cmpi "ugt", %exp_1, %c0 : i64
        cond_br %cond4, ^while(%base_1, %exp_1, %res_1 : i64, i64, i64), ^ret(%res_1 : i64)

    ^ret(%r: i64):
        return %r : i64
}

func @mod_inv(%C: i64, %N: i64) -> i64 {
    %c0 = constant 0 : i64
    %c1 = constant 1 : i64
    br ^while(%N, %C, %c0, %c1 : i64, i64, i64, i64)

    ^while(%r_0: i64, %old_r: i64, %s_0: i64, %old_s: i64):
        %q = divi_unsigned %old_r, %r_0 : i64
        %qr = muli %q, %r_0 : i64
        %r_1 = subi %old_r, %qr : i64

        %qs = muli %q, %s_0 : i64
        %s_1 = subi %old_s, %qs : i64

        %cond = cmpi "ne", %r_1, %c0 : i64
        cond_br %cond, ^while(%r_1, %r_0, %s_1, %s_0 : i64, i64, i64, i64), ^ret(%s_0 : i64)

    ^ret(%s: i64):
        %0 = addi %s, %N : i64
        %1 = call @mod(%0, %N) : (i64, i64) -> i64
        return %1 : i64
}

func @calc_qft_angle(%j: index) -> f64 {
    %pi = constant 3.141592653589793238 : f64
    %c1 = constant 1 : index
    %0 = addi %c1, %j : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i64
    %3 = uitofp %2 : i64 to f64
    %4 = divf %pi, %3 : f64
    return %4 : f64
}

func @calc_add_angle(%i: index, %j: index) -> f64 {
    %pi = constant 3.141592653589793238 : f64
    %c1 = constant 1 : index
    %0 = subi %i, %j : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i64
    %3 = uitofp %2 : i64 to f64
    %4 = divf %pi, %3 : f64
    return %4 : f64
}

func @calc_cur_a(%N: i64, %n: index, %a: i64, %i: index) -> i64 {
    %c1 = constant 1 : i64
    %c2 = constant 2 : i64
    %k = index_cast %i : index to i64
    %nbits = index_cast %n : index to i64

    %0 = muli %nbits, %c2 : i64
    %1 = subi %0, %c1 : i64
    %2 = subi %1, %k : i64
    %3 = shift_left %c1, %2 : i64
    %4 = call @mod_exp(%a, %3, %N) : (i64, i64, i64) -> i64

    return %4 : i64
}

func @calc_shor_angle(%i: index, %j: index) -> f64 {
    %mpi = constant -3.141592653589793238 : f64
    %c1 = constant 1 : index
    %0 = subi %i, %j : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i64
    %3 = uitofp %2 : i64 to f64
    %4 = divf %mpi, %3 : f64
    return %4 : f64
}

// quantum fourier transform on register r
q.circ @QFT(%r: !q.qureg<>, %n : index) attributes {no_inline} {
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

// add a positive or negative constant to register of size n
q.circ @addConstant(%C: i64, %r: !q.qureg<>, %n: index) {
    %c0 = constant 0 : index
    %s1 = constant 1 : index
    %c1 = constant 1 : i64

    // compute
    q.call @QFT(%r, %n) {compute} : !q.qureg<>, index

    scf.for %i = %c0 to %n step %s1 {
        %ip1 = addi %i, %s1 : index
        scf.for %j = %c0 to %ip1 step %s1 {
            %k = subi %i, %j : index
            %0 = index_cast %k : index to i64
            %1 = shift_right_signed %C, %0 : i64
            %2 = and %1, %c1 : i64
            %cond = cmpi "eq", %2, %c1 : i64
            scf.if %cond {
                %phi = call @calc_add_angle(%i, %k) : (index, index) -> f64
                q.R(%phi: f64) %r[%i] : !q.qureg<>
            }
        }
    }

    // uncompute
    %qft = q.getval @QFT -> !q.circ
    %qft_inv = q.adj %qft : !q.circ -> !q.circ
    q.apply %qft_inv(%r, %n) {uncompute} : !q.circ(!q.qureg<>, index)
}

// substract a constant from register of size n
q.circ @subConstant(%C: i64, %r: !q.qureg<>, %n: index) {
    %cm1 = constant -1 : i64
    %mC = muli %C, %cm1 : i64
    q.call @addConstant(%mC, %r, %n) : i64, !q.qureg<>, index
}

// add a positive constant to register modulo N
q.circ @addCmodN(%C: i64, %N: i64, %r: !q.qureg<>, %n: index) {
    %c1 = constant 1 : index
    %nm1 = subi %n, %c1 : index

    q.call @addConstant(%C, %r, %n) : i64, !q.qureg<>, index

    // compute
    q.call @subConstant(%N, %r, %n) {compute} : i64, !q.qureg<>, index
    %anc = q.alloc -> !q.qubit
    q.CX %r[%nm1], %anc {compute} : !q.qureg<>, !q.qubit
    %addOp = q.getval @addConstant -> !q.circ
    %ctrlAdd = q.ctrl %addOp, %anc : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
    q.apply %ctrlAdd(%N, %r, %n) {compute} : !q.cop<1, !q.circ>(i64, !q.qureg<>, index)

    q.call @subConstant(%C, %r, %n) : i64, !q.qureg<>, index

    // uncompute
    q.X %r[%nm1] {uncompute} : !q.qureg<>
    q.CX %r[%nm1], %anc {uncompute} : !q.qureg<>, !q.qubit
    q.X %r[%nm1] {uncompute} : !q.qureg<>
    q.free %anc : !q.qubit

    q.call @addConstant(%C, %r, %n) : i64, !q.qureg<>, index
}

// subtract a positive constant to register modulo N
q.circ @subCmodN(%C: i64, %N: i64, %r: !q.qureg<>, %n: index) {
    %NmC = subi %N, %C : i64
    q.call @addCmodN(%NmC, %N, %r, %n) : i64, i64, !q.qureg<>, index
}

// multiply a positive constant by a register modulo N, need gcd(C, N) = 1
q.circ @mulCmodN(%C: i64, %N: i64, %r: !q.qureg<>, %n: index) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %np1 = addi %n, %c1 : index
    %anc = q.allocreg(%np1) -> !q.qureg<>
    %Cinv = call @mod_inv(%C, %N) : (i64, i64) -> i64

    scf.for %i = %c0 to %n step %c1 {
        %addOp = q.getval @addCmodN -> !q.circ
        %ctrlAdd = q.ctrl %addOp, %r[%i] : !q.circ, !q.qureg<> -> !q.cop<1, !q.circ>

        %0 = index_cast %i : index to i64
        %1 = shift_left %C, %0 : i64
        %2 = call @mod(%1, %N) : (i64, i64) -> i64
        q.apply %ctrlAdd(%2, %N, %anc, %np1) : !q.cop<1, !q.circ>(i64, i64, !q.qureg<>, index)
    }

    scf.for %i = %c0 to %n step %c1 {
        q.SWAP %anc[%i], %r[%i] : !q.qureg<> , !q.qureg<>
    }

    scf.for %i = %c0 to %n step %c1 {
        %subOp = q.getval @subCmodN -> !q.circ
        %ctrlSub = q.ctrl %subOp, %r[%i] : !q.circ, !q.qureg<> -> !q.cop<1, !q.circ>

        %3 = index_cast %i : index to i64
        %4 = shift_left %Cinv, %3 : i64
        %5 = call @mod(%4, %N) : (i64, i64) -> i64
        q.apply %ctrlSub(%5, %N, %anc, %np1) : !q.cop<1, !q.circ>(i64, i64, !q.qureg<>, index)
    }

    q.freereg %anc : !q.qureg<>
}

q.circ @shor(%N: i64, %a: i64) attributes {no_inline_target} {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    %0 = uitofp %N : i64 to f64
    %1 = log2 %0 : f64
    %2 = ceilf %1 : f64
    %3 = fptoui %2 : f64 to i64
    %n = index_cast %3 : i64 to index
    %n2 = muli %n, %c2 : index

    %m0 = constant 0 : i1
    %meas = alloc(%n2) : memref<?xi1>
    scf.for %i = %c0 to %n2 step %c1 {
        store %m0, %meas[%i] : memref<?xi1>
    }

    %r = q.allocreg(%n) -> !q.qureg<>
    %cqb = q.alloc -> !q.qubit

    q.X %r[0] : !q.qureg<>

    scf.for %i = %c0 to %n2 step %c1 {
        %cur_a = call @calc_cur_a(%N, %n, %a, %i) : (i64, index, i64, index) -> i64

        q.H %cqb : !q.qubit
        %mulOp = q.getval @mulCmodN -> !q.circ
        %ctrlMul = q.ctrl %mulOp, %cqb : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
        q.apply %ctrlMul(%cur_a, %N, %r, %n) : !q.cop<1, !q.circ>(i64, i64, !q.qureg<>, index)

        scf.for %j = %c0 to %i step %c1 {
            %cond = load %meas[%j] : memref<?xi1>
            scf.if %cond {
                %phi = call @calc_shor_angle(%i, %j) : (index, index) -> f64
                q.R(%phi: f64) %cqb : !q.qubit
            }
        }
        q.H %cqb : !q.qubit

        %m = q.meas %cqb : !q.qubit -> i1
        store %m, %meas[%i] : memref<?xi1>
        scf.if %m {
            q.X %cqb : !q.qubit
        }
    }

    q.free %cqb : !q.qubit
    q.freereg %r : !q.qureg<>

    // return result
}

q.circ @main() attributes {no_inline_target} {
    %N = constant 15 : i64
    %a = constant 2 : i64

    q.call @shor(%N, %a) : i64, i64
}
