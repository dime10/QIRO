func @mod(%a: i32, %N: i32) -> i32 {
    %0 = divi_unsigned %a, %N : i32
    %1 = muli %N, %0 : i32
    %2 = subi %a, %1 : i32
    return %2 : i32
}

func @mod_exp(%b: i32, %e: i32, %N: i32) -> i32 {
    %c0 = constant 0 : i32
    %c1 = constant 1 : i32
    %c2 = constant 2 : i32
    %cond = cmpi "eq", %N, %c1 : i32
    cond_br %cond, ^ret(%c0 : i32), ^reduce

    ^reduce:
        %res = constant 1 : i32
        %base = call @mod(%b, %N) : (i32, i32) -> i32
        %cond2 = cmpi "ugt", %e, %c0 : i32
        cond_br %cond2, ^while(%base, %e, %res : i32, i32, i32), ^ret(%res : i32)

    ^while(%base_0: i32, %exp_0: i32, %res_0: i32):
        %0 = call @mod(%exp_0, %c2) : (i32, i32) -> i32
        %cond3 = cmpi "eq", %0, %c1 : i32
        %res_1 = scf.if %cond3 -> i32 {
            %1 = muli %res_0, %base_0 : i32
            %2 = call @mod(%1, %N) : (i32, i32) -> i32
            scf.yield %2 : i32
        } else {
            scf.yield %res_0 : i32
        }

        %exp_1 = shift_right_unsigned %exp_0, %c1 : i32

        %3 = muli %base_0, %base_0 : i32
        %base_1 = call @mod(%3, %N) : (i32, i32) -> i32

        %cond4 = cmpi "ugt", %exp_1, %c0 : i32
        cond_br %cond4, ^while(%base_1, %exp_1, %res_1 : i32, i32, i32), ^ret(%res_1 : i32)

    ^ret(%r: i32):
        return %r : i32
}

func @mod_inv(%C: i32, %N: i32) -> i32 {
    %c0 = constant 0 : i32
    %c1 = constant 1 : i32
    br ^while(%N, %C, %c0, %c1 : i32, i32, i32, i32)

    ^while(%r_0: i32, %old_r: i32, %s_0: i32, %old_s: i32):
        %q = divi_unsigned %old_r, %r_0 : i32
        %qr = muli %q, %r_0 : i32
        %r_1 = subi %old_r, %qr : i32

        %qs = muli %q, %s_0 : i32
        %s_1 = subi %old_s, %qs : i32

        %cond = cmpi "ne", %r_1, %c0 : i32
        cond_br %cond, ^while(%r_1, %r_0, %s_1, %s_0 : i32, i32, i32, i32), ^ret(%s_0 : i32)

    ^ret(%s: i32):
        %0 = addi %s, %N : i32
        %1 = call @mod(%0, %N) : (i32, i32) -> i32
        return %1 : i32
}

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

func @calc_add_angle(%i: index, %j: index) -> f64 {
    %pi = constant 3.141592653589793238 : f64
    %c1 = constant 1 : index
    %0 = subi %i, %j : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i32
    %3 = uitofp %2 : i32 to f64
    %4 = divf %pi, %3 : f64
    return %4 : f64
}

func @calc_cur_a(%N: i32, %n: index, %a: i32, %i: index) -> i32 {
    %c1 = constant 1 : i32
    %c2 = constant 2 : i32
    %k = index_cast %i : index to i32
    %nbits = index_cast %n : index to i32

    %0 = muli %nbits, %c2 : i32
    %1 = subi %0, %c1 : i32
    %2 = subi %1, %k : i32
    %3 = shift_left %c1, %2 : i32
    %4 = call @mod_exp(%a, %3, %N) : (i32, i32, i32) -> i32

    return %4 : i32
}

func @calc_shor_angle(%i: index, %j: index) -> f64 {
    %mpi = constant -3.141592653589793238 : f64
    %c1 = constant 1 : index
    %0 = subi %i, %j : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i32
    %3 = uitofp %2 : i32 to f64
    %4 = divf %mpi, %3 : f64
    return %4 : f64
}

// quantum fourier transform on register r
q.circ @QFT(%r: !q.qureg<>, %n : index) {
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    affine.for %i = 0 to %n {
        %0 = addi %i, %c1 : index
        %k = subi %n, %0 : index
        q.H %r[%k] : !q.qureg<>
        affine.for %j = 0 to affine_map<(n,i)->(n-i-1)>(%n,%i) {
            %phi = call @calc_qft_angle(%j) : (index) -> f64
            %R = q.R(%phi: f64) -> !q.u1
            %1 = addi %j, %c1 : index
            %h = subi %k, %1 : index
            q.ctrl %R, %r[%h], %r[%k] : !q.u1, !q.qureg<>, !q.qureg<>
        }
    }

    %nd2 = divi_unsigned %n, %c2 : index
    affine.for %i = 0 to %nd2 {
        %0 = addi %i, %c1 : index
        %j = subi %n, %0 : index
        q.SWAP %r[%i], %r[%j] : !q.qureg<>, !q.qureg<>
    }
}

// add a positive or negative constant to register of size n
q.circ @addConstant(%C: i32, %r: !q.qureg<>, %n: index) {
    %c1 = constant 1 : i32
    // compute
    q.call @QFT(%r, %n) : !q.qureg<>, index

    affine.for %i = 0 to %n {
        affine.for %j = 0 to affine_map<(i)->(i+1)>(%i) {
            %k = subi %i, %j : index
            %0 = index_cast %k : index to i32
            %1 = shift_right_signed %C, %0 : i32
            %2 = and %1, %c1 : i32
            %cond = cmpi "eq", %2, %c1 : i32
            scf.if %cond {
                %phi = call @calc_add_angle(%i, %k) : (index, index) -> f64
                q.R(%phi: f64) %r[%i] : !q.qureg<>
            }
        }
    }

    // uncompute
    %qft = q.getval @QFT -> !q.circ
    %qft_inv = q.adj %qft : !q.circ -> !q.circ
    q.apply %qft_inv(%r, %n) : !q.circ(!q.qureg<>, index)
}

// substract a constant from register of size n
q.circ @subConstant(%C: i32, %r: !q.qureg<>, %n: index) {
    %cm1 = constant -1 : i32
    %mC = muli %C, %cm1 : i32
    q.call @addConstant(%mC, %r, %n) : i32, !q.qureg<>, index
}

// add a positive constant to register modulo N
q.circ @addCmodN(%C: i32, %N: i32, %r: !q.qureg<>, %n: index) {
    %c1 = constant 1 : index
    %nm1 = subi %n, %c1 : index

    q.call @addConstant(%C, %r, %n) : i32, !q.qureg<>, index

    // compute
    q.call @subConstant(%N, %r, %n) : i32, !q.qureg<>, index
    %anc = q.alloc -> !q.qubit
    q.CX %r[%nm1], %anc : !q.qureg<>, !q.qubit
    %addOp = q.getval @addConstant -> !q.circ
    %ctrlAdd = q.ctrl %addOp, %anc : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
    q.apply %ctrlAdd(%N, %r, %n) : !q.cop<1, !q.circ>(i32, !q.qureg<>, index)

    q.call @subConstant(%C, %r, %n) : i32, !q.qureg<>, index

    // uncompute
    q.X %r[%nm1] : !q.qureg<>
    q.CX %r[%nm1], %anc : !q.qureg<>, !q.qubit
    q.X %r[%nm1] : !q.qureg<>
    q.free %anc : !q.qubit

    q.call @addConstant(%C, %r, %n) : i32, !q.qureg<>, index
}

// subtract a positive constant to register modulo N
q.circ @subCmodN(%C: i32, %N: i32, %r: !q.qureg<>, %n: index) {
    %NmC = subi %N, %C : i32
    q.call @addCmodN(%NmC, %N, %r, %n) : i32, i32, !q.qureg<>, index
}

// multiply a positive constant by a register modulo N, need gcd(C, N) = 1
q.circ @mulCmodN(%C: i32, %N: i32, %r: !q.qureg<>, %n: index) {
    %c1 = constant 1 : index
    %np1 = muli %n, %c1 : index
    %anc = q.allocreg(%np1) -> !q.qureg<>
    %Cinv = call @mod_inv(%C, %N) : (i32, i32) -> i32

    affine.for %i = 0 to %n {
        %addOp = q.getval @addCmodN -> !q.circ
        %ctrlAdd = q.ctrl %addOp, %r[%i] : !q.circ, !q.qureg<> -> !q.cop<1, !q.circ>

        %0 = index_cast %i : index to i32
        %1 = shift_left %C, %0 : i32
        %2 = call @mod(%1, %N) : (i32, i32) -> i32
        q.apply %ctrlAdd(%2, %N, %anc, %np1) : !q.cop<1, !q.circ>(i32, i32, !q.qureg<>, index)
    }

    affine.for %i = 0 to %n {
        q.SWAP %anc[%i], %r[%i] : !q.qureg<> , !q.qureg<>
    }

    affine.for %i = 0 to %n {
        %subOp = q.getval @subCmodN -> !q.circ
        %ctrlSub = q.ctrl %subOp, %r[%i] : !q.circ, !q.qureg<> -> !q.cop<1, !q.circ>

        %3 = index_cast %i : index to i32
        %4 = shift_left %Cinv, %3 : i32
        %5 = call @mod(%4, %N) : (i32, i32) -> i32
        q.apply %ctrlSub(%5, %N, %anc, %np1) : !q.cop<1, !q.circ>(i32, i32, !q.qureg<>, index)
    }

    q.freereg %anc : !q.qureg<>
}

q.circ @shor(%N: i32, %a: i32) {
    %0 = uitofp %N : i32 to f64
    %1 = log2 %0 : f64
    %2 = ceilf %0 : f64
    %3 = fptoui %2 : f64 to i32
    %n = index_cast %3 : i32 to index
    %c2 = constant 2 : index
    %n2 = muli %n, %c2 : index

    %c0 = constant 0 : i1
    %meas = alloc(%n2) : memref<?xi1>
    affine.for %i = 0 to %n2 {
        store %c0, %meas[%i] : memref<?xi1>
    }

    %r = q.allocreg(%n) -> !q.qureg<>
    %cqb = q.alloc -> !q.qubit

    q.X %r[0] : !q.qureg<>

    affine.for %i = 0 to %n2 {
        %cur_a = call @calc_cur_a(%N, %n, %a, %i) : (i32, index, i32, index) -> i32

        q.H %cqb : !q.qubit
        %mulOp = q.getval @mulCmodN -> !q.circ
        %ctrlMul = q.ctrl %mulOp, %cqb : !q.circ, !q.qubit -> !q.cop<1, !q.circ>
        q.apply %ctrlMul(%cur_a, %N, %r, %n) : !q.cop<1, !q.circ>(i32, i32, !q.qureg<>, index)

        affine.for %j = 0 to affine_map<(i)->(i)>(%i) {
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
