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

func @main() {
    %N = constant 10 : i32
    %b = constant 4 : i32
    %e = constant 3 : i32

    %res = call @mod_exp(%b, %e, %N) : (i32, i32, i32) -> i32
    vector.print %res : i32

    return
}
