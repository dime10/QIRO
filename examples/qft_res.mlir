module {
  func @calc_qft_angle(%arg0: index) -> f64 {
    %cst = constant 3.1415926535897931 : f64
    %c1 = constant 1 : index
    %0 = addi %arg0, %c1 : index
    %1 = shift_left %c1, %0 : index
    %2 = index_cast %1 : index to i32
    %3 = uitofp %2 : i32 to f64
    %4 = divf %cst, %3 : f64
    return %4 : f64
  }
  func @QFT(%arg0: i64, %arg1: i64, %arg2: index) -> (i64, i64) attributes {_was_circ} {
    %c0_i64 = constant 0 : i64
    %c3_i64 = constant 3 : i64
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0:2 = scf.for %arg3 = %c0 to %arg2 step %c1 iter_args(%arg4 = %c0_i64, %arg5 = %c0_i64) -> (i64, i64) {
      %5 = addi %arg3, %c1 : index
      %6 = subi %arg2, %5 : index
      %7:2 = scf.for %arg6 = %c0 to %6 step %c1 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (i64, i64) {
        %8 = call @calc_qft_angle(%arg6) : (index) -> f64
        %9 = addi %arg7, %c3_i64 : i64
        scf.yield %9, %arg8 : i64, i64
      }
      scf.yield %7#0, %7#1 : i64, i64
    }
    %1 = divi_unsigned %arg2, %c2 : index
    %2:2 = scf.for %arg3 = %c0 to %1 step %c1 iter_args(%arg4 = %0#0, %arg5 = %0#1) -> (i64, i64) {
      scf.yield %arg4, %arg5 : i64, i64
    }
    %3 = addi %arg0, %2#0 : i64
    %4 = addi %arg1, %2#1 : i64
    return %3, %4 : i64, i64
  }
  func @run_qft(%arg0: i64, %arg1: i64, %arg2: index) -> (i64, i64) attributes {_was_circ} {
    %c0_i64 = constant 0 : i64
    %0:2 = call @QFT(%c0_i64, %c0_i64, %arg2) : (i64, i64, index) -> (i64, i64)
    %1 = addi %arg0, %0#0 : i64
    %2 = addi %arg1, %0#1 : i64
    return %1, %2 : i64, i64
  }
  func @main() {
    %c0_i64 = constant 0 : i64
    %c5 = constant 5 : index
    %0:2 = call @run_qft(%c0_i64, %c0_i64, %c5) : (i64, i64, index) -> (i64, i64)
    vector.print %0#0 : i64
    vector.print %0#1 : i64
    return
  }
}
