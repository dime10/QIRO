; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @print_newline()

declare void @print_i64(i64)

define double @calc_qft_angle(i64 %0) !dbg !3 {
  %2 = add i64 %0, 1, !dbg !7
  %3 = shl i64 1, %2, !dbg !9
  %4 = trunc i64 %3 to i32, !dbg !10
  %5 = uitofp i32 %4 to double, !dbg !11
  %6 = fdiv double 0x400921FB54442D18, %5, !dbg !12
  ret double %6, !dbg !13
}

define { i64, i64 } @QFT(i64 %0, i64 %1, i64 %2) !dbg !14 {
  br label %4, !dbg !15

4:                                                ; preds = %25, %3
  %5 = phi i64 [ %26, %25 ], [ 0, %3 ]
  %6 = phi i64 [ %14, %25 ], [ 0, %3 ]
  %7 = phi i64 [ %15, %25 ], [ 0, %3 ]
  %8 = icmp slt i64 %5, %2, !dbg !15
  br i1 %8, label %9, label %27, !dbg !15

9:                                                ; preds = %4
  %10 = add i64 %5, 1, !dbg !17
  %11 = sub i64 %2, %10, !dbg !18
  br label %12, !dbg !19

12:                                               ; preds = %17, %9
  %13 = phi i64 [ %24, %17 ], [ 0, %9 ]
  %14 = phi i64 [ %23, %17 ], [ %6, %9 ]
  %15 = phi i64 [ %15, %17 ], [ %7, %9 ]
  %16 = icmp slt i64 %13, %11, !dbg !19
  br i1 %16, label %17, label %25, !dbg !19

17:                                               ; preds = %12
  %18 = add i64 %13, 1, !dbg !20
  %19 = shl i64 1, %18, !dbg !22
  %20 = trunc i64 %19 to i32, !dbg !23
  %21 = uitofp i32 %20 to double, !dbg !24
  %22 = fdiv double 0x400921FB54442D18, %21, !dbg !25
  %23 = add i64 %14, 3, !dbg !26
  %24 = add i64 %13, 1, !dbg !19
  br label %12, !dbg !19

25:                                               ; preds = %12
  %26 = add i64 %5, 1, !dbg !15
  br label %4, !dbg !15

27:                                               ; preds = %4
  %28 = udiv i64 %2, 2, !dbg !27
  br label %29, !dbg !28

29:                                               ; preds = %34, %27
  %30 = phi i64 [ %35, %34 ], [ 0, %27 ]
  %31 = phi i64 [ %31, %34 ], [ %6, %27 ]
  %32 = phi i64 [ %32, %34 ], [ %7, %27 ]
  %33 = icmp slt i64 %30, %28, !dbg !28
  br i1 %33, label %34, label %36, !dbg !28

34:                                               ; preds = %29
  %35 = add i64 %30, 1, !dbg !28
  br label %29, !dbg !28

36:                                               ; preds = %29
  %37 = add i64 %0, %31, !dbg !29
  %38 = add i64 %1, %32, !dbg !29
  %39 = insertvalue { i64, i64 } undef, i64 %37, 0, !dbg !29
  %40 = insertvalue { i64, i64 } %39, i64 %38, 1, !dbg !29
  ret { i64, i64 } %40, !dbg !29
}

define { i64, i64 } @run_qft(i64 %0, i64 %1, i64 %2) !dbg !30 {
  br label %4, !dbg !31

4:                                                ; preds = %23, %3
  %5 = phi i64 [ %24, %23 ], [ 0, %3 ]
  %6 = phi i64 [ %13, %23 ], [ 0, %3 ]
  %7 = icmp slt i64 %5, %2, !dbg !31
  br i1 %7, label %8, label %25, !dbg !31

8:                                                ; preds = %4
  %9 = add i64 %5, 1, !dbg !34
  %10 = sub i64 %2, %9, !dbg !35
  br label %11, !dbg !36

11:                                               ; preds = %15, %8
  %12 = phi i64 [ %22, %15 ], [ 0, %8 ]
  %13 = phi i64 [ %21, %15 ], [ %6, %8 ]
  %14 = icmp slt i64 %12, %10, !dbg !36
  br i1 %14, label %15, label %23, !dbg !36

15:                                               ; preds = %11
  %16 = add i64 %12, 1, !dbg !37
  %17 = shl i64 1, %16, !dbg !39
  %18 = trunc i64 %17 to i32, !dbg !40
  %19 = uitofp i32 %18 to double, !dbg !41
  %20 = fdiv double 0x400921FB54442D18, %19, !dbg !42
  %21 = add i64 %13, 3, !dbg !43
  %22 = add i64 %12, 1, !dbg !36
  br label %11, !dbg !36

23:                                               ; preds = %11
  %24 = add i64 %5, 1, !dbg !31
  br label %4, !dbg !31

25:                                               ; preds = %4
  %26 = udiv i64 %2, 2, !dbg !44
  br label %27, !dbg !45

27:                                               ; preds = %30, %25
  %28 = phi i64 [ %31, %30 ], [ 0, %25 ]
  %29 = icmp slt i64 %28, %26, !dbg !45
  br i1 %29, label %30, label %QFT.exit, !dbg !45

30:                                               ; preds = %27
  %31 = add i64 %28, 1, !dbg !45
  br label %27, !dbg !45

QFT.exit:                                         ; preds = %27
  %32 = insertvalue { i64, i64 } undef, i64 %6, 0, !dbg !46
  %33 = insertvalue { i64, i64 } %32, i64 0, 1, !dbg !46
  %34 = extractvalue { i64, i64 } %33, 0, !dbg !47
  %35 = extractvalue { i64, i64 } %33, 1, !dbg !47
  %36 = add i64 %0, %34, !dbg !48
  %37 = add i64 %1, %35, !dbg !48
  %38 = insertvalue { i64, i64 } undef, i64 %36, 0, !dbg !48
  %39 = insertvalue { i64, i64 } %38, i64 %37, 1, !dbg !48
  ret { i64, i64 } %39, !dbg !48
}

define void @main() !dbg !49 {
  br label %1, !dbg !50

1:                                                ; preds = %20, %0
  %2 = phi i64 [ %21, %20 ], [ 0, %0 ]
  %3 = phi i64 [ %10, %20 ], [ 0, %0 ]
  %4 = icmp slt i64 %2, 5, !dbg !50
  br i1 %4, label %5, label %22, !dbg !50

5:                                                ; preds = %1
  %6 = add i64 %2, 1, !dbg !54
  %7 = sub i64 5, %6, !dbg !55
  br label %8, !dbg !56

8:                                                ; preds = %12, %5
  %9 = phi i64 [ %19, %12 ], [ 0, %5 ]
  %10 = phi i64 [ %18, %12 ], [ %3, %5 ]
  %11 = icmp slt i64 %9, %7, !dbg !56
  br i1 %11, label %12, label %20, !dbg !56

12:                                               ; preds = %8
  %13 = add i64 %9, 1, !dbg !57
  %14 = shl i64 1, %13, !dbg !59
  %15 = trunc i64 %14 to i32, !dbg !60
  %16 = uitofp i32 %15 to double, !dbg !61
  %17 = fdiv double 0x400921FB54442D18, %16, !dbg !62
  %18 = add i64 %10, 3, !dbg !63
  %19 = add i64 %9, 1, !dbg !56
  br label %8, !dbg !56

20:                                               ; preds = %8
  %21 = add i64 %2, 1, !dbg !50
  br label %1, !dbg !50

22:                                               ; preds = %1
  br label %23, !dbg !64

23:                                               ; preds = %26, %22
  %24 = phi i64 [ %27, %26 ], [ 0, %22 ]
  %25 = icmp slt i64 %24, 2, !dbg !64
  br i1 %25, label %26, label %run_qft.exit, !dbg !64

26:                                               ; preds = %23
  %27 = add i64 %24, 1, !dbg !64
  br label %23, !dbg !64

run_qft.exit:                                     ; preds = %23
  %28 = insertvalue { i64, i64 } undef, i64 %3, 0, !dbg !65
  %29 = insertvalue { i64, i64 } %28, i64 0, 1, !dbg !65
  %30 = insertvalue { i64, i64 } undef, i64 %3, 0, !dbg !66
  %31 = insertvalue { i64, i64 } %30, i64 0, 1, !dbg !66
  %32 = extractvalue { i64, i64 } %31, 0, !dbg !67
  %33 = extractvalue { i64, i64 } %31, 1, !dbg !67
  call void @print_i64(i64 %32), !dbg !68
  call void @print_newline(), !dbg !68
  call void @print_i64(i64 %33), !dbg !68
  call void @print_newline(), !dbg !68
  ret void, !dbg !68
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "calc_qft_angle", linkageName: "calc_qft_angle", scope: null, file: !4, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "examples/qft.mlir", directory: "/home/dave/QCompile")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 8, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 9, column: 10, scope: !8)
!10 = !DILocation(line: 10, column: 10, scope: !8)
!11 = !DILocation(line: 11, column: 10, scope: !8)
!12 = !DILocation(line: 12, column: 10, scope: !8)
!13 = !DILocation(line: 13, column: 5, scope: !8)
!14 = distinct !DISubprogram(name: "QFT", linkageName: "QFT", scope: null, file: !4, line: 17, type: !5, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!15 = !DILocation(line: 22, column: 5, scope: !16)
!16 = !DILexicalBlockFile(scope: !14, file: !4, discriminator: 0)
!17 = !DILocation(line: 23, column: 14, scope: !16)
!18 = !DILocation(line: 24, column: 14, scope: !16)
!19 = !DILocation(line: 26, column: 9, scope: !16)
!20 = !DILocation(line: 8, column: 10, scope: !8, inlinedAt: !21)
!21 = distinct !DILocation(line: 27, column: 20, scope: !16)
!22 = !DILocation(line: 9, column: 10, scope: !8, inlinedAt: !21)
!23 = !DILocation(line: 10, column: 10, scope: !8, inlinedAt: !21)
!24 = !DILocation(line: 11, column: 10, scope: !8, inlinedAt: !21)
!25 = !DILocation(line: 12, column: 10, scope: !8, inlinedAt: !21)
!26 = !DILocation(line: 31, column: 13, scope: !16)
!27 = !DILocation(line: 35, column: 12, scope: !16)
!28 = !DILocation(line: 36, column: 5, scope: !16)
!29 = !DILocation(line: 17, column: 1, scope: !16)
!30 = distinct !DISubprogram(name: "run_qft", linkageName: "run_qft", scope: null, file: !4, line: 43, type: !5, scopeLine: 43, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!31 = !DILocation(line: 22, column: 5, scope: !16, inlinedAt: !32)
!32 = distinct !DILocation(line: 45, column: 5, scope: !33)
!33 = !DILexicalBlockFile(scope: !30, file: !4, discriminator: 0)
!34 = !DILocation(line: 23, column: 14, scope: !16, inlinedAt: !32)
!35 = !DILocation(line: 24, column: 14, scope: !16, inlinedAt: !32)
!36 = !DILocation(line: 26, column: 9, scope: !16, inlinedAt: !32)
!37 = !DILocation(line: 8, column: 10, scope: !8, inlinedAt: !38)
!38 = distinct !DILocation(line: 27, column: 20, scope: !16, inlinedAt: !32)
!39 = !DILocation(line: 9, column: 10, scope: !8, inlinedAt: !38)
!40 = !DILocation(line: 10, column: 10, scope: !8, inlinedAt: !38)
!41 = !DILocation(line: 11, column: 10, scope: !8, inlinedAt: !38)
!42 = !DILocation(line: 12, column: 10, scope: !8, inlinedAt: !38)
!43 = !DILocation(line: 31, column: 13, scope: !16, inlinedAt: !32)
!44 = !DILocation(line: 35, column: 12, scope: !16, inlinedAt: !32)
!45 = !DILocation(line: 36, column: 5, scope: !16, inlinedAt: !32)
!46 = !DILocation(line: 17, column: 1, scope: !16, inlinedAt: !32)
!47 = !DILocation(line: 45, column: 5, scope: !33)
!48 = !DILocation(line: 43, column: 1, scope: !33)
!49 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 48, type: !5, scopeLine: 48, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!50 = !DILocation(line: 22, column: 5, scope: !16, inlinedAt: !51)
!51 = distinct !DILocation(line: 45, column: 5, scope: !33, inlinedAt: !52)
!52 = distinct !DILocation(line: 50, column: 5, scope: !53)
!53 = !DILexicalBlockFile(scope: !49, file: !4, discriminator: 0)
!54 = !DILocation(line: 23, column: 14, scope: !16, inlinedAt: !51)
!55 = !DILocation(line: 24, column: 14, scope: !16, inlinedAt: !51)
!56 = !DILocation(line: 26, column: 9, scope: !16, inlinedAt: !51)
!57 = !DILocation(line: 8, column: 10, scope: !8, inlinedAt: !58)
!58 = distinct !DILocation(line: 27, column: 20, scope: !16, inlinedAt: !51)
!59 = !DILocation(line: 9, column: 10, scope: !8, inlinedAt: !58)
!60 = !DILocation(line: 10, column: 10, scope: !8, inlinedAt: !58)
!61 = !DILocation(line: 11, column: 10, scope: !8, inlinedAt: !58)
!62 = !DILocation(line: 12, column: 10, scope: !8, inlinedAt: !58)
!63 = !DILocation(line: 31, column: 13, scope: !16, inlinedAt: !51)
!64 = !DILocation(line: 36, column: 5, scope: !16, inlinedAt: !51)
!65 = !DILocation(line: 17, column: 1, scope: !16, inlinedAt: !51)
!66 = !DILocation(line: 43, column: 1, scope: !33, inlinedAt: !52)
!67 = !DILocation(line: 50, column: 5, scope: !53)
!68 = !DILocation(line: 48, column: 1, scope: !53)
