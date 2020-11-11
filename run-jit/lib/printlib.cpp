#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

extern "C" void print_i32(int32_t i) { fprintf(stdout, "%" PRId32, i); }
extern "C" void print_i64(int64_t l) { fprintf(stdout, "%" PRId64, l); }
extern "C" void print_f32(float f)   { fprintf(stdout, "%g", f); }
extern "C" void print_f64(double d)  { fprintf(stdout, "%lg", d); }
extern "C" void print_open()         { fputs("( ", stdout); }
extern "C" void print_close()        { fputs(" )", stdout); }
extern "C" void print_comma()        { fputs(", ", stdout); }
extern "C" void print_newline()      { fputc('\n', stdout); }
