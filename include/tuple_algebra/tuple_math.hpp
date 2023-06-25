#include <cmath>
#include "tuple_algebra.hpp"

// unary
TP_MAP_UNARY_STD_FN(fpclassify);
TP_MAP_UNARY_STD_FN(isfinite);
TP_MAP_UNARY_STD_FN(isnan);
TP_MAP_UNARY_STD_FN(isnormal);

TP_MAP_UNARY_STD_FN(abs);
TP_MAP_UNARY_STD_FN(fabs);

TP_MAP_UNARY_STD_FN(exp);
TP_MAP_UNARY_STD_FN(exp2);
TP_MAP_UNARY_STD_FN(expm1);
TP_MAP_UNARY_STD_FN(log);
TP_MAP_UNARY_STD_FN(log10);
TP_MAP_UNARY_STD_FN(log2);
TP_MAP_UNARY_STD_FN(log1p);

TP_MAP_UNARY_STD_FN(sqrt);
TP_MAP_UNARY_STD_FN(cbrt);

TP_MAP_UNARY_STD_FN(sin);
TP_MAP_UNARY_STD_FN(cos);
TP_MAP_UNARY_STD_FN(tan);
TP_MAP_UNARY_STD_FN(asin);
TP_MAP_UNARY_STD_FN(acos);
TP_MAP_UNARY_STD_FN(atan);
TP_MAP_UNARY_STD_FN(atan2);

TP_MAP_UNARY_STD_FN(sinh);
TP_MAP_UNARY_STD_FN(cosh);
TP_MAP_UNARY_STD_FN(tanh);
TP_MAP_UNARY_STD_FN(asinh);
TP_MAP_UNARY_STD_FN(acosh);
TP_MAP_UNARY_STD_FN(atanh);

TP_MAP_UNARY_STD_FN(erf);
TP_MAP_UNARY_STD_FN(erfc);
TP_MAP_UNARY_STD_FN(tgamma);
TP_MAP_UNARY_STD_FN(lgamma);

TP_MAP_UNARY_STD_FN(ceil);
TP_MAP_UNARY_STD_FN(floor);
TP_MAP_UNARY_STD_FN(trunc);
TP_MAP_UNARY_STD_FN(round);
TP_MAP_UNARY_STD_FN(nearbyint);
TP_MAP_UNARY_STD_FN(rint);
TP_MAP_UNARY_STD_FN(lrint);
TP_MAP_UNARY_STD_FN(llrint);

// binary
TP_MAKE_BINARY_OP(operator<, a < b);
TP_MAKE_BINARY_OP(operator<=, a <= b);
TP_MAKE_BINARY_OP(operator>, a > b);
TP_MAKE_BINARY_OP(operator>=, a >= b);
TP_MAKE_BINARY_OP(operator==, a == b);
TP_MAKE_BINARY_OP(operator<=>, a <=> b);
TP_MAKE_BINARY_OP(operator||, a || b);
TP_MAKE_BINARY_OP(operator&&, a && b);

TP_MAP_BINARY_STD_FN(pow);

TP_MAP_BINARY_STD_FN(fdim);
TP_MAP_BINARY_STD_FN(fmin);
TP_MAP_BINARY_STD_FN(fmax);

TP_MAP_BINARY_STD_FN(hypot);
TP_MAP_BINARY_STD_FN(fmod);
TP_MAP_BINARY_STD_FN(remainder);

// ternary
TP_MAP_TERNARY_STD_FN(hypot);
TP_MAP_TERNARY_STD_FN(fma);
TP_MAP_TERNARY_STD_FN(lerp);
