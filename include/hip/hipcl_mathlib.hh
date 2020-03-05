/*
 * This file provides math library prototypes for HIP device code,
 * which indirectly call OpenCL math library.
 * The reasons we can't directly call OpenCL here are
 * 1) This file is compiled in C++ mode, which results in different mangling
 *    than files compiled in OpenCL mode
 * 2) some functions have the same name in HIP as in OpenCL but different
 *    signature
 * 3) some OpenCL functions (e.g. geometric) take vector arguments
 *    but HIP/CUDA do not have vectors.
 *
 * the counterpart to this file, compiled in OpenCL mode, is mathlib.cl
 */

#if defined(__HIP_DEVICE_COMPILE__)
#define EXPORT static inline __device__
#define OVLD __attribute__((overloadable)) __device__
#define NON_OVLD __device__
#define GEN_NAME(N) opencl_##N
#define GEN_NAME2(N, S) opencl__##N##_##S
#else
#define EXPORT extern __device__
#define NON_OVLD
#define OVLD
#define GEN_NAME(N) N
#define GEN_NAME2(N, S) N
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#if defined(__HIP_DEVICE_COMPILE__)
typedef _Float16 api_half;
typedef _Float16 api_half2 __attribute__((ext_vector_type(2)));
#else
typedef short api_half;
typedef short api_half2 __attribute__((ext_vector_type(2)));
#endif

#if defined(__HIP_DEVICE_COMPILE__)

#define DEFOPENCL1F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float f);                                  \
  double NON_OVLD GEN_NAME2(NAME, d)(double f);                                \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half f);                            \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 f);                         \
  }                                                                            \
  EXPORT float NAME##f(float x) { return GEN_NAME2(NAME, f)(x); }              \
  EXPORT double NAME(double x) { return GEN_NAME2(NAME, d)(x); }               \
  EXPORT api_half NAME##_h(api_half x) { return GEN_NAME2(NAME, h)(x); }       \
  EXPORT api_half2 NAME##_2h(api_half2 x) { return GEN_NAME2(NAME, h2)(x); }

#define DEFOPENCL2F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float x, float y);                         \
  double NON_OVLD GEN_NAME2(NAME, d)(double x, double y);                      \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half x, api_half y);                \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 x, api_half2 y);            \
  }                                                                            \
  EXPORT float NAME##f(float x, float y) { return GEN_NAME2(NAME, f)(x, y); }  \
  EXPORT double NAME(double x, double y) { return GEN_NAME2(NAME, d)(x, y); }  \
  EXPORT api_half NAME##_h(api_half x, api_half y) {                           \
    return GEN_NAME2(NAME, h)(x, y);                                           \
  }                                                                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y) {                       \
    return GEN_NAME2(NAME, h2)(x, y);                                          \
  }

#define DEFOPENCL3F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float x, float y, float z);                \
  double NON_OVLD GEN_NAME2(NAME, d)(double x, double y, double z);            \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half x, api_half y, api_half z);    \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 x, api_half2 y,             \
                                         api_half2 z);                         \
  }                                                                            \
  EXPORT float NAME##f(float x, float y, float z) {                            \
    return GEN_NAME2(NAME, f)(x, y, z);                                        \
  }                                                                            \
  EXPORT double NAME(double x, double y, double z) {                           \
    return GEN_NAME2(NAME, d)(x, y, z);                                        \
  }                                                                            \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z) {               \
    return GEN_NAME2(NAME, h)(x, y, z);                                        \
  }                                                                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z) {          \
    return GEN_NAME2(NAME, h2)(x, y, z);                                       \
  }

#define DEFOPENCL4F(NAME)                                                      \
  extern "C" {                                                                 \
  float NON_OVLD GEN_NAME2(NAME, f)(float x, float y, float z, float w);       \
  double NON_OVLD GEN_NAME2(NAME, d)(double x, double y, double z, double w);  \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half x, api_half y, api_half z,     \
                                       api_half w);                            \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 x, api_half2 y,             \
                                         api_half2 z, api_half2 w);            \
  }                                                                            \
  EXPORT float NAME##f(float x, float y, float z, float w) {                   \
    return GEN_NAME2(NAME, f)(x, y, z, w);                                     \
  }                                                                            \
  EXPORT double NAME(double x, double y, double z, double w) {                 \
    return GEN_NAME2(NAME, d)(x, y, z, w);                                     \
  }                                                                            \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z, api_half w) {   \
    return GEN_NAME2(NAME, h)(x, y, z, w);                                     \
  }                                                                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z,            \
                             api_half2 w) {                                    \
    return GEN_NAME2(NAME, h2)(x, y, z, w);                                    \
  }

#define DEFOPENCL1B(NAME)                                                      \
  extern "C" {                                                                 \
  int NON_OVLD GEN_NAME2(NAME, f)(float f);                                    \
  long NON_OVLD GEN_NAME2(NAME, d)(double f);                                  \
  api_half NON_OVLD GEN_NAME2(NAME, h)(api_half f);                            \
  api_half2 NON_OVLD GEN_NAME2(NAME, h2)(api_half2 f);                         \
  }                                                                            \
  EXPORT bool NAME##f(float x) { return (bool)GEN_NAME2(NAME, f)(x); }         \
  EXPORT bool NAME(double x) { return (bool)GEN_NAME2(NAME, d)(x); }           \
  EXPORT api_half NAME##_h(api_half x) { return GEN_NAME2(NAME, h)(x); }       \
  EXPORT api_half2 NAME##_2h(api_half2 x) { return GEN_NAME2(NAME, h2)(x); }

#define DEFOPENCL1INT(NAME)                                                    \
  int NON_OVLD GEN_NAME2(NAME, f)(float f);                                    \
  int NON_OVLD GEN_NAME2(NAME, d)(double f);                                   \
  int NON_OVLD GEN_NAME2(NAME, h)(api_half f);                                 \
  EXPORT int NAME##f(float x) { return GEN_NAME2(NAME, f)(x); }                \
  EXPORT int NAME(double x) { return GEN_NAME2(NAME, d)(x); }                  \
  EXPORT int NAME##_h(api_half x) { return GEN_NAME2(NAME, h)(x); }

#define DEFOPENCL1LL(NAME)                                                     \
  int64_t NON_OVLD GEN_NAME2(LL##NAME, f)(float f);                            \
  int64_t NON_OVLD GEN_NAME2(LL##NAME, d)(double f);                           \
  EXPORT long int l##NAME##f(float x) {                                        \
    return (long int)GEN_NAME2(LL##NAME, f)(x);                                \
  }                                                                            \
  EXPORT long int l##NAME(double x) {                                          \
    return (long int)GEN_NAME2(LL##NAME, d)(x);                                \
  }                                                                            \
  EXPORT long long int ll##NAME##f(float x) {                                  \
    return (long long int)GEN_NAME2(LL##NAME, f)(x);                           \
  }                                                                            \
  EXPORT long long int ll##NAME(double x) {                                    \
    return (long long int)GEN_NAME2(LL##NAME, d)(x);                           \
  }

#define DEFOPENCL1F_NATIVE(NAME)                                               \
  float NON_OVLD GEN_NAME2(NAME##_native, f)(float f);                         \
  EXPORT float __##NAME##f(float x) { return GEN_NAME2(NAME##_native, f)(x); }

#define FAKE_ROUNDINGS2(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x, float y) { return CODE; }               \
  EXPORT float __f##NAME##_rn(float x, float y) { return CODE; }               \
  EXPORT float __f##NAME##_ru(float x, float y) { return CODE; }               \
  EXPORT float __f##NAME##_rz(float x, float y) { return CODE; }               \
  EXPORT double __d##NAME##_rd(double x, double y) { return CODE; }            \
  EXPORT double __d##NAME##_rn(double x, double y) { return CODE; }            \
  EXPORT double __d##NAME##_ru(double x, double y) { return CODE; }            \
  EXPORT double __d##NAME##_rz(double x, double y) { return CODE; }

#define FAKE_ROUNDINGS1(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x) { return CODE; }                        \
  EXPORT float __f##NAME##_rn(float x) { return CODE; }                        \
  EXPORT float __f##NAME##_ru(float x) { return CODE; }                        \
  EXPORT float __f##NAME##_rz(float x) { return CODE; }                        \
  EXPORT double __d##NAME##_rd(double x) { return CODE; }                      \
  EXPORT double __d##NAME##_rn(double x) { return CODE; }                      \
  EXPORT double __d##NAME##_ru(double x) { return CODE; }                      \
  EXPORT double __d##NAME##_rz(double x) { return CODE; }

#define FAKE_ROUNDINGS3(NAME, CODE)                                            \
  EXPORT float __##NAME##f_rd(float x, float y, float z) { return CODE; }      \
  EXPORT float __##NAME##f_rn(float x, float y, float z) { return CODE; }      \
  EXPORT float __##NAME##f_ru(float x, float y, float z) { return CODE; }      \
  EXPORT float __##NAME##f_rz(float x, float y, float z) { return CODE; }      \
  EXPORT double __##NAME##_rd(double x, double y, double z) { return CODE; }   \
  EXPORT double __##NAME##_rn(double x, double y, double z) { return CODE; }   \
  EXPORT double __##NAME##_ru(double x, double y, double z) { return CODE; }   \
  EXPORT double __##NAME##_rz(double x, double y, double z) { return CODE; }

#else

#define DEFOPENCL1F(NAME)                                                      \
  EXPORT float NAME##f(float x);                                               \
  EXPORT double NAME(double x);                                                \
  EXPORT api_half NAME##_h(api_half x);                                        \
  EXPORT api_half2 NAME##_2h(api_half2 x);

#define DEFOPENCL2F(NAME)                                                      \
  EXPORT float NAME##f(float x, float y);                                      \
  EXPORT double NAME(double x, double y);                                      \
  EXPORT api_half NAME##_h(api_half x, api_half y);                            \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y);

#define DEFOPENCL3F(NAME)                                                      \
  EXPORT float NAME##f(float x, float y, float z);                             \
  EXPORT double NAME(double x, double y, double z);                            \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z);                \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z);

#define DEFOPENCL4F(NAME)                                                      \
  EXPORT float NAME##f(float x, float y, float z, float w);                    \
  EXPORT double NAME(double x, double y, double z, double w);                  \
  EXPORT api_half NAME##_h(api_half x, api_half y, api_half z, api_half w);    \
  EXPORT api_half2 NAME##_2h(api_half2 x, api_half2 y, api_half2 z,            \
                             api_half2 w);

#define DEFOPENCL1B(NAME)                                                      \
  EXPORT bool NAME##f(float x);                                                \
  EXPORT bool NAME(double x);                                                  \
  EXPORT api_half NAME##_h(api_half x);                                        \
  EXPORT api_half2 NAME##_2h(api_half2 x);

#define DEFOPENCL1INT(NAME)                                                    \
  EXPORT int NAME##f(float x);                                                 \
  EXPORT int NAME(double x);                                                   \
  EXPORT int NAME##_h(api_half x);

#define DEFOPENCL1LL(NAME)                                                     \
  EXPORT long int l##NAME##f(float x);                                         \
  EXPORT long int l##NAME(double x);                                           \
  EXPORT long long int ll##NAME##f(float x);                                   \
  EXPORT long long int ll##NAME(double x);

#define DEFOPENCL1F_NATIVE(NAME) EXPORT float __##NAME##f(float x);

#define FAKE_ROUNDINGS2(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x, float y);                               \
  EXPORT float __f##NAME##_rn(float x, float y);                               \
  EXPORT float __f##NAME##_ru(float x, float y);                               \
  EXPORT float __f##NAME##_rz(float x, float y);                               \
  EXPORT double __d##NAME##_rd(double x, double y);                            \
  EXPORT double __d##NAME##_rn(double x, double y);                            \
  EXPORT double __d##NAME##_ru(double x, double y);                            \
  EXPORT double __d##NAME##_rz(double x, double y);

#define FAKE_ROUNDINGS1(NAME, CODE)                                            \
  EXPORT float __f##NAME##_rd(float x);                                        \
  EXPORT float __f##NAME##_rn(float x);                                        \
  EXPORT float __f##NAME##_ru(float x);                                        \
  EXPORT float __f##NAME##_rz(float x);                                        \
  EXPORT double __d##NAME##_rd(double x);                                      \
  EXPORT double __d##NAME##_rn(double x);                                      \
  EXPORT double __d##NAME##_ru(double x);                                      \
  EXPORT double __d##NAME##_rz(double x);

#define FAKE_ROUNDINGS3(NAME, CODE)                                            \
  EXPORT float __##NAME##f_rd(float x, float y, float z);                      \
  EXPORT float __##NAME##f_rn(float x, float y, float z);                      \
  EXPORT float __##NAME##f_ru(float x, float y, float z);                      \
  EXPORT float __##NAME##f_rz(float x, float y, float z);                      \
  EXPORT double __##NAME##_rd(double x, double y, double z);                   \
  EXPORT double __##NAME##_rn(double x, double y, double z);                   \
  EXPORT double __##NAME##_ru(double x, double y, double z);                   \
  EXPORT double __##NAME##_rz(double x, double y, double z);

#endif

DEFOPENCL1F(acos)
DEFOPENCL1F(asin)
DEFOPENCL1F(acosh)
DEFOPENCL1F(asinh)
DEFOPENCL1F(atan)
DEFOPENCL2F(atan2)
DEFOPENCL1F(atanh)
DEFOPENCL1F(cbrt)
DEFOPENCL1F(ceil)

DEFOPENCL2F(copysign)

DEFOPENCL1F(cos)
DEFOPENCL1F(cosh)
DEFOPENCL1F(cospi)

DEFOPENCL1F(cyl_bessel_i1)
DEFOPENCL1F(cyl_bessel_i0)

DEFOPENCL1F(erfc)
DEFOPENCL1F(erf)
DEFOPENCL1F(erfcinv)
DEFOPENCL1F(erfcx)
DEFOPENCL1F(erfinv)

DEFOPENCL1F(exp10)
DEFOPENCL1F(exp2)
DEFOPENCL1F(exp)
DEFOPENCL1F(expm1)
DEFOPENCL1F(fabs)
DEFOPENCL2F(fdim)
DEFOPENCL1F(floor)

EXPORT float fdividef(float x, float y) { return x / y; }
EXPORT double fdivide(double x, double y) { return x / y; }

DEFOPENCL3F(fma)

DEFOPENCL2F(fmax)
DEFOPENCL2F(fmin)
DEFOPENCL2F(fmod)

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
float NON_OVLD GEN_NAME2(frexp, f)(float f, int *i);
double NON_OVLD GEN_NAME2(frexp, d)(double f, int *i);
}
EXPORT float frexpf(float f, int *i) { return GEN_NAME2(frexp, f)(f, i); }
EXPORT double frexp(double f, int *i) { return GEN_NAME2(frexp, d)(f, i); }
#else
EXPORT float frexpf(float f, int *i);
EXPORT double frexp(double f, int *i);
#endif

DEFOPENCL2F(hypot)
DEFOPENCL1INT(ilogb)

DEFOPENCL1B(isfinite)
DEFOPENCL1B(isinf)
DEFOPENCL1B(isnan)

DEFOPENCL1F(j0)
DEFOPENCL1F(j1)

EXPORT float jnf(int n, float x) { // TODO: we could use Ahmes multiplication
                                   // and the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case.
  if (n == 0)
    return j0f(x);
  if (n == 1)
    return j1f(x);

  float x0 = j0f(x);
  float x1 = j1f(x);
  for (int i = 1; i < n; ++i) {
    float x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}
EXPORT double jn(int n, double x) { // TODO: we could use Ahmes multiplication
                                    // and the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case. Placeholder until OCML adds
  //       support.
  if (n == 0)
    return j0(x);
  if (n == 1)
    return j1(x);

  double x0 = j0(x);
  double x1 = j1(x);
  for (int i = 1; i < n; ++i) {
    double x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD float GEN_NAME2(ldexp, f)(float f, int k);
NON_OVLD double GEN_NAME2(ldexp, d)(double f, int k);
}
EXPORT float ldexpf(float x, int k) { return GEN_NAME2(ldexp, f)(x, k); }
EXPORT double ldexp(double x, int k) { return GEN_NAME2(ldexp, d)(x, k); }
#else
EXPORT float ldexpf(float x, int k);
EXPORT double ldexp(double x, int k);
#endif

DEFOPENCL1F(lgamma)

DEFOPENCL1LL(rint)
DEFOPENCL1F(rint)
DEFOPENCL1LL(round)

DEFOPENCL1F(log10)
DEFOPENCL1F(log1p)
DEFOPENCL1F(log2)
DEFOPENCL1F(logb)
DEFOPENCL1F(log)

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD float GEN_NAME2(modf, f)(float f, float *i);
NON_OVLD double GEN_NAME2(modf, d)(double f, double *i);
}
EXPORT float modff(float f, float *i) { return GEN_NAME2(modf, f)(f, i); }
EXPORT double modf(double f, double *i) { return GEN_NAME2(modf, d)(f, i); }
#else
EXPORT float modff(float f, float *i);
EXPORT double modf(double f, double *i);
#endif

DEFOPENCL1F(nearbyint)
DEFOPENCL2F(nextafter)

DEFOPENCL3F(norm3d)
DEFOPENCL4F(norm4d)
DEFOPENCL1F(normcdf)
DEFOPENCL1F(normcdfinv)

DEFOPENCL2F(pow)
DEFOPENCL2F(remainder)
DEFOPENCL1F(rcbrt)

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD float GEN_NAME2(remquo, f)(float x, float y, int *quo);
NON_OVLD double GEN_NAME2(remquo, d)(double x, double y, int *quo);
}
EXPORT float remquof(float x, float y, int *quo) {
  return GEN_NAME2(remquo, f)(x, y, quo);
}
EXPORT double remquo(double x, double y, int *quo) {
  return GEN_NAME2(remquo, d)(x, y, quo);
}
#else
EXPORT float remquof(float x, float y, int *quo);
EXPORT double remquo(double x, double y, int *quo);
#endif

DEFOPENCL2F(rhypot)

DEFOPENCL3F(rnorm3d)
DEFOPENCL4F(rnorm4d)

DEFOPENCL1F(round)
DEFOPENCL1F(rsqrt)

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
float NON_OVLD GEN_NAME2(scalbn, f)(float f, int k);
double NON_OVLD GEN_NAME2(scalbn, d)(double f, int k);
float NON_OVLD GEN_NAME2(scalb, f)(float x, float y);
double NON_OVLD GEN_NAME2(scalb, d)(double x, double y);
}

EXPORT float scalblnf(float x, long int n) {
  return (n < INT_MAX) ? GEN_NAME2(scalbn, f)(x, (int)n)
                       : GEN_NAME2(scalb, f)(x, (float)n);
}
EXPORT float scalbnf(float x, int n) { return GEN_NAME2(scalbn, f)(x, n); }
EXPORT double scalbln(double x, long int n) {
  return (n < INT_MAX) ? GEN_NAME2(scalbn, d)(x, (int)n)
                       : GEN_NAME2(scalb, d)(x, (double)n);
}
EXPORT double scalbn(double x, int n) { return GEN_NAME2(scalbn, d)(x, n); }
#else
EXPORT float scalblnf(float x, long int n);
EXPORT float scalbnf(float x, int n);
EXPORT double scalbln(double x, long int n);
EXPORT double scalbn(double x, int n);
#endif

DEFOPENCL1B(signbit)

DEFOPENCL1F(sin)
DEFOPENCL1F(sinh)
DEFOPENCL1F(sinpi)
DEFOPENCL1F(sqrt)
DEFOPENCL1F(tan)
DEFOPENCL1F(tanh)
DEFOPENCL1F(tgamma)
DEFOPENCL1F(trunc)

#if defined(__HIP_DEVICE_COMPILE__)
// float normf ( int dim, const float *a )
EXPORT
float normf(int dim,
            const float *a) { // TODO: placeholder until OCML adds support.
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME2(sqrt, f)(r);
}

// float rnormf ( int  dim, const float* t )
EXPORT
float rnormf(int dim,
             const float *a) { // TODO: placeholder until OCML adds support.
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME2(sqrt, f)(r);
}

EXPORT
double norm(int dim,
            const double *a) { // TODO: placeholder until OCML adds support.
  double r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME2(sqrt, d)(r);
}

EXPORT
double rnorm(int dim,
             const double *a) { // TODO: placeholder until OCML adds support.
  double r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME2(sqrt, d)(r);
}

// sincos
extern "C" {
NON_OVLD float GEN_NAME2(sincos, f)(float x, float *cos);
NON_OVLD double GEN_NAME2(sincos, d)(double x, double *cos);
}
EXPORT
void sincosf(float x, float *sptr, float *cptr) {
  float tmp;
  *sptr = GEN_NAME2(sincos, f)(x, &tmp);
  *cptr = tmp;
}
EXPORT
void sincos(double x, double *sptr, double *cptr) {
  double tmp;
  *sptr = GEN_NAME2(sincos, d)(x, &tmp);
  *cptr = tmp;
}

// sincospi
EXPORT
void sincospif(float x, float *sptr, float *cptr) {
  *sptr = GEN_NAME2(sinpi, f)(x);
  *cptr = GEN_NAME2(cospi, f)(x);
}

EXPORT
void sincospi(double x, double *sptr, double *cptr) {
  *sptr = GEN_NAME2(sinpi, d)(x);
  *cptr = GEN_NAME2(cospi, d)(x);
}
#else
EXPORT float normf(int dim, const float *a);
EXPORT float rnormf(int dim, const float *a);
EXPORT double norm(int dim, const double *a);
EXPORT double rnorm(int dim, const double *a);
EXPORT void sincosf(float x, float *sptr, float *cptr);
EXPORT void sincos(double x, double *sptr, double *cptr);
EXPORT void sincospif(float x, float *sptr, float *cptr);
EXPORT void sincospi(double x, double *sptr, double *cptr);
#endif

DEFOPENCL1F(y0)
DEFOPENCL1F(y1)
EXPORT float ynf(int n, float x) { // TODO: we could use Ahmes multiplication
                                   // and the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case. Placeholder until OCML adds
  //       support.
  if (n == 0)
    return y0f(x);
  if (n == 1)
    return y1f(x);

  float x0 = y0f(x);
  float x1 = y1f(x);
  for (int i = 1; i < n; ++i) {
    float x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}
EXPORT double yn(int n, double x) { // TODO: we could use Ahmes multiplication
                                    // and the Miller & Brown algorithm
  //       for linear recurrences to get O(log n) steps, but it's unclear if
  //       it'd be beneficial in this case. Placeholder until OCML adds
  //       support.
  if (n == 0)
    return j0(x);
  if (n == 1)
    return j1(x);

  double x0 = j0(x);
  double x1 = j1(x);
  for (int i = 1; i < n; ++i) {
    double x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

/**********************************************************************/

FAKE_ROUNDINGS2(add, x + y)
FAKE_ROUNDINGS2(sub, x - y)
FAKE_ROUNDINGS2(div, x / y)
FAKE_ROUNDINGS2(mul, x *y)

FAKE_ROUNDINGS1(rcp, (1.0f / x))
FAKE_ROUNDINGS2(sqrt, GEN_NAME2(sqrt, f)(x))

FAKE_ROUNDINGS3(fma, GEN_NAME2(fma, f)(x, y, z))

DEFOPENCL1F_NATIVE(cos)
DEFOPENCL1F_NATIVE(sin)
DEFOPENCL1F_NATIVE(tan)

DEFOPENCL1F_NATIVE(exp10)
DEFOPENCL1F_NATIVE(exp)

DEFOPENCL1F_NATIVE(log10)
DEFOPENCL1F_NATIVE(log2)
DEFOPENCL1F_NATIVE(log)

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
float NON_OVLD GEN_NAME2(powr_native, f)(float x, float y);
}

EXPORT float __powf(float x, float y) {
  return GEN_NAME2(powr_native, f)(x, y);
}

EXPORT float __saturatef(float x) {
  return (x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x);
}

EXPORT void __sincosf(float x, float *sptr, float *cptr) {
  *sptr = GEN_NAME2(sin_native, f)(x);
  *cptr = GEN_NAME2(cos_native, f)(x);
}

/**********************************************************************/

extern "C" {
NON_OVLD void GEN_NAME(local_barrier)();
}
EXPORT void __syncthreads() { GEN_NAME(local_barrier)(); }

/**********************************************************************/

#else
EXPORT float __powf(float x, float y);
EXPORT float __saturatef(float x);
EXPORT void __sincosf(float x, float *sptr, float *cptr);
EXPORT void __syncthreads();
#endif

// NAN/NANF

EXPORT
uint64_t __make_mantissa_base8(const char *tagp) {
  uint64_t r = 0;
  while (tagp) {
    char tmp = *tagp;

    if (tmp >= '0' && tmp <= '7')
      r = (r * 8u) + tmp - '0';
    else
      return 0;

    ++tagp;
  }

  return r;
}

EXPORT
uint64_t __make_mantissa_base10(const char *tagp) {
  uint64_t r = 0;
  while (tagp) {
    char tmp = *tagp;

    if (tmp >= '0' && tmp <= '9')
      r = (r * 10u) + tmp - '0';
    else
      return 0;

    ++tagp;
  }

  return r;
}

EXPORT
uint64_t __make_mantissa_base16(const char *tagp) {
  uint64_t r = 0;
  while (tagp) {
    char tmp = *tagp;

    if (tmp >= '0' && tmp <= '9')
      r = (r * 16u) + tmp - '0';
    else if (tmp >= 'a' && tmp <= 'f')
      r = (r * 16u) + tmp - 'a' + 10;
    else if (tmp >= 'A' && tmp <= 'F')
      r = (r * 16u) + tmp - 'A' + 10;
    else
      return 0;

    ++tagp;
  }

  return r;
}

EXPORT
uint64_t __make_mantissa(const char *tagp) {
  if (!tagp)
    return 0u;

  if (*tagp == '0') {
    ++tagp;

    if (*tagp == 'x' || *tagp == 'X')
      return __make_mantissa_base16(tagp);
    else
      return __make_mantissa_base8(tagp);
  }

  return __make_mantissa_base10(tagp);
}

EXPORT
float nanf(const char *tagp) {
  union {
    float val;
    struct ieee_float {
      uint32_t mantissa : 22;
      uint32_t quiet : 1;
      uint32_t exponent : 8;
      uint32_t sign : 1;
    } bits;

    static_assert(sizeof(float) == sizeof(ieee_float), "");
  } tmp;

  tmp.bits.sign = 0u;
  tmp.bits.exponent = ~0u;
  tmp.bits.quiet = 1u;
  tmp.bits.mantissa = __make_mantissa(tagp);

  return tmp.val;
}

EXPORT
double nan(const char *tagp) {
  union {
    double val;
    struct ieee_double {
      uint64_t mantissa : 51;
      uint32_t quiet : 1;
      uint32_t exponent : 11;
      uint32_t sign : 1;
    } bits;
    static_assert(sizeof(double) == sizeof(ieee_double), "");
  } tmp;

  tmp.bits.sign = 0u;
  tmp.bits.exponent = ~0u;
  tmp.bits.quiet = 1u;
  tmp.bits.mantissa = __make_mantissa(tagp);

  return tmp.val;
}

/**********************************************************************/

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

// BEGIN INTEGER
EXPORT int abs(int x) {
  int sgn = x >> (sizeof(int) * CHAR_BIT - 1);
  return (x ^ sgn) - sgn;
}
EXPORT long labs(long x) {
  long sgn = x >> (sizeof(long) * CHAR_BIT - 1);
  return (x ^ sgn) - sgn;
}
EXPORT long long llabs(long long x) {
  long long sgn = x >> (sizeof(long long) * CHAR_BIT - 1);
  return (x ^ sgn) - sgn;
}

#if defined(__cplusplus)
EXPORT long abs(long x) { return labs(x); }
EXPORT long long abs(long long x) { return llabs(x); }
#endif
// END INTEGER

#if defined(__cplusplus)
EXPORT float fma(float x, float y, float z) { return fmaf(x, y, z); }
#endif

EXPORT float max(float x, float y) { return fmaxf(x, y); }

EXPORT double max(double x, double y) { return fmax(x, y); }

EXPORT float min(float x, float y) { return fminf(x, y); }

EXPORT double min(double x, double y) { return fmin(x, y); }

/**********************************************************************/

#if defined(__HIP_DEVICE_COMPILE__)

#define DEFOPENCL_ATOMIC2(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  NON_OVLD int GEN_NAME2(atomic_##CLNAME, i)(volatile int *address, int i);    \
  NON_OVLD unsigned int GEN_NAME2(atomic_##CLNAME,                             \
                                  u)(volatile unsigned int *address,           \
                                     unsigned int ui);                         \
  NON_OVLD unsigned long long                                                  \
      GEN_NAME2(atomic_##CLNAME, l)(volatile unsigned long long *address,      \
                                    unsigned long long ull);                   \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(volatile int *address, int val) {            \
    return GEN_NAME2(atomic_##CLNAME, i)(address, val);                        \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(volatile unsigned int *address,     \
                                           unsigned int val) {                 \
    return GEN_NAME2(atomic_##CLNAME, u)(address, val);                        \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      volatile unsigned long long *address, unsigned long long val) {          \
    return GEN_NAME2(atomic_##CLNAME, l)(address, val);                        \
  }

#define DEFOPENCL_ATOMIC1(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  NON_OVLD int GEN_NAME2(atomic_##CLNAME, i)(volatile int *address);           \
  NON_OVLD unsigned int GEN_NAME2(atomic_##CLNAME,                             \
                                  u)(volatile unsigned int *address);          \
  NON_OVLD unsigned long long                                                  \
      GEN_NAME2(atomic_##CLNAME, l)(volatile unsigned long long *address);     \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(volatile int *address) {                     \
    return GEN_NAME2(atomic_##CLNAME, i)(address);                             \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(volatile unsigned int *address) {   \
    return GEN_NAME2(atomic_##CLNAME, u)(address);                             \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      volatile unsigned long long *address) {                                  \
    return GEN_NAME2(atomic_##CLNAME, l)(address);                             \
  }

#define DEFOPENCL_ATOMIC3(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  NON_OVLD int GEN_NAME2(atomic_##CLNAME, i)(volatile int *address, int cmp,   \
                                             int val);                         \
  NON_OVLD unsigned int GEN_NAME2(atomic_##CLNAME,                             \
                                  u)(volatile unsigned int *address,           \
                                     unsigned int cmp, unsigned int val);      \
  NON_OVLD unsigned long long                                                  \
      GEN_NAME2(atomic_##CLNAME, l)(volatile unsigned long long *address,      \
                                    unsigned long long cmp,                    \
                                    unsigned long long val);                   \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(volatile int *address, int cmp, int val) {   \
    return GEN_NAME2(atomic_##CLNAME, i)(address, cmp, val);                   \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(                                    \
      volatile unsigned int *address, unsigned int cmp, unsigned int val) {    \
    return GEN_NAME2(atomic_##CLNAME, u)(address, cmp, val);                   \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      volatile unsigned long long *address, unsigned long long cmp,            \
      unsigned long long val) {                                                \
    return GEN_NAME2(atomic_##CLNAME, l)(address, cmp, val);                   \
  }

#else

#define DEFOPENCL_ATOMIC2(HIPNAME, CLNAME)                                     \
  EXPORT OVLD int atomic##HIPNAME(volatile int *address, int val);             \
  EXPORT OVLD unsigned int atomic##HIPNAME(volatile unsigned int *address,     \
                                           unsigned int val);                  \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      volatile unsigned long long *address, unsigned long long val);

#define DEFOPENCL_ATOMIC1(HIPNAME, CLNAME)                                     \
  EXPORT OVLD int atomic##HIPNAME(volatile int *address);                      \
  EXPORT OVLD unsigned int atomic##HIPNAME(volatile unsigned int *address);    \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      volatile unsigned long long *address);

#define DEFOPENCL_ATOMIC3(HIPNAME, CLNAME)                                     \
  EXPORT OVLD int atomic##HIPNAME(volatile int *address, int cmp, int val);    \
  EXPORT OVLD unsigned int atomic##HIPNAME(                                    \
      volatile unsigned int *address, unsigned int cmp, unsigned int val);     \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      volatile unsigned long long *address, unsigned long long cmp,            \
      unsigned long long val);

#endif

DEFOPENCL_ATOMIC2(Add, add);
DEFOPENCL_ATOMIC2(Sub, sub);
DEFOPENCL_ATOMIC2(Exch, xchg);
DEFOPENCL_ATOMIC2(Min, min);
DEFOPENCL_ATOMIC2(Max, max);
DEFOPENCL_ATOMIC2(And, and);
DEFOPENCL_ATOMIC2(Or, or);
DEFOPENCL_ATOMIC2(Xor, xor);

DEFOPENCL_ATOMIC1(Inc, inc);
DEFOPENCL_ATOMIC1(Dec, dec);

DEFOPENCL_ATOMIC3(CAS, cmpxchg)

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD float GEN_NAME2(atomic_add, f)(volatile float *address, float val);
NON_OVLD double GEN_NAME2(atomic_add, d)(volatile double *address, double val);
NON_OVLD float GEN_NAME2(atomic_exch, f)(volatile float *address, float val);
}
EXPORT float atomicAdd(float *address, float val) {
  return GEN_NAME2(atomic_add, f)(address, val);
}
EXPORT double atomicAdd(double *address, double val) {
  return GEN_NAME2(atomic_add, d)(address, val);
}
EXPORT float atomicExch(float *address, float val) {
  return GEN_NAME2(atomic_exch, f)(address, val);
}
#else
EXPORT float atomicAdd(float *address, float val);
EXPORT double atomicAdd(double *address, double val);
EXPORT float atomicExch(float *address, float val);
#endif

/**********************************************************************/

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD int GEN_NAME2(shfl, i)(int var, int srcLane);
NON_OVLD float GEN_NAME2(shfl, f)(float var, int srcLane);
NON_OVLD int GEN_NAME2(shfl_xor, i)(int var, int laneMask);
NON_OVLD float GEN_NAME2(shfl_xor, f)(float var, int laneMask);
NON_OVLD int GEN_NAME2(shfl_up, i)(int var, unsigned int delta);
NON_OVLD float GEN_NAME2(shfl_up, f)(float var, unsigned int delta);
NON_OVLD int GEN_NAME2(shfl_down, i)(int var, unsigned int delta);
NON_OVLD float GEN_NAME2(shfl_down, f)(float var, unsigned int delta);
NON_OVLD int GEN_NAME(group_all)(int pred);
NON_OVLD int GEN_NAME(group_any)(int pred);
}

EXPORT OVLD int __shfl(int var, int srcLane) {
  return GEN_NAME2(shfl, i)(var, srcLane);
};
EXPORT OVLD float __shfl(float var, int srcLane) {
  return GEN_NAME2(shfl, f)(var, srcLane);
};
EXPORT OVLD int __shfl_xor(int var, int laneMask) {
  return GEN_NAME2(shfl_xor, i)(var, laneMask);
};
EXPORT OVLD float __shfl_xor(float var, int laneMask) {
  return GEN_NAME2(shfl_xor, f)(var, laneMask);
};
EXPORT OVLD int __shfl_up(int var, unsigned int delta) {
  return GEN_NAME2(shfl_up, i)(var, delta);
};
EXPORT OVLD float __shfl_up(float var, unsigned int delta) {
  return GEN_NAME2(shfl_up, f)(var, delta);
};
EXPORT OVLD int __shfl_down(int var, unsigned int delta) {
  return GEN_NAME2(shfl_down, i)(var, delta);
};
EXPORT OVLD float __shfl_down(float var, unsigned int delta) {
  return GEN_NAME2(shfl_down, f)(var, delta);
};
EXPORT int __all(int predicate) { return GEN_NAME(group_all)(predicate); };
EXPORT int __any(int predicate) { return GEN_NAME(group_any)(predicate); };
#else

EXPORT OVLD int __shfl(int var, int srcLane);
EXPORT OVLD float __shfl(float var, int srcLane);

EXPORT OVLD int __shfl_xor(int var, int laneMask);
EXPORT OVLD float __shfl_xor(float var, int laneMask);

EXPORT OVLD int __shfl_up(int var, unsigned int delta);
EXPORT OVLD float __shfl_up(float var, unsigned int delta);

EXPORT OVLD int __shfl_down(int var, unsigned int delta);
EXPORT OVLD float __shfl_down(float var, unsigned int delta);

EXPORT int __all(int predicate);
EXPORT int __any(int predicate);

#endif
