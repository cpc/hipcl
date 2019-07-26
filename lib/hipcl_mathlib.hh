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

#define OVLD __attribute__((overloadable)) __device__

#define EXPORT static inline __device__

#define GEN_NAME(N) opencl_##N

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#define DEFOPENCL1F(NAME)                                                      \
  float OVLD GEN_NAME(NAME)(float f);                                          \
  double OVLD GEN_NAME(NAME)(double f);                                        \
  EXPORT float NAME##f(float x) { return GEN_NAME(NAME)(x); }                  \
  EXPORT double NAME(double x) { return GEN_NAME(NAME)(x); }

#define DEFOPENCL2F(NAME)                                                      \
  float OVLD GEN_NAME(NAME)(float x, float y);                                 \
  double OVLD GEN_NAME(NAME)(double x, double y);                              \
  EXPORT float NAME##f(float x, float y) { return GEN_NAME(NAME)(x, y); }      \
  EXPORT double NAME(double x, double y) { return GEN_NAME(NAME)(x, y); }

#define DEFOPENCL3F(NAME)                                                      \
  float OVLD GEN_NAME(NAME)(float x, float y, float z);                        \
  double OVLD GEN_NAME(NAME)(double x, double y, double z);                    \
  EXPORT float NAME##f(float x, float y, float z) {                            \
    return GEN_NAME(NAME)(x, y, z);                                            \
  }                                                                            \
  EXPORT double NAME(double x, double y, double z) {                           \
    return GEN_NAME(NAME)(x, y, z);                                            \
  }

#define DEFOPENCL4F(NAME)                                                      \
  float OVLD GEN_NAME(NAME)(float x, float y, float z, float w);               \
  double OVLD GEN_NAME(NAME)(double x, double y, double z, double w);          \
  EXPORT float NAME##f(float x, float y, float z, float w) {                   \
    return GEN_NAME(NAME)(x, y, z, w);                                         \
  }                                                                            \
  EXPORT double NAME(double x, double y, double z, double w) {                 \
    return GEN_NAME(NAME)(x, y, z, w);                                         \
  }

#define DEFOPENCL1B(NAME)                                                      \
  int OVLD GEN_NAME(NAME)(float f);                                            \
  long OVLD GEN_NAME(NAME)(double f);                                          \
  EXPORT bool NAME##f(float x) { return (bool)GEN_NAME(NAME)(x); }             \
  EXPORT bool NAME(double x) { return (bool)GEN_NAME(NAME)(x); }

#define DEFOPENCL1INT(NAME)                                                    \
  int OVLD GEN_NAME(NAME)(float f);                                            \
  int OVLD GEN_NAME(NAME)(double f);                                           \
  EXPORT int NAME##f(float x) { return GEN_NAME(NAME)(x); }                    \
  EXPORT int NAME(double x) { return GEN_NAME(NAME)(x); }

#define DEFOPENCL1LL(NAME)                                                     \
  int64_t OVLD GEN_NAME(LL##NAME)(float f);                                    \
  int64_t OVLD GEN_NAME(LL##NAME)(double f);                                   \
  EXPORT long int l##NAME##f(float x) {                                        \
    return (long int)GEN_NAME(LL##NAME)(x);                                    \
  }                                                                            \
  EXPORT long int l##NAME(double x) {                                          \
    return (long int)GEN_NAME(LL##NAME)(x);                                    \
  }                                                                            \
  EXPORT long long int ll##NAME##f(float x) {                                  \
    return (long long int)GEN_NAME(LL##NAME)(x);                               \
  }                                                                            \
  EXPORT long long int ll##NAME(double x) {                                    \
    return (long long int)GEN_NAME(LL##NAME)(x);                               \
  }

#define DEFOPENCL1F_NATIVE(NAME)                                               \
  float OVLD GEN_NAME(NAME##_native)(float f);                                 \
  EXPORT float __##NAME##f(float x) { return GEN_NAME(NAME##_native)(x); }

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

float OVLD GEN_NAME(frexp)(float f, int *i);
double OVLD GEN_NAME(frexp)(double f, int *i);
EXPORT float frexpf(float f, int *i) { return GEN_NAME(frexp)(f, i); }
EXPORT double frexp(double f, int *i) { return GEN_NAME(frexp)(f, i); }

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

float OVLD GEN_NAME(ldexp)(float f, int k);
double OVLD GEN_NAME(ldexp)(double f, int k);
EXPORT float ldexpf(float x, int k) { return GEN_NAME(ldexp)(x, k); }
EXPORT double ldexp(double x, int k) { return GEN_NAME(ldexp)(x, k); }

DEFOPENCL1F(lgamma)

DEFOPENCL1LL(rint)
DEFOPENCL1F(rint)
DEFOPENCL1LL(round)

DEFOPENCL1F(log10)
DEFOPENCL1F(log1p)
DEFOPENCL1F(log2)
DEFOPENCL1F(logb)
DEFOPENCL1F(log)

float OVLD GEN_NAME(modf)(float f, float *i);
double OVLD GEN_NAME(modf)(double f, double *i);
EXPORT float modff(float f, float *i) { return GEN_NAME(modf)(f, i); }
EXPORT double modf(double f, double *i) { return GEN_NAME(modf)(f, i); }

DEFOPENCL1F(nearbyint)
DEFOPENCL2F(nextafter)

DEFOPENCL3F(norm3d)
DEFOPENCL4F(norm4d)
DEFOPENCL1F(normcdf)
DEFOPENCL1F(normcdfinv)

DEFOPENCL2F(pow)
DEFOPENCL2F(remainder)
DEFOPENCL1F(rcbrt)

float OVLD GEN_NAME(remquo)(float x, float y, int *quo);
double OVLD GEN_NAME(remquo)(double x, double y, int *quo);
EXPORT float remquof(float x, float y, int *quo) {
  return GEN_NAME(remquo)(x, y, quo);
}
EXPORT double remquo(double x, double y, int *quo) {
  return GEN_NAME(remquo)(x, y, quo);
}

DEFOPENCL2F(rhypot)

DEFOPENCL3F(rnorm3d)
DEFOPENCL4F(rnorm4d)

DEFOPENCL1F(round)
DEFOPENCL1F(rsqrt)

float OVLD GEN_NAME(scalbn)(float f, int k);
double OVLD GEN_NAME(scalbn)(double f, int k);
float OVLD GEN_NAME(scalb)(float x, float y);
double OVLD GEN_NAME(scalb)(double x, double y);

EXPORT float scalblnf(float x, long int n) {
  return (n < INT_MAX) ? GEN_NAME(scalbn)(x, (int)n)
                       : GEN_NAME(scalb)(x, (float)n);
}
EXPORT float scalbnf(float x, int n) { return GEN_NAME(scalbn)(x, n); }
EXPORT double scalbln(double x, long int n) {
  return (n < INT_MAX) ? GEN_NAME(scalbn)(x, (int)n)
                       : GEN_NAME(scalb)(x, (double)n);
}
EXPORT double scalbn(double x, int n) { return GEN_NAME(scalbn)(x, n); }

DEFOPENCL1B(signbit)

DEFOPENCL1F(sin)
DEFOPENCL1F(sinh)
DEFOPENCL1F(sinpi)
DEFOPENCL1F(sqrt)
DEFOPENCL1F(tan)
DEFOPENCL1F(tanh)
DEFOPENCL1F(tgamma)
DEFOPENCL1F(trunc)

// float normf ( int dim, const float *a )
EXPORT
float normf(int dim,
            const float *a) { // TODO: placeholder until OCML adds support.
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME(sqrt)(r);
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

  return GEN_NAME(sqrt)(r);
}

EXPORT
double norm(int dim,
            const double *a) { // TODO: placeholder until OCML adds support.
  double r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME(sqrt)(r);
}

EXPORT
double rnorm(int dim,
             const double *a) { // TODO: placeholder until OCML adds support.
  double r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return GEN_NAME(sqrt)(r);
}

// sincos
float OVLD GEN_NAME(sincos)(float x, float *cos);
double OVLD GEN_NAME(sincos)(double x, double *cos);
EXPORT
void sincosf(float x, float *sptr, float *cptr) {
  float tmp;
  *sptr = GEN_NAME(sincos)(x, &tmp);
  *cptr = tmp;
}
EXPORT
void sincos(double x, double *sptr, double *cptr) {
  double tmp;
  *sptr = GEN_NAME(sincos)(x, &tmp);
  *cptr = tmp;
}

// sincospi
EXPORT
void sincospif(float x, float *sptr, float *cptr) {
  *sptr = GEN_NAME(sinpi)(x);
  *cptr = GEN_NAME(cospi)(x);
}

EXPORT
void sincospi(double x, double *sptr, double *cptr) {
  *sptr = GEN_NAME(sinpi)(x);
  *cptr = GEN_NAME(cospi)(x);
}

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
FAKE_ROUNDINGS2(sqrt, GEN_NAME(sqrt)(x))

FAKE_ROUNDINGS3(fma, GEN_NAME(fma)(x, y, z))

DEFOPENCL1F_NATIVE(cos)
DEFOPENCL1F_NATIVE(sin)
DEFOPENCL1F_NATIVE(tan)

DEFOPENCL1F_NATIVE(exp10)
DEFOPENCL1F_NATIVE(exp)

DEFOPENCL1F_NATIVE(log10)
DEFOPENCL1F_NATIVE(log2)
DEFOPENCL1F_NATIVE(log)

float OVLD GEN_NAME(powr_native)(float x, float y);
EXPORT float __powf(float x, float y) { return GEN_NAME(powr_native)(x, y); }

EXPORT float __saturatef(float x) {
  return (x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x);
}

EXPORT void __sincosf(float x, float *sptr, float *cptr) {
  *sptr = GEN_NAME(sin_native)(x);
  *cptr = GEN_NAME(cos_native)(x);
}

/**********************************************************************/

void OVLD GEN_NAME(local_barrier)();

EXPORT
void __syncthreads() { GEN_NAME(local_barrier)(); }

/**********************************************************************/

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
