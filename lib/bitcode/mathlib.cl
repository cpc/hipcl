/*
 * This is counterpart to hipcl_mathlib.hh
 * ATM it can't be used right after compilation because of a problem with mangling.
 *
 * HIP with default AS set to 4 mangles functions with pointer args to:
 *   float @_Z13opencl_sincosfPf(float, float addrspace(4)*)
 * while OpenCL code compiled for SPIR mangles to either
 *   float @_Z6sincosfPU3AS4f(float, float addrspace(4)*)
 * or
 *   float @_Z6sincosfPf(float, float *)
*/

#define CL_NAME_MANGLED_ATOM3(NAME, X, POSTFIX) _Z##X##opencl_##NAME##POSTFIX
#define CL_NAME_MANGLED_ATOM(NAME, S, X, POSTFIX)                              \
  CL_NAME_MANGLED_ATOM3(atomic_##NAME##S, X, POSTFIX)

#define OVLD __attribute__((overloadable))
//#define AI __attribute__((always_inline))
#define EXPORT OVLD

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define DEFAULT_AS
#define PRIVATE_AS __private

#define CL_NAME(N) opencl_ ## N

#define DEFOCML_OPENCL1F(NAME) \
float OVLD NAME(float f); \
double OVLD NAME(double f); \
EXPORT float CL_NAME(NAME)(float x) { return NAME(x); } \
EXPORT double CL_NAME(NAME)(double x) { return NAME(x); }

#define DEFOCML_OPENCL2F(NAME) \
float OVLD NAME(float x, float y); \
double OVLD NAME(double x, double y); \
EXPORT float CL_NAME(NAME)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME(NAME)(double x, double y) { return NAME(x, y); }


#define DEF_OPENCL1F(NAME) \
EXPORT float CL_NAME(NAME)(float x) { return NAME(x); } \
EXPORT double CL_NAME(NAME)(double x) { return NAME(x); }

#define DEF_OPENCL2F(NAME) \
EXPORT float CL_NAME(NAME)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME(NAME)(double x, double y) { return NAME(x, y); }

#define DEF_OPENCL3F(NAME) \
EXPORT float CL_NAME(NAME)(float x, float y, float z) { return NAME(x, y, z); } \
EXPORT double CL_NAME(NAME)(double x, double y, double z) { return NAME(x, y, z); }

#define DEF_OPENCL4F(NAME) \
EXPORT float CL_NAME(NAME)(float x, float y, float z, float w) { return NAME(x, y, z, w); } \
EXPORT double CL_NAME(NAME)(double x, double y, double z, double w) { return NAME(x, y, z, w); }

#define DEF_OPENCL1B(NAME) \
EXPORT int CL_NAME(NAME)(float x) { return NAME(x); } \
EXPORT long CL_NAME(NAME)(double x) { return NAME(x); }

#define DEF_OPENCL1INT(NAME) \
EXPORT int CL_NAME(NAME)(float x) { return NAME(x); } \
EXPORT int CL_NAME(NAME)(double x) { return NAME(x); }

#define DEF_OPENCL1F_NATIVE(NAME) \
EXPORT float CL_NAME(NAME##_native)(float x) { return native_##NAME(x); }


DEF_OPENCL1F(acos)
DEF_OPENCL1F(asin)
DEF_OPENCL1F(acosh)
DEF_OPENCL1F(asinh)
DEF_OPENCL1F(atan)
DEF_OPENCL2F(atan2)
DEF_OPENCL1F(atanh)
DEF_OPENCL1F(cbrt)
DEF_OPENCL1F(ceil)

DEF_OPENCL2F(copysign)

DEF_OPENCL1F(cos)
DEF_OPENCL1F(cosh)
DEF_OPENCL1F(cospi)

// OCML
float OVLD i0(float f);
double OVLD i0(double f);
EXPORT float CL_NAME(cyl_bessel_i0)(float x) { return i0(x); }
EXPORT double CL_NAME(cyl_bessel_i0)(double x) { return i0(x); }
float OVLD i1(float f);
double OVLD i1(double f);
EXPORT float CL_NAME(cyl_bessel_i1)(float x) { return i1(x); }
EXPORT double CL_NAME(cyl_bessel_i1)(double x) { return i1(x); }


DEF_OPENCL1F(erfc)
DEF_OPENCL1F(erf)

// OCML
DEFOCML_OPENCL1F(erfcinv)
DEFOCML_OPENCL1F(erfcx)
DEFOCML_OPENCL1F(erfinv)

DEF_OPENCL1F(exp10)
DEF_OPENCL1F(exp2)
DEF_OPENCL1F(exp)
DEF_OPENCL1F(expm1)
DEF_OPENCL1F(fabs)
DEF_OPENCL2F(fdim)
DEF_OPENCL1F(floor)

DEF_OPENCL3F(fma)

DEF_OPENCL2F(fmax)
DEF_OPENCL2F(fmin)
DEF_OPENCL2F(fmod)

float OVLD frexp(float f, PRIVATE_AS int *i);
double OVLD frexp(double f, PRIVATE_AS int *i);
float _Z14opencl_frexp_ffPi(float x, DEFAULT_AS int *i) {
  int tmp;
  float ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}
double _Z14opencl_frexp_ddPi(double x, DEFAULT_AS int *i) {
  int tmp;
  double ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}

DEF_OPENCL2F(hypot)
DEF_OPENCL1INT(ilogb)

DEF_OPENCL1B(isfinite)
DEF_OPENCL1B(isinf)
DEF_OPENCL1B(isnan)

DEFOCML_OPENCL1F(j0)
DEFOCML_OPENCL1F(j1)

float OVLD ldexp(float f, int k);
double OVLD ldexp(double f, int k);
EXPORT float CL_NAME(ldexp)(float x, int k) { return ldexp(x, k); }
EXPORT double CL_NAME(ldexp)(double x, int k) { return ldexp(x, k); }

float OVLD lgamma(float f, PRIVATE_AS int *signp);
double OVLD lgamma(double f, PRIVATE_AS int *signp);
EXPORT float CL_NAME(lgamma)(float x) { int sign; return lgamma(x, &sign); }
EXPORT double CL_NAME(lgamma)(double x) { int sign; return lgamma(x, &sign); }

DEF_OPENCL1F(log10)
DEF_OPENCL1F(log1p)
DEF_OPENCL1F(log2)
DEF_OPENCL1F(logb)
DEF_OPENCL1F(log)

// modf
float OVLD modf(float f, PRIVATE_AS float *i);
double OVLD modf(double f, PRIVATE_AS double *i);
float _Z13opencl_modf_ffPf(float x, DEFAULT_AS float *i) {
  float tmp;
  float ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}
double _Z13opencl_modf_ddPd(double x, DEFAULT_AS double *i) {
  double tmp;
  double ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}

// OCML
DEFOCML_OPENCL1F(nearbyint)
DEFOCML_OPENCL2F(nextafter)

float OVLD length(float4 f);
double OVLD length(double4 f);
EXPORT float CL_NAME(norm4d)(float x, float y, float z, float w) { float4 temp = (float4)(x, y, z, w); return length(temp); }
EXPORT double CL_NAME(norm4d)(double x, double y, double z, double w) { double4 temp = (double4)(x, y, z, w); return length(temp); }
EXPORT float CL_NAME(norm3d)(float x, float y, float z) { float4 temp = (float4)(x, y, z, 0.0f); return length(temp); }
EXPORT double CL_NAME(norm3d)(double x, double y, double z) { double4 temp = (double4)(x, y, z, 0.0); return length(temp); }


// OCML ncdf / ncdfinv
DEFOCML_OPENCL1F(normcdf)
DEFOCML_OPENCL1F(normcdfinv)

DEF_OPENCL2F(pow)
DEF_OPENCL2F(remainder)
// OCML
DEFOCML_OPENCL1F(rcbrt)

// remquo
float OVLD remquo(float x,   float y,  PRIVATE_AS int *quo);
double OVLD remquo(double x, double y, PRIVATE_AS int *quo);
float _Z15opencl_remquo_fffPi(float x, float y, DEFAULT_AS int *quo) {
  int tmp;
  float rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}
double _Z15opencl_remquo_dddPi(double x, double y, DEFAULT_AS int *quo) {
  int tmp;
  double rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}

// OCML
DEFOCML_OPENCL2F(rhypot)

// OCML rlen3 / rlen4
float OVLD rlen4(float4 f);
double OVLD rlen4(double4 f);
float OVLD rlen3(float3 f);
double OVLD rlen3(double3 f);

EXPORT float CL_NAME(rnorm4d)(float x, float y, float z, float w) { float4 temp = (float4)(x, y, z, w); return rlen4(temp); }
EXPORT double CL_NAME(rnorm4d)(double x, double y, double z, double w) { double4 temp = (double4)(x, y, z, w); return rlen4(temp); }
EXPORT float CL_NAME(rnorm3d)(float x, float y, float z) { float3 temp = (float3)(x, y, z); return rlen3(temp); }
EXPORT double CL_NAME(rnorm3d)(double x, double y, double z) { double3 temp = (double3)(x, y, z); return rlen3(temp); }


DEF_OPENCL1F(round)
DEF_OPENCL1F(rsqrt)

// OCML
float OVLD scalbn(float f, int k);
double OVLD scalbn(double f, int k);
EXPORT float CL_NAME(scalbn)(float x, int k) { return scalbn(x, k); }
EXPORT double CL_NAME(scalbn)(double x, int k) { return scalbn(x, k); }
// OCML
DEFOCML_OPENCL2F(scalb)

DEF_OPENCL1B(signbit)

DEF_OPENCL1F(sin)
DEF_OPENCL1F(sinh)
DEF_OPENCL1F(sinpi)
DEF_OPENCL1F(sqrt)
DEF_OPENCL1F(tan)
DEF_OPENCL1F(tanh)
DEF_OPENCL1F(tgamma)
DEF_OPENCL1F(trunc)



// sincos
float OVLD sincos(float x, PRIVATE_AS float *cosval);
double OVLD sincos(double x, PRIVATE_AS double *cosval);

float _Z15opencl_sincos_ffPf(float x, DEFAULT_AS float *cos) {
  PRIVATE_AS float tmp;
  PRIVATE_AS float sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

double _Z15opencl_sincos_ddPd(double x, DEFAULT_AS double *cos) {
  PRIVATE_AS double tmp;
  PRIVATE_AS double sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

// OCML
DEFOCML_OPENCL1F(y0)
DEFOCML_OPENCL1F(y1)

/* native */

DEF_OPENCL1F_NATIVE(cos)
DEF_OPENCL1F_NATIVE(sin)
DEF_OPENCL1F_NATIVE(tan)

DEF_OPENCL1F_NATIVE(exp10)
DEF_OPENCL1F_NATIVE(exp)

DEF_OPENCL1F_NATIVE(log10)
DEF_OPENCL1F_NATIVE(log2)
DEF_OPENCL1F_NATIVE(log)

/* other */

OVLD void CL_NAME(local_barrier)() { barrier(CLK_LOCAL_MEM_FENCE); }

/**********************************************************************/

#define DEF_OPENCL_ATOMIC2(NAME, LEN)                                          \
  int CL_NAME_MANGLED_ATOM(NAME, _i, LEN,                                      \
                           PVii)(volatile DEFAULT_AS int *address, int i) {    \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi, i);                                             \
    else {                                                                     \
      volatile local int *li = to_local(address);                              \
      if (gi)                                                                  \
        return atomic_##NAME(li, i);                                           \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };                                                                           \
  unsigned int CL_NAME_MANGLED_ATOM(NAME, _u, LEN, PVjj)(                      \
      volatile DEFAULT_AS unsigned int *address, unsigned int ui) {            \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi, ui);                                            \
    else {                                                                     \
      volatile local uint *li = to_local(address);                             \
      if (gi)                                                                  \
        return atomic_##NAME(li, ui);                                          \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };                                                                           \
  unsigned long long CL_NAME_MANGLED_ATOM(NAME, _l, LEN, PVyy)(                \
      volatile DEFAULT_AS unsigned long long *address,                         \
      unsigned long long ull) {                                                \
    volatile global ulong *gi =                                                \
        to_global((volatile DEFAULT_AS ulong *)address);                       \
    if (gi)                                                                    \
      return atom_##NAME(gi, ull);                                             \
    else {                                                                     \
      volatile local ulong *li =                                               \
          to_local((volatile DEFAULT_AS ulong *)address);                      \
      if (gi)                                                                  \
        return atom_##NAME(li, ull);                                           \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };

DEF_OPENCL_ATOMIC2(add, 19)
DEF_OPENCL_ATOMIC2(sub, 19)
DEF_OPENCL_ATOMIC2(xchg, 20)
DEF_OPENCL_ATOMIC2(min, 19)
DEF_OPENCL_ATOMIC2(max, 19)
DEF_OPENCL_ATOMIC2(and, 19)
DEF_OPENCL_ATOMIC2(or, 18)
DEF_OPENCL_ATOMIC2 (xor, 19)

#define DEF_OPENCL_ATOMIC1(NAME, LEN)                                          \
  int CL_NAME_MANGLED_ATOM(NAME, _i, LEN,                                      \
                           PVi)(volatile DEFAULT_AS int *address) {            \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi);                                                \
    volatile local int *li = to_local(address);                                \
    if (gi)                                                                    \
      return atomic_##NAME(li);                                                \
    return 0;                                                                  \
  };                                                                           \
  unsigned int CL_NAME_MANGLED_ATOM(NAME, _u, LEN, PVj)(                       \
      volatile DEFAULT_AS unsigned int *address) {                             \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi);                                                \
    volatile local uint *li = to_local(address);                               \
    if (gi)                                                                    \
      return atomic_##NAME(li);                                                \
    return 0;                                                                  \
  };                                                                           \
  unsigned long long CL_NAME_MANGLED_ATOM(NAME, _l, LEN, PVy)(                 \
      volatile DEFAULT_AS unsigned long long *address) {                       \
    volatile global ulong *gi =                                                \
        to_global((volatile DEFAULT_AS ulong *)address);                       \
    if (gi)                                                                    \
      return atom_##NAME(gi);                                                  \
    volatile local ulong *li = to_local((volatile DEFAULT_AS ulong *)address); \
    if (gi)                                                                    \
      return atom_##NAME(li);                                                  \
    return 0;                                                                  \
  };

DEF_OPENCL_ATOMIC1(inc, 19)
DEF_OPENCL_ATOMIC1(dec, 19)

#define DEF_OPENCL_ATOMIC3(NAME, LEN)                                          \
  int CL_NAME_MANGLED_ATOM(NAME, _i, LEN, PViii)(                              \
      volatile DEFAULT_AS int *address, int cmp, int val) {                    \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi, cmp, val);                                      \
    volatile local int *li = to_local(address);                                \
    if (gi)                                                                    \
      return atomic_##NAME(li, cmp, val);                                      \
    return 0;                                                                  \
  };                                                                           \
  unsigned int CL_NAME_MANGLED_ATOM(NAME, _u, LEN, PVjjj)(                     \
      volatile DEFAULT_AS unsigned int *address, unsigned int cmp,             \
      unsigned int val) {                                                      \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi, cmp, val);                                      \
    volatile local uint *li = to_local(address);                               \
    if (gi)                                                                    \
      return atomic_##NAME(li, cmp, val);                                      \
    return 0;                                                                  \
  };                                                                           \
  unsigned long long CL_NAME_MANGLED_ATOM(NAME, _l, LEN, PVyyy)(               \
      volatile DEFAULT_AS unsigned long long *address, unsigned long long cmp, \
      unsigned long long val) {                                                \
    volatile global ulong *gi =                                                \
        to_global((volatile DEFAULT_AS ulong *)address);                       \
    if (gi)                                                                    \
      return atom_##NAME(gi, cmp, val);                                        \
    volatile local ulong *li = to_local((volatile DEFAULT_AS ulong *)address); \
    if (gi)                                                                    \
      return atom_##NAME(li, cmp, val);                                        \
    return 0;                                                                  \
  };

DEF_OPENCL_ATOMIC3(cmpxchg, 23)

/* This code adapted from AMD's HIP sources */

OVLD float atomic_add_f(volatile local float *address, float val) {
  volatile local uint *uaddr = (volatile local uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

OVLD double atom_add_d(volatile local double *address, double val) {
  volatile local ulong *uaddr = (volatile local ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

OVLD float atomic_exch_f(volatile local float *address, float val) {
  return as_float(atomic_xchg((volatile local uint *)(address), as_uint(val)));
}

OVLD float atomic_add_f(volatile global float *address, float val) {
  volatile global uint *uaddr = (volatile global uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

OVLD double atom_add_d(volatile global double *address, double val) {
  volatile global ulong *uaddr = (volatile global ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

OVLD float atomic_exch_f(volatile global float *address, float val) {
  return as_float(atomic_xchg((volatile global uint *)(address), as_uint(val)));
}

float CL_NAME_MANGLED_ATOM(add, _f, 19,
                           PVff)(volatile DEFAULT_AS float *address,
                                 float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return atomic_add_f(gi, val);
  volatile local float *li = to_local(address);
  if (gi)
    return atomic_add_f(li, val);
  return 0;
}

double CL_NAME_MANGLED_ATOM(add, _d, 19,
                            PVdd)(volatile DEFAULT_AS double *address,
                                  double val) {
  volatile global double *gi = to_global((volatile DEFAULT_AS double *)address);
  if (gi)
    return atom_add_d(gi, val);
  volatile local double *li = to_local((volatile DEFAULT_AS double *)address);
  if (gi)
    return atom_add_d(li, val);
  return 0;
}

float CL_NAME_MANGLED_ATOM(exch, _f, 20,
                           PVff)(volatile DEFAULT_AS float *address,
                                 float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return atomic_exch_f(gi, val);
  volatile local float *li = to_local(address);
  if (gi)
    return atomic_exch_f(li, val);
  return 0;
}

/**********************************************************************/

int OVLD intel_sub_group_shuffle(int var, uint srcLane);
float OVLD intel_sub_group_shuffle(float var, uint srcLane);
OVLD int CL_NAME(shfl_i)(int var, int srcLane) {
  return intel_sub_group_shuffle(var, srcLane);
};
OVLD float CL_NAME(shfl_f)(float var, int srcLane) {
  return intel_sub_group_shuffle(var, srcLane);
};

int OVLD intel_sub_group_shuffle_xor(int var, uint value);
float OVLD intel_sub_group_shuffle_xor(float var, uint value);
OVLD int CL_NAME(shfl_xor_i)(int var, int value) {
  return intel_sub_group_shuffle_xor(var, value);
};
OVLD float CL_NAME(shfl_xor_f)(float var, int value) {
  return intel_sub_group_shuffle_xor(var, value);
};

int OVLD intel_sub_group_shuffle_up(int prev, int curr, uint delta);
float OVLD intel_sub_group_shuffle_up(float prev, float curr, uint delta);
OVLD int CL_NAME(shfl_up_i)(int var, unsigned int delta) {
  int tmp = 0;
  int tmp2 = intel_sub_group_shuffle_down(tmp, var, delta);
  return intel_sub_group_shuffle_up(tmp2, var, delta);
};
OVLD float CL_NAME(shfl_up_f)(float var, unsigned int delta) {
  float tmp = 0;
  float tmp2 = intel_sub_group_shuffle_down(tmp, var, delta);
  return intel_sub_group_shuffle_up(tmp2, var, delta);
};

int OVLD intel_sub_group_shuffle_down(int prev, int curr, uint delta);
float OVLD intel_sub_group_shuffle_down(float prev, float curr, uint delta);
OVLD int CL_NAME(shfl_down_i)(int var, unsigned int delta) {
  int tmp = 0;
  int tmp2 = intel_sub_group_shuffle_up(var, tmp, delta);
  return intel_sub_group_shuffle_down(var, tmp2, delta);
};
OVLD float CL_NAME(shfl_down_f)(float var, unsigned int delta) {
  float tmp = 0;
  float tmp2 = intel_sub_group_shuffle_up(var, tmp, delta);
  return intel_sub_group_shuffle_down(var, tmp2, delta);
};

int CL_NAME(group_all)(int pred) { return sub_group_all(pred); }

int CL_NAME(group_any)(int pred) { return sub_group_any(pred); }
