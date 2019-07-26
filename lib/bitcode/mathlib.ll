; ModuleID = 'mathlib.bc'
source_filename = "/home/devel/0/HIP_CL/lib/bitcode/mathlib.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_acosf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4acosf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4acosf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_acosd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4acosd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4acosd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_asinf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4asinf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4asinf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_asind(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4asind(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4asind(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_acoshf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5acoshf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5acoshf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_acoshd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5acoshd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5acoshd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_asinhf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5asinhf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5asinhf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_asinhd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5asinhd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5asinhd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_atanf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4atanf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4atanf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_atand(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4atand(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4atand(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_atan2ff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5atan2ff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5atan2ff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_atan2dd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5atan2dd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5atan2dd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_atanhf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5atanhf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5atanhf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_atanhd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5atanhd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5atanhd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_cbrtf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4cbrtf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4cbrtf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_cbrtd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4cbrtd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4cbrtd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_ceilf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4ceilf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4ceilf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_ceild(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4ceild(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4ceild(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z15opencl_copysignff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z8copysignff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z8copysignff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z15opencl_copysigndd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z8copysigndd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z8copysigndd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_cosf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3cosf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3cosf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_cosd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3cosd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3cosd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_coshf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4coshf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4coshf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_coshd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4coshd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4coshd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_cospif(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5cospif(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5cospif(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_cospid(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5cospid(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5cospid(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z20opencl_cyl_bessel_i0f(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z2i0f(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z2i0f(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z20opencl_cyl_bessel_i0d(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z2i0d(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z2i0d(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z20opencl_cyl_bessel_i1f(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z2i1f(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z2i1f(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z20opencl_cyl_bessel_i1d(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z2i1d(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z2i1d(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_erfcf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4erfcf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4erfcf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_erfcd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4erfcd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4erfcd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_erff(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3erff(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3erff(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_erfd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3erfd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3erfd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z14opencl_erfcinvf(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z7erfcinvf(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z7erfcinvf(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z14opencl_erfcinvd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z7erfcinvd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z7erfcinvd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z12opencl_erfcxf(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z5erfcxf(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z5erfcxf(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z12opencl_erfcxd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z5erfcxd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z5erfcxd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z13opencl_erfinvf(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z6erfinvf(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z6erfinvf(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z13opencl_erfinvd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z6erfinvd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z6erfinvd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_exp10f(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5exp10f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5exp10f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_exp10d(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5exp10d(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5exp10d(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_exp2f(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4exp2f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4exp2f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_exp2d(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4exp2d(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4exp2d(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_expf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3expf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3expf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_expd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3expd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3expd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_expm1f(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5expm1f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5expm1f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_expm1d(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5expm1d(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5expm1d(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_fabsf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4fabsf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4fabsf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_fabsd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4fabsd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4fabsd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_fdimff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4fdimff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4fdimff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_fdimdd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4fdimdd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4fdimdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_floorf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5floorf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5floorf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_floord(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5floord(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5floord(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_fmafff(float %x, float %y, float %z) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3fmafff(float %x, float %y, float %z) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3fmafff(float, float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_fmaddd(double %x, double %y, double %z) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3fmaddd(double %x, double %y, double %z) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3fmaddd(double, double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_fmaxff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4fmaxff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4fmaxff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_fmaxdd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4fmaxdd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4fmaxdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_fminff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4fminff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4fminff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_fmindd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4fmindd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4fmindd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_fmodff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4fmodff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4fmodff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_fmoddd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4fmoddd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4fmoddd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z12opencl_frexpfPi(float %x, i32 addrspace(4)* nocapture %i) local_unnamed_addr #2 {
entry:
  %tmp = alloca i32, align 4
  %0 = bitcast i32* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func float @_Z5frexpfPi(float %x, i32* nonnull %tmp) #12
  %1 = load i32, i32* %tmp, align 4, !tbaa !3
  store i32 %1, i32 addrspace(4)* %i, align 4, !tbaa !3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret float %call
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: convergent
declare dso_local spir_func float @_Z5frexpfPi(float, i32*) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z12opencl_frexpdPi(double %x, i32 addrspace(4)* nocapture %i) local_unnamed_addr #2 {
entry:
  %tmp = alloca i32, align 4
  %0 = bitcast i32* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func double @_Z5frexpdPi(double %x, i32* nonnull %tmp) #12
  %1 = load i32, i32* %tmp, align 4, !tbaa !3
  store i32 %1, i32 addrspace(4)* %i, align 4, !tbaa !3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z5frexpdPi(double, i32*) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_hypotff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5hypotff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5hypotff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_hypotdd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5hypotdd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5hypotdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i32 @_Z12opencl_ilogbf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z5ilogbf(float %x) #11
  ret i32 %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z5ilogbf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i32 @_Z12opencl_ilogbd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z5ilogbd(double %x) #11
  ret i32 %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z5ilogbd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i32 @_Z15opencl_isfinitef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z8isfinitef(float %x) #11
  ret i32 %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z8isfinitef(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i64 @_Z15opencl_isfinited(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z8isfinited(double %x) #11
  %conv = sext i32 %call to i64
  ret i64 %conv
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z8isfinited(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i32 @_Z12opencl_isinff(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z5isinff(float %x) #11
  ret i32 %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z5isinff(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i64 @_Z12opencl_isinfd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z5isinfd(double %x) #11
  %conv = sext i32 %call to i64
  ret i64 %conv
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z5isinfd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i32 @_Z12opencl_isnanf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z5isnanf(float %x) #11
  ret i32 %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z5isnanf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i64 @_Z12opencl_isnand(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z5isnand(double %x) #11
  %conv = sext i32 %call to i64
  ret i64 %conv
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z5isnand(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z9opencl_j0f(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z2j0f(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z2j0f(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z9opencl_j0d(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z2j0d(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z2j0d(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z9opencl_j1f(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z2j1f(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z2j1f(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z9opencl_j1d(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z2j1d(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z2j1d(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_ldexpfi(float %x, i32 %k) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5ldexpfi(float %x, i32 %k) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5ldexpfi(float, i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_ldexpdi(double %x, i32 %k) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5ldexpdi(double %x, i32 %k) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5ldexpdi(double, i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z13opencl_lgammaf(float %x) local_unnamed_addr #2 {
entry:
  %sign = alloca i32, align 4
  %0 = bitcast i32* %sign to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func float @_Z6lgammafPi(float %x, i32* nonnull %sign) #12
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z6lgammafPi(float, i32*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z13opencl_lgammad(double %x) local_unnamed_addr #2 {
entry:
  %sign = alloca i32, align 4
  %0 = bitcast i32* %sign to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func double @_Z6lgammadPi(double %x, i32* nonnull %sign) #12
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z6lgammadPi(double, i32*) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_log10f(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5log10f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5log10f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_log10d(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5log10d(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5log10d(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_log1pf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5log1pf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5log1pf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_log1pd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5log1pd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5log1pd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_log2f(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4log2f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4log2f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_log2d(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4log2d(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4log2d(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_logbf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4logbf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4logbf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_logbd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4logbd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4logbd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_logf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3logf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3logf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_logd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3logd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3logd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z11opencl_modffPf(float %x, float addrspace(4)* nocapture %i) local_unnamed_addr #2 {
entry:
  %tmp = alloca float, align 4
  %0 = bitcast float* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func float @_Z4modffPf(float %x, float* nonnull %tmp) #12
  %1 = load float, float* %tmp, align 4, !tbaa !7
  store float %1, float addrspace(4)* %i, align 4, !tbaa !7
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z4modffPf(float, float*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z11opencl_modfdPd(double %x, double addrspace(4)* nocapture %i) local_unnamed_addr #2 {
entry:
  %tmp = alloca double, align 8
  %0 = bitcast double* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #13
  %call = call spir_func double @_Z4modfdPd(double %x, double* nonnull %tmp) #12
  %1 = load double, double* %tmp, align 8, !tbaa !9
  store double %1, double addrspace(4)* %i, align 8, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #13
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z4modfdPd(double, double*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z16opencl_nearbyintf(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z9nearbyintf(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z9nearbyintf(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z16opencl_nearbyintd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z9nearbyintd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z9nearbyintd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z16opencl_nextafterff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z9nextafterff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z9nextafterff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z16opencl_nextafterdd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z9nextafterdd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z9nextafterdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z13opencl_norm4dffff(float %x, float %y, float %z, float %w) local_unnamed_addr #5 {
entry:
  %vecinit = insertelement <4 x float> undef, float %x, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float %y, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit1, float %z, i32 2
  %vecinit3 = insertelement <4 x float> %vecinit2, float %w, i32 3
  %call = tail call spir_func float @_Z6lengthDv4_f(<4 x float> %vecinit3) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z6lengthDv4_f(<4 x float>) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z13opencl_norm4ddddd(double %x, double %y, double %z, double %w) local_unnamed_addr #6 {
entry:
  %vecinit = insertelement <4 x double> undef, double %x, i32 0
  %vecinit1 = insertelement <4 x double> %vecinit, double %y, i32 1
  %vecinit2 = insertelement <4 x double> %vecinit1, double %z, i32 2
  %vecinit3 = insertelement <4 x double> %vecinit2, double %w, i32 3
  %call = tail call spir_func double @_Z6lengthDv4_d(<4 x double> %vecinit3) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z6lengthDv4_d(<4 x double>) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z13opencl_norm3dfff(float %x, float %y, float %z) local_unnamed_addr #5 {
entry:
  %0 = insertelement <4 x float> <float undef, float undef, float undef, float 0.000000e+00>, float %x, i32 0
  %1 = insertelement <4 x float> %0, float %y, i32 1
  %vecinit3 = insertelement <4 x float> %1, float %z, i32 2
  %call = tail call spir_func float @_Z6lengthDv4_f(<4 x float> %vecinit3) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z13opencl_norm3dddd(double %x, double %y, double %z) local_unnamed_addr #6 {
entry:
  %0 = insertelement <4 x double> <double undef, double undef, double undef, double 0.000000e+00>, double %x, i32 0
  %1 = insertelement <4 x double> %0, double %y, i32 1
  %vecinit3 = insertelement <4 x double> %1, double %z, i32 2
  %call = tail call spir_func double @_Z6lengthDv4_d(<4 x double> %vecinit3) #11
  ret double %call
}

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z14opencl_normcdff(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z7normcdff(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z7normcdff(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z14opencl_normcdfd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z7normcdfd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z7normcdfd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z17opencl_normcdfinvf(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z10normcdfinvf(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z10normcdfinvf(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z17opencl_normcdfinvd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z10normcdfinvd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z10normcdfinvd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_powff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3powff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3powff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_powdd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3powdd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3powdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z16opencl_remainderff(float %x, float %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z9remainderff(float %x, float %y) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z9remainderff(float, float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z16opencl_remainderdd(double %x, double %y) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z9remainderdd(double %x, double %y) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z9remainderdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z12opencl_rcbrtf(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z5rcbrtf(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z5rcbrtf(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z12opencl_rcbrtd(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z5rcbrtd(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z5rcbrtd(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z13opencl_remquoffPi(float %x, float %y, i32 addrspace(4)* nocapture %quo) local_unnamed_addr #2 {
entry:
  %tmp = alloca i32, align 4
  %0 = bitcast i32* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func float @_Z6remquoffPi(float %x, float %y, i32* nonnull %tmp) #12
  %1 = load i32, i32* %tmp, align 4, !tbaa !3
  store i32 %1, i32 addrspace(4)* %quo, align 4, !tbaa !3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z6remquoffPi(float, float, i32*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z13opencl_remquoddPi(double %x, double %y, i32 addrspace(4)* nocapture %quo) local_unnamed_addr #2 {
entry:
  %tmp = alloca i32, align 4
  %0 = bitcast i32* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func double @_Z6remquoddPi(double %x, double %y, i32* nonnull %tmp) #12
  %1 = load i32, i32* %tmp, align 4, !tbaa !3
  store i32 %1, i32 addrspace(4)* %quo, align 4, !tbaa !3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z6remquoddPi(double, double, i32*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z13opencl_rhypotff(float %x, float %y) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z6rhypotff(float %x, float %y) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z6rhypotff(float, float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z13opencl_rhypotdd(double %x, double %y) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z6rhypotdd(double %x, double %y) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z6rhypotdd(double, double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z14opencl_rnorm4dffff(float %x, float %y, float %z, float %w) local_unnamed_addr #7 {
entry:
  %vecinit = insertelement <4 x float> undef, float %x, i32 0
  %vecinit1 = insertelement <4 x float> %vecinit, float %y, i32 1
  %vecinit2 = insertelement <4 x float> %vecinit1, float %z, i32 2
  %vecinit3 = insertelement <4 x float> %vecinit2, float %w, i32 3
  %call = tail call spir_func float @_Z5rlen4Dv4_f(<4 x float> %vecinit3) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z5rlen4Dv4_f(<4 x float>) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z14opencl_rnorm4ddddd(double %x, double %y, double %z, double %w) local_unnamed_addr #8 {
entry:
  %vecinit = insertelement <4 x double> undef, double %x, i32 0
  %vecinit1 = insertelement <4 x double> %vecinit, double %y, i32 1
  %vecinit2 = insertelement <4 x double> %vecinit1, double %z, i32 2
  %vecinit3 = insertelement <4 x double> %vecinit2, double %w, i32 3
  %call = tail call spir_func double @_Z5rlen4Dv4_d(<4 x double> %vecinit3) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z5rlen4Dv4_d(<4 x double>) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z14opencl_rnorm3dfff(float %x, float %y, float %z) local_unnamed_addr #9 {
entry:
  %vecinit = insertelement <3 x float> undef, float %x, i32 0
  %vecinit1 = insertelement <3 x float> %vecinit, float %y, i32 1
  %vecinit2 = insertelement <3 x float> %vecinit1, float %z, i32 2
  %call = tail call spir_func float @_Z5rlen3Dv3_f(<3 x float> %vecinit2) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z5rlen3Dv3_f(<3 x float>) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z14opencl_rnorm3dddd(double %x, double %y, double %z) local_unnamed_addr #10 {
entry:
  %vecinit = insertelement <3 x double> undef, double %x, i32 0
  %vecinit1 = insertelement <3 x double> %vecinit, double %y, i32 1
  %vecinit2 = insertelement <3 x double> %vecinit1, double %z, i32 2
  %call = tail call spir_func double @_Z5rlen3Dv3_d(<3 x double> %vecinit2) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z5rlen3Dv3_d(<3 x double>) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_roundf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5roundf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5roundf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_roundd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5roundd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5roundd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_rsqrtf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5rsqrtf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5rsqrtf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_rsqrtd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5rsqrtd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5rsqrtd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z13opencl_scalbnfi(float %x, i32 %k) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z6scalbnfi(float %x, i32 %k) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z6scalbnfi(float, i32) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z13opencl_scalbndi(double %x, i32 %k) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z6scalbndi(double %x, i32 %k) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z6scalbndi(double, i32) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z12opencl_scalbff(float %x, float %y) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z5scalbff(float %x, float %y) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z5scalbff(float, float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z12opencl_scalbdd(double %x, double %y) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z5scalbdd(double %x, double %y) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z5scalbdd(double, double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i32 @_Z14opencl_signbitf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z7signbitf(float %x) #11
  ret i32 %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z7signbitf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func i64 @_Z14opencl_signbitd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z7signbitd(double %x) #11
  %conv = sext i32 %call to i64
  ret i64 %conv
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z7signbitd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_sinf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3sinf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3sinf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_sind(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3sind(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3sind(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_sinhf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4sinhf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4sinhf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_sinhd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4sinhd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4sinhd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_sinpif(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5sinpif(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5sinpif(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_sinpid(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5sinpid(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5sinpid(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_sqrtf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4sqrtf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4sqrtf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_sqrtd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4sqrtd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4sqrtd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z10opencl_tanf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z3tanf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z3tanf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z10opencl_tand(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z3tand(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z3tand(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z11opencl_tanhf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z4tanhf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z4tanhf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z11opencl_tanhd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z4tanhd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z4tanhd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z13opencl_tgammaf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z6tgammaf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z6tgammaf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z13opencl_tgammad(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z6tgammad(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z6tgammad(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z12opencl_truncf(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z5truncf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z5truncf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func double @_Z12opencl_truncd(double %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func double @_Z5truncd(double %x) #11
  ret double %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func double @_Z5truncd(double) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z13opencl_sincosfPf(float %x, float addrspace(4)* nocapture %cos) local_unnamed_addr #2 {
entry:
  %tmp = alloca float, align 4
  %0 = bitcast float* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #13
  %call = call spir_func float @_Z6sincosfPf(float %x, float* nonnull %tmp) #12
  %1 = load float, float* %tmp, align 4, !tbaa !7
  store float %1, float addrspace(4)* %cos, align 4, !tbaa !7
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #13
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z6sincosfPf(float, float*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z13opencl_sincosdPd(double %x, double addrspace(4)* nocapture %cos) local_unnamed_addr #2 {
entry:
  %tmp = alloca double, align 8
  %0 = bitcast double* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #13
  %call = call spir_func double @_Z6sincosdPd(double %x, double* nonnull %tmp) #12
  %1 = load double, double* %tmp, align 8, !tbaa !9
  store double %1, double addrspace(4)* %cos, align 8, !tbaa !9
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #13
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z6sincosdPd(double, double*) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z9opencl_y0f(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z2y0f(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z2y0f(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z9opencl_y0d(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z2y0d(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z2y0d(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func float @_Z9opencl_y1f(float %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func float @_Z2y1f(float %x) #12
  ret float %call
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z2y1f(float) local_unnamed_addr #3

; Function Attrs: convergent nounwind
define dso_local spir_func double @_Z9opencl_y1d(double %x) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func double @_Z2y1d(double %x) #12
  ret double %call
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z2y1d(double) local_unnamed_addr #3

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z17opencl_cos_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z10native_cosf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z10native_cosf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z17opencl_sin_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z10native_sinf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z10native_sinf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z17opencl_tan_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z10native_tanf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z10native_tanf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z19opencl_exp10_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z12native_exp10f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z12native_exp10f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z17opencl_exp_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z10native_expf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z10native_expf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z19opencl_log10_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z12native_log10f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z12native_log10f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z18opencl_log2_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z11native_log2f(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z11native_log2f(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
define dso_local spir_func float @_Z17opencl_log_nativef(float %x) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func float @_Z10native_logf(float %x) #11
  ret float %call
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func float @_Z10native_logf(float) local_unnamed_addr #1

; Function Attrs: convergent nounwind
define dso_local spir_func void @_Z20opencl_local_barrierv() local_unnamed_addr #2 {
entry:
  tail call spir_func void @_Z7barrierj(i32 1) #12
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z7barrierj(i32) local_unnamed_addr #3

attributes #0 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="96" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="192" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { convergent nounwind readnone }
attributes #12 = { convergent nounwind }
attributes #13 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 8.0.1 (https://github.com/llvm-mirror/clang.git b4c3616b9b0976c6a59867219d2f3e3711b03726) (https://github.com/llvm-mirror/llvm.git ff8c1be17aa3ba7bacb1ef7dcdbecf05d5ab4eb7)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !5, i64 0}
