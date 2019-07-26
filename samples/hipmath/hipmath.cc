
#include <stdio.h>
#include "hipcl.hh"

#include <cstdlib>
#include <cmath>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(1);                                                                    \
        }                                                                                          \
    }

#define LEN 512
#define SIZE LEN << 4


__global__ void floatMath(float* In, float* SinOut, float* CosOut) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    sincosf(In[tid], SinOut+tid, CosOut+tid);
}

#define CHECK_N 32

int main() {
    float *Ind, *Inh;
    float *SinOutd, *SinOuth;
    float *CosOutd, *CosOuth;

    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);

    Inh = (float*)malloc(SIZE);
    CHECK(Inh == 0 ? hipErrorMemoryAllocation : hipSuccess);
    SinOuth = (float*)malloc(SIZE);
    CHECK(SinOuth == 0 ? hipErrorMemoryAllocation : hipSuccess);
    CosOuth = (float*)malloc(SIZE);
    CHECK(CosOuth == 0 ? hipErrorMemoryAllocation : hipSuccess);

    for (size_t i = 0; i < LEN; i++) {
        Inh[i] = 0.618f + ((float)i) / 7.0;
    }

    hipMalloc((void**)&Ind, SIZE);
    hipMalloc((void**)&SinOutd, SIZE);
    hipMalloc((void**)&CosOutd, SIZE);

    printf("info: copy Host2Device\n");
    CHECK(hipMemcpy(Ind, Inh, SIZE, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(floatMath, dim3(LEN, 1, 1), dim3(1, 1, 1), 0, 0, Ind, SinOutd, CosOutd);

    printf("info: copy Device2Host\n");
    CHECK(hipMemcpy(SinOuth, SinOutd, SIZE, hipMemcpyDeviceToHost));

    printf("info: copy Device2Host\n");
    CHECK(hipMemcpy(CosOuth, CosOutd, SIZE, hipMemcpyDeviceToHost));

    float diff = 0.0f;
    float eps = 1.0E-6;

    for (size_t i = 0; i < CHECK_N; i++) {
        // Compare to C library values
        float msin = std::sin(Inh[i]);
        float sindiff = std::fabs(SinOuth[i]-msin);
        float mcos = std::cos(Inh[i]);
        float cosdiff = std::fabs(CosOuth[i]-mcos);
        printf("IN: %1.5f Sin %1.5f Diff %1.5f Cos %1.5f Diff %1.5f\n",
               Inh[i], SinOuth[i], sindiff, CosOuth[i], cosdiff);
    }

    if (diff < (eps * CHECK_N))
      printf("PASSED\n");
    else
      printf("FAILED\n");
}

