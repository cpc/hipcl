#include <hip/hip_runtime.h>

extern "C" __global__ void _occa_addVectors_0(const size_t entries,
					      const float * a,
					      const float * b,
					      float * ab) {

    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < entries) {
        ab[i] = a[i] + b[i];
    }
}
