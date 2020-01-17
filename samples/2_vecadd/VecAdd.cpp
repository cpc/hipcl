#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>

// hip header file
#include "hipcl.hh"
#include "hip_vector_types.h"

#define SEED 19284975223

#define LOC_WG 16
#define GRID_WG 128
#define NUM (GRID_WG*LOC_WG)

/*****************************************************************************/

typedef std::function<float(void)> RandomGenFuncFloat;
typedef std::function<int(void)> RandomGenFuncInt;

template <typename T>
__global__ void
VecMAD (const T * __restrict A, const T * __restrict B, T * __restrict C, const T multiplier)
{
  const uint i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  C[i] = A[i] + B[i] * multiplier;
}


/*****************************************************************************/

template <typename T>
__host__
void ArrayMADcpuReference(const T* __restrict A,
                          const T* __restrict B,
                          T * __restrict C,
                          const T multiplier) {
  for (uint i = 0; i < NUM; i++) {
    C[i] = A[i] + B[i] * multiplier; // + make_float4(6.0f, 1.0f, 2.0f, 3.0f);
  }
}

template <typename T, typename CMPT>
bool compareRes4(size_t i, bool print, T res1, T res2)
{
  CMPT res = (res1 != res2);
  if (res.x + res.y + res.z + res.w == 0)
    return true;

  if (print) {
      std::cerr << "FAIL AT: " << i << "\n";
      std::cerr << "CMP: " << res.x << " "
                << res.y << " "
                << res.z << " "
                << res.w << "\n";

      std::cerr << "CPU: " << res1.x << " "
                << res1.y << " "
                << res1.z << " "
                << res1.w << "\n";
      std::cerr << "GPU: " << res2.x << " "
                << res2.y << " "
                << res2.z << " "
                << res2.w << "\n";
    }
  return false;
}

template <typename T, typename CMPT>
bool compareRes2(size_t i, bool print, T res1, T res2)
{
  CMPT res = (res1 != res2);

  if (res.x + res.y == 0)
    return true;

  if (print) {
      std::cerr << "FAIL AT: " << i << "\n";
      std::cerr << "CMP: " << res.x << " "
                << res.y << "\n";

      std::cerr << "CPU: " << res1.x << " "
                << res1.y << "\n";
      std::cerr << "GPU: " << res2.x << " "
                << res2.y << "\n";
  }
  return false;
}


#define ERR_CHECK_2 \
  do { \
  err = hipGetLastError(); \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


#define ERR_CHECK \
  do { \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


template <typename T, typename RNG>
__host__ int TestVectors(RNG rnd, const T multiplier,
                         bool (*comparator)(size_t i, bool print, T res1, T res2)
                         ) {
    hipError_t err;
    T* Array1;
    T* Array2;
    T* ResArray;
    T* cpuResArray;

    T* gpuArray1;
    T* gpuArray2;
    T* gpuResArray;

    Array1 = new T [NUM];
    Array2 = new T [NUM];
    ResArray = new T [NUM];
    cpuResArray = new T [NUM];

    // initialize the input data
    for (size_t i = 0; i < NUM; i++) {
        Array1[i] = (T)rnd();
        Array2[i] = (T)rnd();
    }

    err = hipMalloc((void**)&gpuArray1, NUM * sizeof(T));
    ERR_CHECK;
    err = hipMalloc((void**)&gpuArray2, NUM * sizeof(T));
    ERR_CHECK;
    err = hipMalloc((void**)&gpuResArray, NUM * sizeof(T));
    ERR_CHECK;

    err = hipMemcpy(gpuArray1, Array1, NUM * sizeof(T), hipMemcpyHostToDevice);
    ERR_CHECK;
    err = hipMemcpy(gpuArray2, Array2, NUM * sizeof(T), hipMemcpyHostToDevice);
    ERR_CHECK;
    hipLaunchKernelGGL(VecMAD<T>,
                       dim3(GRID_WG),
                       dim3(LOC_WG),
                       0, 0,
                       gpuArray1, gpuArray2, gpuResArray, multiplier);
    ERR_CHECK_2;
    err = hipMemcpy(ResArray, gpuResArray, NUM * sizeof(T), hipMemcpyDeviceToHost);
    ERR_CHECK;

    ArrayMADcpuReference<T>(Array1, Array2, cpuResArray, multiplier);

    size_t failures = 0;
    for (size_t i = 0; i < NUM; i++) {
        if (comparator(i, (failures < 50), cpuResArray[i], ResArray[i]))
          continue;
        ++failures;
      }

    if (failures > 0) {
        std::cout << "FAIL: " << failures << " failures \n";
      }
    else {
        std::cout << "PASSED\n";
      }

    // free the resources on device side
    err = hipFree(gpuArray1);
    ERR_CHECK;
    err = hipFree(gpuArray2);
    ERR_CHECK;
    err = hipFree(gpuResArray);
    ERR_CHECK;

    // free the resources on host side
    delete [] Array1;
    delete [] Array2;
    delete [] ResArray;
    delete [] cpuResArray;

    return 0;
}


int main() {

  hipError_t err;

  std::mt19937 gen(SEED);
  RandomGenFuncFloat rndf = std::bind(std::uniform_real_distribution<float>{100.0f, 1000.0f}, gen);
  RandomGenFuncInt rndi = std::bind(std::uniform_int_distribution<int>{100, 1000}, gen);
  //RandomGenFunc a = rnd;
  //std::function<float(void)> fun = rnd;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << "Device name " << devProp.name << std::endl;

  std::cout << "FLOAT4 test\n";
  float4 m_f4 = make_float4(7.0f, 7.0f, 7.0f, 7.0f);
  TestVectors<float4, RandomGenFuncFloat>(rndf, m_f4, compareRes2<float4, int4>);

  std::cout << "INT2 test\n";
  int2 m_i2 = make_int2(22, 19);
  TestVectors<int2, RandomGenFuncInt>(rndi, m_i2, compareRes2<int2, int2>);

  std::cout << "INT4 test\n";
  int4 m_i4 = make_int4(3, 17, 48, 29);
  TestVectors<int4, RandomGenFuncInt>(rndi, m_i4, compareRes4<int4, int4>);
}
