/*
Copyright (c) 2019 Michal Babej / Tampere University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>

// hip header file
#include "hipcl.hh"

#define SEED 19284975223

// Defined = use the MatMul with shared memory, not defined = use simplest possible MatMul
#define MM_SHARED

// Defined = use FMA instruction
//#define USE_FMA

// the required shared memory is (2 * 4 * THREADS_PER_BLOCK * THREADS_PER_BLOCK) bytes
#define THREADS_PER_BLOCK 64

// configure matrix size here. Must be power of 4 at least 64
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)

/*****************************************************************************/

#ifndef MM_SHARED

// Simple version, myGEMM2
__global__ void
gpuMatrixMul (const float * __restrict A, const float * __restrict B, float * __restrict C,
         uint M, uint N, uint K)

{
  // Thread identifiers
  const uint globalRow = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // Row ID of C (0..M)
  const uint globalCol = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (uint k = 0; k < K; k++)
    {
#ifdef USE_FMA
      acc = fmaf (A[k * M + globalRow], B[globalCol * K + k], acc);
#else
      acc += A[k * M + globalRow] * B[globalCol * K + k];
#endif
    }

  // Store the result
  C[globalCol * M + globalRow] = acc;
}


#else
#define TS THREADS_PER_BLOCK
/* work per thread */
#define WPT (THREADS_PER_BLOCK / 4)

// TS/WPT == RTS
#define RTS 4

// Tiled and coalesced version, myGEMM4
__global__ void
gpuMatrixMul (const float * __restrict A,
              const float * __restrict B,
              float * __restrict C,
              uint M, uint N, uint K)
{

  // Thread identifiers
  const uint row = hipThreadIdx_x; // Local row ID (max: TS)
  const uint col = hipThreadIdx_y; // Local col ID (max: TS/WPT == RTS)
  const uint globalRow = TS * hipBlockIdx_x + row; // Row ID of C (0..M)
  const uint globalCol = TS * hipBlockIdx_y + col; // Col ID of C (0..N)

  // Local memory to fit a tile of TS*TS elements of A and B
  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  // Initialise the accumulation registers
  float acc[WPT];
  for (uint w = 0; w < WPT; w++)
    {
      acc[w] = 0.0f;
    }

  // Loop over all tiles
  const uint numTiles = K / TS;
  for (uint t = 0; t < numTiles; t++)
    {

      // Load one tile of A and B into local memory
      for (uint w = 0; w < WPT; w++)
        {
          const uint tiledRow = TS * t + row;
          const uint tiledCol = TS * t + col;
          Asub[col + w * RTS][row] = A[(tiledCol + w * RTS) * M + globalRow];
          Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tiledRow];
        }

      // Synchronise to make sure the tile is loaded
      __syncthreads();

      // Perform the computation for a single tile
      for (uint k = 0; k < TS; k++)
        {
          for (uint w = 0; w < WPT; w++)
            {
#ifdef USE_FMA
              acc[w] = fmaf (Asub[k][row], Bsub[col + w * RTS][k], acc[w]);
#else
              acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
#endif
            }
        }

      // Synchronise before loading the next tile
      __syncthreads();
    }

  // Store the final results in C
  for (uint w = 0; w < WPT; w++)
    {
      C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}
#endif

/*****************************************************************************/


// CPU implementation of matrix transpose
void matrixMultiplyCPUReference(const float * __restrict A,
                                const float * __restrict B,
                                float * __restrict C) {
  for (uint i = 0; i < WIDTH; i++) {
    for (uint j = 0; j < WIDTH; j++) {
          float acc = 0.0f;
          for (uint k = 0; k < WIDTH; k++) {
#ifdef USE_FMA
            acc = __builtin_fmaf (A[k*WIDTH + j], B[i*WIDTH + k], acc);
#else
            acc += B[i*WIDTH + k] * A[k*WIDTH + j];
#endif
          }
          C[i*WIDTH + j] = acc;
        }
    }
}

int main() {

  std::mt19937 gen(SEED);
  auto rnd = std::bind(std::uniform_real_distribution<float>{100.0f, 1000.0f}, gen);

    float* Matrix1;
    float* Matrix2;
    float* MultiplyMatrix;
    float* cpuMultiplyMatrix;

    float* gpuMatrix1;
    float* gpuMatrix2;
    float* gpuMultiplyMatrix;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    size_t i, j;
    int errors, err;

    Matrix1 = new float [NUM];
    Matrix2 = new float [NUM];
    MultiplyMatrix = new float [NUM];
    cpuMultiplyMatrix = new float [NUM];

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix1[i] = rnd();
        Matrix2[i] = rnd();
    }

    hipEvent_t start1, stop1, start2, stop2, start3, stop3;
    hipEventCreate(&start1);
    hipEventCreate(&stop1);
    hipEventCreate(&start2);
    hipEventCreate(&stop2);
    hipEventCreate(&start3);
    hipEventCreate(&stop3);
    float eventMs = 1.0f;

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix1, NUM * sizeof(float));
    hipMalloc((void**)&gpuMatrix2, NUM * sizeof(float));
    hipMalloc((void**)&gpuMultiplyMatrix, NUM * sizeof(float));

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix1, Matrix1, NUM * sizeof(float), hipMemcpyHostToDevice);

    hipEventRecord(start1, NULL);
    hipMemcpy(gpuMatrix2, Matrix2, NUM * sizeof(float), hipMemcpyHostToDevice);
    hipEventRecord(stop1, NULL);

    hipEventRecord(start2, NULL);
    // Lauching kernel from host
    hipLaunchKernelGGL(gpuMatrixMul,
#ifndef MM_SHARED
                       dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
#else
                       dim3((WIDTH / THREADS_PER_BLOCK), (WIDTH / THREADS_PER_BLOCK)),
                       dim3(THREADS_PER_BLOCK, 4),
#endif
                       0, 0,
                       gpuMatrix1, gpuMatrix2, gpuMultiplyMatrix, WIDTH, WIDTH, WIDTH);
    hipEventRecord(stop2, NULL);

    hipEventRecord(start3, NULL);
    // Memory transfer from device to host
    hipMemcpy(MultiplyMatrix, gpuMultiplyMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
    hipEventRecord(stop3, NULL);

    hipDeviceSynchronize();

    err = hipEventElapsedTime(&eventMs, start1, stop1);
    assert(err == hipSuccess);
    printf("hipMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

    err = hipEventElapsedTime(&eventMs, start2, stop2);
    assert(err == hipSuccess);
    printf("hipLaunchKernel time taken  = %6.3fms\n", eventMs);

    err = hipEventElapsedTime(&eventMs, start3, stop3);
    assert (err == hipSuccess);
    printf("hipMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);

    auto time1 = std::chrono::high_resolution_clock::now();
    // CPU MatrixTranspose computation
    matrixMultiplyCPUReference(Matrix1, Matrix2, cpuMultiplyMatrix);
    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = time2 - time1;
    printf("matrixMultiplyCPUReference time taken  = %6.3fms\n", fp_ms.count());

    // verify the results
    errors = 0;
    float eps = 1.0;
    for (i = 0; i < WIDTH; i++) {
        for (j = 0; j < WIDTH; j++) {
          float cpu = cpuMultiplyMatrix[i*WIDTH+j];
          float gpu = MultiplyMatrix[i*WIDTH+j];
          if (std::abs(gpu - cpu) > eps) {
            errors++;
            std::cerr << "E[" << i << "][" << j << "]: M1 "
                      << Matrix1[i*WIDTH+j] << " M2 " << Matrix1[i*WIDTH+j]
                      << " CPU: " << cpu << " GPU: "
                      << gpu << " ERROR: " << std::abs(gpu - cpu) << "\n";
          }
        }
    }
    if (errors != 0) {
        printf("Verification FAILED: %d errors\n", errors);
    } else {
        printf("Verification PASSED!\n");
    }

    hipEventDestroy(start1);
    hipEventDestroy(stop1);
    hipEventDestroy(start2);
    hipEventDestroy(stop2);
    hipEventDestroy(start3);
    hipEventDestroy(stop3);

    // free the resources on device side
    hipFree(gpuMatrix1);
    hipFree(gpuMatrix2);
    hipFree(gpuMultiplyMatrix);

    // free the resources on host side
    delete [] Matrix1;
    delete [] Matrix2;
    delete [] MultiplyMatrix;
    delete [] cpuMultiplyMatrix;

    return errors;
}
