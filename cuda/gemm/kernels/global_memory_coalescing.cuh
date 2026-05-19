#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// In this kernel, the block shape is defined as 1D (BLOCK_SIZE * BLOCK_SIZE),
// but regarded as 2D (BLOCK_SIZE, BLOCK_SIZE). In this way, the threads in the
// same warp can compute elements in the same row of output matrix.


// each threadblock will write a block in the output matrix, and each thread in the block will write a single element in that block. 
// We're writing the block in row-major order, so the threads in the same warp (adjacent threads) will write elements in the same row.
// because we're writing in row-major, For A matrix, we will read in row-major order, which is coalesced.


template <const int BLOCKSIZE>
__global__ void global_memory_coalescing_gemm_kernel(float *__restrict__ a,
                                                     float *__restrict__ b,
                                                     float *__restrict__ c,
                                                     const int M, const int N,
                                                     const int K) {

  const uint m = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint n = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (m < M && n < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = sum;
  }
}




template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
