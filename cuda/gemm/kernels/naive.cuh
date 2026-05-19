#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// A simple matmul implementation: the idea is we are going to fill an output matrix C which is of shape [M, N].
// Each thread will fill a value at a position of [x, y], which corresponds to the dot product of x row in A and y column in B.

// Notice that the indexing in kernel is always 1-dim. We can think of it as the offset to the beginning pointer.

// In naive kernel, each thread is responsible for a single element in the
// output matrix.
__global__ void naive_gemm_kernel(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, const int M,
                                  const int N, const int K) {

  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
      sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = sum;
  }
}


// Here we directly map each thread position [x, y] to the output matrix position.
