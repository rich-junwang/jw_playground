#include <stdio.h>
#include <cuda_runtime.h>
// #include <cuda.h>
#include <cuda_bf16.h>

using dtype = __nv_bfloat16;


// kernel must return void
__global__ void adder(dtype* a, dtype* b, dtype* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = dtype(float(a[idx]) + float(b[idx]));
    }
}


int main() {
    const int N = 1 << 10; // 1M elements
    const int size = N * sizeof(dtype);

    // Allocate host memory
    dtype *h_a = (dtype*)malloc(size);
    dtype *h_b = (dtype*)malloc(size);
    dtype *h_c = (dtype*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = dtype(i);
        h_b[i] = dtype(i + 1);
    }

    // Allocate device memory
    dtype *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy host arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    adder<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i) + static_cast<float>(i + 1);
        float result = __bfloat162float(h_c[i]);
        if (fabs(result - expected) > 1.0) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, result);
            break;
        }
    }
    printf("Computation completed successfully.\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;    
}
