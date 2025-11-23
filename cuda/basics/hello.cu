#include <stdio.h>

__global__ void helloFromGPU() {
    
    // printf("Hello World from GPU!\n");
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Block %d, Thread %d says Hello World!\n", bid, tid);
}

int main() {
    // Launch kernel with 1 block and 1 thread
    // helloFromGPU<<<1, 1>>>();

    // this will print 2 x 3 = 6 times
    helloFromGPU<<<2, 3>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}


// I used this version, but it doesn't print anything.
// it turns out that there is mismatch between cuda driver and cuda toolkit versions
// catching the error helps me identify the issue.

// int main() {
//     // Launch kernel with 1 block and 1 thread
//     helloFromGPU<<<1, 1>>>();
    
//     // Wait for GPU to finish before accessing on host
//     cudaDeviceSynchronize();
//     return 0;
// }

// To compile and run this code, use the following commands:
// nvcc -o hello hello.cu
// ./hello
