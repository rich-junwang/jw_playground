#include<stdio.h>

__global__ void kernel(int a, int b, int *c){
	*c = a + b;
}

int main(){
	int c = 20;
	int *c_cuda;

	cudaMalloc((void**)&c_cuda,sizeof(int));

	kernel<<<1,1>>>(1,1,c_cuda);
	cudaMemcpy(&c,c_cuda,sizeof(int),cudaMemcpyDeviceToHost);
	printf("c=%d\n",c);
	cudaFree(c_cuda);
	return 0;
}
