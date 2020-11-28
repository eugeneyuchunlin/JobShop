#include <c++/9/bits/c++config.h>
#include <stdio.h>
#include <stdlib.h>

#define ROW 5000
#define COL 5000


__global__ void matrixAddition(float *a, float *b, float *c, int N){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if( index < N ){
		c[index] = a[index] + b[index];
	}
}

__global__ void test(){
	printf("Hi Cuda World");
}


int main(){
	int N = ROW*COL;
	size_t size = N * sizeof(float);

	float * h_A = (float *)malloc(size);
	float * h_B = (float *)malloc(size);
	float * h_C = (float *)malloc(size);

	for(int i = 0; i < N; ++i){
		h_A[i] = i;
		h_B[i] = i;
	}

	// for(int i = 0; i < N; ++i){
	// 	printf("%.0f ",h_A[i]);
	// }
	// printf("\n");

	// for(int i = 0; i < N; ++i){
	// 	printf("%.0f ",h_B[i]);
	// }
	// printf("\n");


	float * d_A;
	float * d_B;
	float * d_C;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	matrixAddition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
		
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	for(int i = ROW*COL - 100; i < ROW*COL; ++i){
		printf("%.0f ", h_C[i]);
	}
	printf("\n");

	// test<<<1, 1>>>();
	cudaDeviceSynchronize();

}
