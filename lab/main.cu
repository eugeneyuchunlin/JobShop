#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define ROW 5000
#define COL 5000


__global__ void matrixAddition(float *a, float *b, float *c, int N){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if( index < N ){
		c[index] = a[index] + b[index];
	}
}

__global__ void test(int ** data){
	int indexx = threadIdx.x;
	int indexy = threadIdx.y;
	// printf("index = %d\n", threadIdx.x);
	// printf("%d\n", data[threadIdx.x][threadIdx.y]);
	data[indexx][indexy] += 1;
}

__global__ void test2(int N){
	int index = threadIdx.x;
	printf("CUDA said: hello world\n");
}

__global__ void setup_kernel(curandState * state){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]);
}

__global__ void testrand(curandState * state){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = state[id];
	unsigned int num = ceilf(curand_uniform(&localState) * 30) - 1;
	printf("rand = %u\n", num);
}

__global__ void testmemcpy(double * dest, double * src, unsigned int length){
	memcpy(dest + 5, src, sizeof(double)*(length - 5));
}


int main(){
	int ** array = (int **)malloc(10*sizeof(int*));
	for(int i = 0; i < 10; ++i){
		array[i] = (int *)malloc(10*sizeof(int));
		for(int j = 0; j < 10; ++j){
			array[i][j] = i + j;	
			printf("%d ", array[i][j]);
		}
		printf("\n");
	}

	

	int ** dev_array;
	cudaMalloc((void **)&dev_array, sizeof(int*) *10);
	int * dev_temp;
	int ** dev_temp_array = (int **)malloc(sizeof(int *)*10);
	for(int i = 0; i < 10; ++i){
		cudaMalloc((void **)&dev_temp, sizeof(int)*10);
		cudaMemcpy(dev_temp, array[i], 10*sizeof(int), cudaMemcpyHostToDevice);
		dev_temp_array[i] = dev_temp;
		cudaMemcpy(&(dev_array[i]), &dev_temp, sizeof(dev_temp), cudaMemcpyHostToDevice);
	}

	dim3 threadsPerBlock(10, 10);
	curandState * d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	setup_kernel<<<1, threadsPerBlock>>>(d_state);
	// testrand<<<1, threadsPerBlock>>>(d_state);
	double testArray[10];
	double resultArray[10];
	double * dev_test_array_src;
	double * dev_test_array_dest;
	cudaMalloc((void**)&dev_test_array_src, sizeof(double)*10);
	cudaMalloc((void**)&dev_test_array_dest, sizeof(double)*10);
	for(double i = 0; i < 10; i+=1){
		testArray[(int)i] = i / 10.0;
	}
	cudaMemcpy(dev_test_array_src, testArray, sizeof(double)*10, cudaMemcpyHostToDevice);
	testmemcpy<<<1, 1>>>(dev_test_array_dest, dev_test_array_src, 10);
	cudaMemcpy(resultArray, dev_test_array_dest, sizeof(double)*10, cudaMemcpyDeviceToHost);

	int * temp;
	for(int i = 0; i < 10; ++i){
		cudaMemcpy(&temp, &(dev_array[i]), sizeof(int *), cudaMemcpyDeviceToHost);
		cudaMemcpy(array[i], (temp), sizeof(int)*10, cudaMemcpyDeviceToHost);
		for(int j = 0; j < 10; ++j){
			printf("%d ", array[i][j]);
		}
		printf("\n");
	}	
	cudaDeviceReset();

	return 0;
}
