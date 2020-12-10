#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
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
	test<<<1, threadsPerBlock>>>(dev_array);


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
