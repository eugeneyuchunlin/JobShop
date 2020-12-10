#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

struct test{
	double gene;
	double * OS_gene;
	double test_OS_gene;
	int * canRunTool;	
};

__global__ void pointer(struct test * testStructs, double * gene_array, int size){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size){
		testStructs[index].OS_gene = &gene_array[index];		
	}
}

__global__ void test2(int N){

	int index = threadIdx.x;
	printf("index = %d\n", index);
	printf("CUDA said: hello world");
}


__global__ void testing(struct test *testStructs, int size){
	// int index =	threadIdx.x + blockIdx.x * blockDim.x;
	int index = threadIdx.x;
	printf("index = %d\n", index);
	// printf("gene = %d\n", testStructs[index].gene);
	if(index < size){
		for(int i = 0; i < 10; ++i){
			testStructs[index].canRunTool[i] += 1;
		}
		testStructs[index].test_OS_gene = *testStructs[index].OS_gene;
	}
}

int main(int argc, char const * argv[]){
	int structSize = 10;
	int numBytes = structSize * sizeof(struct test);
	printf("sizeof = %d\n", sizeof(struct test)); 
	int canRunTools = 10;
	int canRunToolsBytes = canRunTools * sizeof(int);
	struct test * stests = (struct test *)malloc(numBytes);
	int i, j;
	

	for(i = j = 0; i < structSize; ++i){
		stests[i].canRunTool = (int *)malloc(canRunToolsBytes);
		for(int k = 0;k < canRunTools; ++j, ++k){
			stests[i].canRunTool[k] = j;
			stests[i].gene = i;
			printf("%d ", j);
		}
		printf("\n");
	}
	

	struct test * dev_tests;
	cudaMalloc((void **) &dev_tests, numBytes);
	cudaMemcpy(dev_tests, stests, numBytes, cudaMemcpyHostToDevice);
	struct test * temp_tests = (struct test*)malloc(numBytes);
	cudaMemcpy(temp_tests, dev_tests, numBytes, cudaMemcpyDeviceToHost);
	for(int i = 0; i < structSize; ++i){
		printf("49 gene = %.0f\n", temp_tests[i].gene); 
	}
	
	int * dev_canRunToolsTemp;
	int ** temps = (int **)malloc(structSize*sizeof(int *));
	for(i = 0; i < structSize; ++i){
		cudaMalloc((void **)&dev_canRunToolsTemp, canRunToolsBytes);		
		temps[i] = dev_canRunToolsTemp;
		cudaMemcpy(dev_canRunToolsTemp, stests[i].canRunTool,canRunToolsBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(&(dev_tests[i].canRunTool), &dev_canRunToolsTemp, sizeof(dev_canRunToolsTemp), cudaMemcpyHostToDevice);	
	}

	dim3 threadsPerBlock(2, 2);
	dim3 numBlocks(structSize / threadsPerBlock.x, structSize / threadsPerBlock.y);

	double gene_array[10];
	for(int i = 0; i < 10; ++i){
		gene_array[i] = (double)i / 10.0;
	}

	double *dev_gene_array;
	cudaMalloc((void**)&dev_gene_array, sizeof(double)*10);
	cudaMemcpy(dev_gene_array, gene_array,sizeof(double)*10, cudaMemcpyHostToDevice);

	test2<<<1, structSize>>>(5);	
	pointer<<<1, structSize>>>(dev_tests, dev_gene_array, structSize);
	
	testing<<<1, structSize>>>(dev_tests, structSize);
	// testing<<<numBlocks, threadsPerBlock>>>(dev_tests, 4);
	
	int * host_canRunToolsTemp;
	host_canRunToolsTemp = (int*)malloc(canRunToolsBytes);
	cudaMemcpy(stests, dev_tests, numBytes, cudaMemcpyDeviceToHost);
	for(i = 0; i < structSize; ++i){
		printf("%.3f\n", stests[i].test_OS_gene);
	}

}
