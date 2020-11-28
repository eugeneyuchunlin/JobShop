#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <nvrtc.h>
#include <nvrtc_helper.h>


int main(int argc, char **argv){
	char *cubin, *kernel_file;
	size_t cubinSize;
	kernel_file = sdkFindFilePath("test_2D_array.cu", argv[0]);
	compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);
	CUmodule module = loadCUBIN(cubin, argc, argv);
	CUfunction kernel_addr;
	checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "vectorAdd"));
}
