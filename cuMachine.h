#ifndef __CU_MACHINE_H__
#define __CU_MACHINE_H__

#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuJob.h"


struct scuMachine{
	double recover_time;
	double total_time;
	unsigned int number;
	unsigned int job_size;
	unsigned int *job_id_list;
	struct scuJob ** job_lists;	
};

scuMachine *create_machine(unsigned int, std::vector<std::string>, unsigned int NUMOF_JOB);

#endif
