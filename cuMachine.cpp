#include "cuMachine.h"
#include "cuJob.h"

scuMachine *create_machine(unsigned int number, std::vector<std::string> status, unsigned int NUMOF_JOB){
	scuMachine * machine = (scuMachine *)malloc(sizeof(scuMachine));
	machine->recover_time = std::stof(status[1]);
	machine->total_time = 0;
	machine->number = number;
	machine->job_size = 9;
	// machine->capacity_of_job = 20;
	scuJob ** job_lists;
	cudaMalloc((void**)&job_lists, sizeof(scuJob *) * NUMOF_JOB);
	unsigned int *job_id_lists;
	cudaMalloc((void**)&job_id_lists, sizeof(unsigned int) * NUMOF_JOB);
	machine->job_lists = job_lists;
	machine->job_id_list = job_id_lists;
	return machine;
}
