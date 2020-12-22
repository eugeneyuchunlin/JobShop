#include "unit_test.h"
#include "cuChromosome.h"
#include <assert.h>
#include "cuJob.h"
#include "cuMachine.h"
#include <cuda.h>
#include <cuda_runtime.h>

bool test_can_run_tools_successfully_copy(unsigned int ** CAN_RUN_TOOLS, unsigned int ** DEV_CAN_RUN_TOOLS, scuJob***jobs, unsigned int NUMOF_JOBS){
	unsigned int *dev_test_unit_temp;
	unsigned int *uint_array;
	size_t size;
	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
		size = sizeof(unsigned int)*jobs[0][i]->sizeof_can_run_tools;
		uint_array = (unsigned int *)malloc(size);
		cudaMemcpy(&dev_test_unit_temp, &(DEV_CAN_RUN_TOOLS[i]), sizeof(unsigned int *), cudaMemcpyDeviceToHost);
		cudaMemcpy(uint_array, dev_test_unit_temp,size, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < jobs[0][i]->sizeof_can_run_tools; ++j){
			assert(uint_array[j] == CAN_RUN_TOOLS[i][j]);
		}
	}

	return true;
}


bool test_process_time_successfully_copy(double ** PROCESS_TIME, double ** DEV_PROCESS_TIME, scuJob ***jobs, unsigned int NUMOF_JOBS){
	double * dev_test_double_temp;
	double * double_array;
	size_t size;
	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
		size = sizeof(double)*jobs[0][i]->sizeof_process_time;
		double_array = (double *)malloc(size);
		cudaMemcpy(&dev_test_double_temp, &(DEV_PROCESS_TIME[i]), sizeof(double *), cudaMemcpyDeviceToHost);
		cudaMemcpy(double_array, dev_test_double_temp, size, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < jobs[0][i]->sizeof_process_time; ++j){
			assert(double_array[j] == PROCESS_TIME[i][j]);
		}
		free(double_array);
	}
	return true;
}


bool test_jobs_are_the_same(scuJob * job1, scuJob * job2){
	assert(job1->number == job2->number);
	assert(job1->sizeof_can_run_tools == job2->sizeof_can_run_tools);
	assert(job1->sizeof_process_time == job2->sizeof_process_time);
	return true;	
}

bool test_job_successfully_copy(scuJob *** jobs, scuJob ** dev_jobs, unsigned int NUMOF_CHROMOSOMES, unsigned int NUMOF_JOBS){
	scuJob * dev_test_row_jobs = (scuJob *)malloc(sizeof(scuJob)*NUMOF_JOBS);
	scuJob * dev_row_pointer;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_row_pointer, &dev_jobs[i], sizeof(scuJob *), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_jobs, dev_row_pointer, sizeof(scuJob) * NUMOF_JOBS, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < NUMOF_JOBS; ++j){
			printf("%d, %d\n",jobs[i][j]->number, dev_test_row_jobs[j].number);
			test_jobs_are_the_same(jobs[i][j], &dev_test_row_jobs[j]);
		}
	}
	free(dev_test_row_jobs);
	return true;
}

bool test_machines_are_the_same(scuMachine * m1, scuMachine * m2){
	if(m1->recover_time != m2->recover_time) return false;
	if(m1->number != m2->number) return false;
	return true;
}


bool test_machines_successfully_copy(
		scuMachine *** machines,
		scuMachine ** dev_machines,
		unsigned int NUMOF_MACHINES,
		unsigned int NUMOF_CHROMOSOMES
){
	scuMachine dev_test_row_machins[NUMOF_MACHINES];
	scuMachine * dev_machine_row_pointer;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_machine_row_pointer, &dev_machines[i], sizeof(scuMachine *), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_machins, dev_machine_row_pointer, sizeof(scuMachine) * NUMOF_MACHINES, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0;j < NUMOF_MACHINES; ++j){
			test_machines_are_the_same(machines[i][j], &dev_test_row_machins[j]);
		}
	}
	
	return true;	
}

bool test_chromosome_initialize(scuChromosome * dev_chromosomes, unsigned int NUMOF_JOBS, unsigned int NUMOF_CHROMOSOMES){
	scuChromosome *dev_test_chromosomes = (scuChromosome *)malloc(sizeof(scuChromosome) * NUMOF_CHROMOSOMES);
	double * gene = (double *)malloc(sizeof(double) * NUMOF_JOBS * 2);
	cudaMemcpy(dev_test_chromosomes, dev_chromosomes, sizeof(scuChromosome) * NUMOF_CHROMOSOMES, cudaMemcpyDeviceToHost);
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		printf("Chromosome %d : %d\n", dev_test_chromosomes[i].number, dev_test_chromosomes[i].size);
		cudaMemcpy(gene, dev_test_chromosomes[i].dev_genes, sizeof(double)*NUMOF_JOBS*2, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < NUMOF_JOBS * 2; ++j){
			printf("%.8f\n", gene[j]);
		}
		printf("\n\n");
	}
	return true;
}

void test_copying_genes(scuChromosome * dev_chromosomes, double ** dev_genes, double ** host_genes, unsigned int NUMOF_JOBS, unsigned int NUMOF_CHROMOSOMES){
	scuChromosome * dev_test_chromosomes = (scuChromosome *)malloc(sizeof(scuChromosome) * NUMOF_CHROMOSOMES);
	double * gene = (double *)malloc(sizeof(double)*NUMOF_JOBS*2);
	cudaMemcpy(dev_test_chromosomes, dev_chromosomes, sizeof(scuChromosome)*NUMOF_CHROMOSOMES, cudaMemcpyDeviceToHost);
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(host_genes[i], dev_genes[i], sizeof(double)*NUMOF_JOBS*2, cudaMemcpyDeviceToHost);
		cudaMemcpy(gene, dev_test_chromosomes[i].dev_genes, sizeof(double)*NUMOF_JOBS*2, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0 ;j < NUMOF_JOBS * 2; ++j){
			assert(gene[j] == host_genes[i][j]);
		}
	}	
}
