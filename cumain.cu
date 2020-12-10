#include <initializer_list>
#include <iostream>
#include <locale>
#include <random>
#include <string>
#include <map>
#include <strings.h>
#include <vector>
#include <ctime>
#include "cuJob.h"
#include "cuMachine.h"
#include "cuChromosome.h"
#include "configure.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

#define TEST

using namespace std;
 

/** Create jobs
 *	 
 *	This function will create jobs and represent them in 2-D array.
 *	Number of rows of this 2-D array dependent on the  number of chromosomes which is specify in parameter @numof_chromosomes
 *	Number of cols of this 2-D array is specify in parameter @numof_jobs
 *	Element of this 2-D array point to a scuJob * object
 *	When creating jobs this function will also collect the can run tool
 *
 *
 *  @param unsigned int nummof_chromosomes
 *  @param nusigned int numof_jobs
 *  @param unsigned int *** canRun_tools is a pointer of 2-D unsigned int array
 *  @param double *** process_time
 *
 *  @return scuJob *** jobs
 */
scuJob ***createJobs(
		unsigned int numof_chromosomes,
		unsigned int numof_jobs,
		unsigned int *** canRun_tools,
		double *** process_time,
		vector<map<string, string> > rows,
		map<string, map<string, int > > eqp_recipe
){
	unsigned int i, j;
	scuJob *** jobs = (scuJob ***)malloc(numof_chromosomes * sizeof(scuJob**));

	// create jobs for first chromosomes;
	jobs[0] = (scuJob **)malloc(numof_jobs * sizeof(scuJob *));
	for(j = 0; j < numof_jobs; ++j){
		jobs[0][j] = createScuJob(j, rows[j], eqp_recipe);
		canRun_tools[0][j] = jobs[0][j]->can_run_tools;
		process_time[0][j] = jobs[0][j]->process_time;
	}

	// clone jobs
	for(i = 1; i < numof_chromosomes; ++i){
		jobs[i] = (scuJob **)malloc(sizeof(scuJob *) * numof_jobs);
		for(j = 0; j < numof_jobs; ++j){
			jobs[i][j] = shared_clone(jobs[0][j]);
		}
	}	


	return jobs;
}

/** Create chromosomes
 *	create a lot of chromsomes
 *  
 *  @param unsigned int NUMOF_CHROMOSOMES
 *  @param unsigned int NUMOF_JOBS
 *
 *  @return schChromosome ** object
 *
 */
scuChromosome ** createChromosomes(
		unsigned int NUMOF_CHROMOSOMES,
		unsigned int NUMOF_JOBS,
		double *** genes_array
){
	scuChromosome ** chrs = (scuChromosome **)malloc(sizeof(scuChromosome *) * NUMOF_CHROMOSOMES);
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		chrs[i] = createScuChromosome(i, NUMOF_JOBS);
		genes_array[0][i] = chrs[i]->dev_genes;
	}
	return chrs;
}


scuMachine *** create_machines(
		unsigned int numof_chromosomes,
		unsigned int numof_machines,
		unsigned int numof_jobs,
		unsigned int **** JOB_ID_LIST,
		vector<vector<string> > statuses
){
	scuMachine *** machines = (scuMachine ***)malloc(sizeof(scuMachine **) * numof_chromosomes);
	unsigned int i, j;
	for(i = 0; i < numof_chromosomes; ++i){
		machines[i] = (scuMachine **)malloc(sizeof(scuMachine *) * numof_machines);
		JOB_ID_LIST[0][i] = (unsigned int **)malloc(sizeof(unsigned int *) * numof_chromosomes);
		for(j = 0; j < numof_machines; ++j){
			machines[i][j] = create_machine(j, statuses[j], numof_jobs);
			JOB_ID_LIST[0][i][j] = machines[i][j]->job_id_list;
		}
	}
	
	return machines;
}

/** Initialize the jobs object parallel
 *  In this kernel, the jobs' shared information will be  binded.
 *  	job.can_run_tools -> CAN_RUN_TOOLS
 *		job.process_time  -> PROCESS_TIME
 * 	In this kernel, the job will also initialize the split value;	
 *
 *	@param scuJob ** jobs which is 2-D array, every row belongs to a chromosome. Number of row of jobs equals to NUMOF_CHROMOSOMES
 *	@param unsigned int ** CAN_RUN_TOOLS is a 2-D array, each row represents a job's can run tools
 *	@param double ** PROCESS_TIME is a 2-D array, each row represents a job's process time in each machine
 *	@param unsigned int NUMOF_JOBS 
 *	@param unsigned int NUMOF_CHROMOSOMES
 */
__global__ void jobs_initialize_kernel(scuJob ** jobs, unsigned int ** CAN_RUN_TOOLS, double ** PROCESS_TIME,unsigned int NUMOF_JOBS,unsigned int NUMOF_CHROMOSOMES){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < NUMOF_CHROMOSOMES && j < NUMOF_JOBS){
		jobs[i][j].can_run_tools = CAN_RUN_TOOLS[j];
		jobs[i][j].process_time = PROCESS_TIME[j];
		jobs[i][j].splitValue = 1.0 / (double)jobs[i][j].sizeof_can_run_tools;
	}
}

__global__ void jobs_binding_chromosome_kernel(scuJob **jobs, double ** chromosomes, unsigned int NUMOF_JOBS, unsigned int NUMOF_CHROMOSOMES){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < NUMOF_CHROMOSOMES && j < NUMOF_JOBS){
		jobs[i][j].ms_gene = &chromosomes[i][j];
		jobs[i][j].os_gene = &chromosomes[i][j+NUMOF_JOBS]; 
	}
}

__global__ void chromosome_initialize_kernel(scuChromosome * chromosomes, curandState * state){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	chromosomes[index].machineSelectionGenes = chromosomes[index].genes;
	chromosomes[index].arrangementGenes = chromosomes[index].genes + chromosomes[index].size / 2;
	for(unsigned int i = 0; i < chromosomes[index].size; ++i){
		chromosomes[index].dev_genes[i]	 = curand_uniform(state + index);
	}
}

__global__ void setup_kernel(curandState * state){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]);
}


__global__ void machine_selection_kernel(scuJob ** jobs, unsigned int NUMOF_JOBS, unsigned int NUMOF_CHROMOSOMES){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int count = 0;
	if( x < NUMOF_CHROMOSOMES && y < NUMOF_JOBS){
		for(double i = jobs[x][y].splitValue; i < 1; i += jobs[x][y].splitValue, ++count){
			if(*jobs[x][y].ms_gene < i){
				jobs[x][y].machine_id = jobs[x][y].can_run_tools[count];
				jobs[x][y].machine_process_time = jobs[x][y].process_time[count];
				break;
			}
		} 
	}
}

__global__ void machine_selection_part2_kernel(scuJob ** jobs, scuMachine ** machines,unsigned int NUMOF_JOBS, unsigned int NUMOF_MACHINES, unsigned int NUMOF_CHROMOSOMES){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = 0;
	if(x < NUMOF_CHROMOSOMES && y < NUMOF_MACHINES){
		machines[x][y].job_size = 0;
		for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
			if(machines[x][y].number == jobs[x][i].machine_id){
				machines[x][y].job_lists[index] = &jobs[x][i];
				++index;
			}
		}
		machines[x][y].job_size = index;
	}
}

__global__ void machine_set_job_list_id(scuMachine ** machines, unsigned int NUMOF_MACHINES, unsigned int NUMOF_CHROMOSOMES){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < NUMOF_CHROMOSOMES && y < NUMOF_MACHINES){
		for(unsigned int i = 0; i < machines[x][y].job_size; ++i){
			machines[x][y].job_id_list[i] = machines[x][y].job_lists[i]->number;
		}
	}
}


int main(int argc, const char * argv[]){
	

	srand(time(NULL));
	map<string, map<string, int> >Data;
	Data = EQP_TIME("./semiconductor-scheduling-data/EQP_RECIPE.txt");
	
	vector< vector<string> >Status;
	Status = vSTATUS("./semiconductor-scheduling-data/Tool.txt");

	vector<map<string, string> > wipData;
	wipData = WIP("./semiconductor-scheduling-data/WIP.txt");

	vector<vector<int> > setup_time;
	setup_time = SETUP_TIME("./semiconductor-scheduling-data/Setup_time.txt");
	
		
	unsigned int NUMOF_CHROMOSOMES = 10;
	unsigned int NUMOF_JOBS = wipData.size();
	unsigned int NUMOF_MACHINES = Status.size();

	unsigned int ** CAN_RUN_TOOLS = (unsigned int**)malloc(sizeof(unsigned *) * NUMOF_JOBS);
	double ** PROCESS_TIME = (double **)malloc(sizeof(double *) * NUMOF_JOBS);
	double ** CHROMOSOMES_2D_ARRAY_POINTER = (double **)malloc(sizeof(double *)*NUMOF_CHROMOSOMES);
	unsigned int *** JOB_ID_LIST = (unsigned int ***)malloc(sizeof(unsigned int **) * NUMOF_CHROMOSOMES);


	scuJob *** jobs = createJobs(
			NUMOF_CHROMOSOMES, 
			NUMOF_JOBS, 
			&CAN_RUN_TOOLS,
			&PROCESS_TIME,
			wipData, Data
	);

	

	
	scuMachine *** machines = create_machines(
			NUMOF_CHROMOSOMES,
			NUMOF_MACHINES,
			NUMOF_JOBS,
			&JOB_ID_LIST,
			Status
	);

	scuChromosome ** chromosomes = createChromosomes(
			NUMOF_CHROMOSOMES,
			NUMOF_JOBS,
			&CHROMOSOMES_2D_ARRAY_POINTER
	);

	// binding CAN_RUN_TOOLS and PROCESS_TIME
	unsigned int ** DEV_CAN_RUN_TOOLS;
	double ** DEV_PROCESS_TIME;

	size_t SIZE_CRT = sizeof(unsigned int *) * NUMOF_JOBS;
	size_t SIZE_PT = sizeof(unsigned int *) * NUMOF_JOBS;

	cudaMalloc((void **)&DEV_CAN_RUN_TOOLS, SIZE_CRT);
	cudaMalloc((void **)&DEV_PROCESS_TIME, SIZE_PT);

	// copy each row of CRT and TIME
	unsigned int *dev_uint_temp;
	double * dev_double_temp;
	size_t size;
	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
		// copy CAN_RUN_TOOLS
		size = sizeof(unsigned int ) * jobs[0][i]->sizeof_can_run_tools;
		cudaMalloc((void**)&dev_uint_temp, size);
		cudaMemcpy(dev_uint_temp, CAN_RUN_TOOLS[i], size, cudaMemcpyHostToDevice); // copy data into dev mem
		cudaMemcpy(&(DEV_CAN_RUN_TOOLS[i]), &dev_uint_temp, sizeof(dev_uint_temp), cudaMemcpyHostToDevice); // copy dev 1-D pointer into 2-D array
		
		// copy PROCESS_TIME
		size = sizeof(double) * jobs[0][i]->sizeof_process_time;
		cudaMalloc((void **)&dev_double_temp, size);
		cudaMemcpy(dev_double_temp, PROCESS_TIME[i],   size, cudaMemcpyHostToDevice);
		cudaMemcpy(&(DEV_PROCESS_TIME[i]), &dev_double_temp, sizeof(dev_double_temp), cudaMemcpyHostToDevice);
	}

#ifdef TEST	
	double * dev_test_double_temp;
	double * double_array;
	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
		// copy back
		size = sizeof(double )*jobs[0][i]->sizeof_process_time;
		double_array = (double*)malloc(size);
		cudaMemcpy(&dev_test_double_temp, &(DEV_PROCESS_TIME[i]), sizeof(double *), cudaMemcpyDeviceToHost);
		cudaMemcpy(double_array, dev_test_double_temp, size, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < jobs[0][i]->sizeof_process_time; ++j){
			printf("%.3f ", double_array[j]);
		}
		printf("\n");
		free(double_array);
	}
	
	unsigned int *dev_test_uint_temp;
	unsigned int *uint_array; 
	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
		// copy back 
		size = sizeof(unsigned int)*jobs[0][i]->sizeof_can_run_tools;
		uint_array = (unsigned int *)malloc(size);
		cudaMemcpy(&dev_test_uint_temp, &(DEV_CAN_RUN_TOOLS[i]), sizeof(int *), cudaMemcpyDeviceToHost);
		cudaMemcpy(uint_array, dev_test_uint_temp,size, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < jobs[0][i]->sizeof_can_run_tools; ++j){
			printf("%d ", uint_array[j]);
		}
		printf("\n");
		free(uint_array);
	}
#endif



	/** Copy job to cuda Memory **/
	
	scuJob ** dev_jobs;
	scuJob *dev_jobs_row;

	cudaMalloc((void**)&dev_jobs, NUMOF_CHROMOSOMES * sizeof(scuJob **));
	
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMalloc((void **)&dev_jobs_row, sizeof(scuJob)*NUMOF_JOBS);
		for(unsigned int j = 0; j < NUMOF_JOBS; ++j){
			cudaMemcpy(&(dev_jobs_row[j]), jobs[i][j], sizeof(scuJob), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(&dev_jobs[i], &dev_jobs_row, sizeof(scuJob *), cudaMemcpyHostToDevice);
	}

#ifdef TEST	
	// scuJob dev_test_row_jobs[NUMOF_JOBS];
	scuJob *dev_test_row_jobs = (scuJob *)malloc(sizeof(scuJob)*NUMOF_JOBS);
	scuJob * dev_row_pointer;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_row_pointer, &dev_jobs[i], sizeof(scuJob *), cudaMemcpyDeviceToHost);	
		cudaMemcpy(dev_test_row_jobs, dev_row_pointer,sizeof(scuJob) * NUMOF_JOBS, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < NUMOF_JOBS; ++j){
			cout<<dev_test_row_jobs[j].number<<" ";
		}
		cout<<endl;
	}
#endif

	//*******************End of copying Jobs**************************//
	
	/** Copy Machine from host to device **/
	scuMachine ** dev_machines;
	scuMachine * dev_machines_row;

	cudaMalloc((void**)&dev_machines, NUMOF_CHROMOSOMES * sizeof(scuMachine **));
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMalloc((void**)&dev_machines_row, sizeof(scuMachine)*NUMOF_MACHINES);
		for(unsigned int j = 0; j < NUMOF_MACHINES; ++j){
			cudaMemcpy(&(dev_machines_row[j]), machines[i][j], sizeof(scuMachine), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(&dev_machines[i], &dev_machines_row, sizeof(scuMachine*), cudaMemcpyHostToDevice);
	}

#ifdef TEST
	scuMachine dev_test_row_machins[NUMOF_MACHINES];
	scuMachine * dev_machine_row_pointer;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_machine_row_pointer, &dev_machines[i], sizeof(scuMachine *), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_machins, dev_machine_row_pointer, sizeof(scuMachine) * NUMOF_MACHINES, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0;j < NUMOF_MACHINES; ++j){
			cout<<dev_test_row_machins[j].number<<" ";	
		}
		cout<<endl;
	}	
#endif

	/********************End of copying Machines*******************/

	
	/** Copy Chromosomes **/
	scuChromosome *dev_chromosomes;
	double ** DEV_CHROMOSOMES_2D_ARRAY_POINTER;
	cudaMalloc((void **)&DEV_CHROMOSOMES_2D_ARRAY_POINTER, sizeof(double *)*NUMOF_CHROMOSOMES);
	cudaMemcpy(DEV_CHROMOSOMES_2D_ARRAY_POINTER, CHROMOSOMES_2D_ARRAY_POINTER, sizeof(double *) * NUMOF_CHROMOSOMES, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_chromosomes, NUMOF_CHROMOSOMES * sizeof(scuChromosome));
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_chromosomes[i], chromosomes[i], sizeof(scuChromosome), cudaMemcpyHostToDevice);
	}
	
	// init chromosome
	curandState * d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	setup_kernel<<<1, NUMOF_CHROMOSOMES>>>(d_state);
	chromosome_initialize_kernel<<<1, NUMOF_CHROMOSOMES>>>(dev_chromosomes, d_state);

#ifndef TEST
	
	double ** test_array = (double **)malloc(sizeof(double *)*NUMOF_CHROMOSOMES);
	cudaMemcpy(test_array, DEV_CHROMOSOMES_2D_ARRAY_POINTER, sizeof(double *) * NUMOF_CHROMOSOMES, cudaMemcpyDeviceToHost);
	double * test_gene = (double *)malloc(sizeof(double) * NUMOF_JOBS*2);
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(test_gene, test_array[i], sizeof(double)*NUMOF_JOBS*2, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < NUMOF_JOBS * 2; ++j){
			printf("%.3f\n", test_gene[j]);
		}
		printf("\n\n");
	}

#endif

#ifndef TEST
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

#endif


	dim3 job_initialize_kernel_thread_dim(16, 16);
	dim3 job_initialize_kernel_block_dim(NUMOF_CHROMOSOMES * NUMOF_JOBS / job_initialize_kernel_thread_dim.x, NUMOF_CHROMOSOMES*NUMOF_JOBS / job_initialize_kernel_thread_dim.y);	
	
	dim3 machine_kernel_thread_dim(16, 16);
	dim3 machine_kernel_block_dim(NUMOF_CHROMOSOMES * NUMOF_MACHINES / machine_kernel_thread_dim.x, NUMOF_CHROMOSOMES * NUMOF_MACHINES / machine_kernel_thread_dim.y);

	jobs_initialize_kernel<<<job_initialize_kernel_block_dim, job_initialize_kernel_thread_dim>>>(dev_jobs, DEV_CAN_RUN_TOOLS, DEV_PROCESS_TIME, NUMOF_JOBS, NUMOF_CHROMOSOMES);
	jobs_binding_chromosome_kernel<<<job_initialize_kernel_block_dim, job_initialize_kernel_thread_dim>>>(dev_jobs, DEV_CHROMOSOMES_2D_ARRAY_POINTER, NUMOF_JOBS, NUMOF_CHROMOSOMES);
	machine_selection_kernel<<<job_initialize_kernel_block_dim, job_initialize_kernel_thread_dim>>>(dev_jobs, NUMOF_JOBS, NUMOF_CHROMOSOMES);
	machine_selection_part2_kernel<<<machine_kernel_block_dim, machine_kernel_block_dim>>>(dev_jobs, dev_machines, NUMOF_JOBS, NUMOF_MACHINES, NUMOF_CHROMOSOMES);
	machine_set_job_list_id<<<machine_kernel_block_dim, machine_kernel_thread_dim>>>(dev_machines, NUMOF_MACHINES, NUMOF_CHROMOSOMES);
	
		
#ifndef TEST	
	unsigned int * test_can_run_tools;
	size_t can_run_tool_size;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_row_pointer, &dev_jobs[i], sizeof(scuJob*), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_jobs, dev_row_pointer, sizeof(scuJob)*NUMOF_JOBS, cudaMemcpyDeviceToHost);
		for(unsigned int j =0; j < NUMOF_JOBS; ++j){
			cout<<dev_test_row_jobs[j].number << " : " <<dev_test_row_jobs[j].splitValue<<" ";
			can_run_tool_size = dev_test_row_jobs[j].sizeof_can_run_tools * sizeof(unsigned int);
			// printf("size = %zu ", can_run_tool_size);
			test_can_run_tools = (unsigned int *)malloc(can_run_tool_size);
			cudaMemcpy(test_can_run_tools, dev_test_row_jobs[j].can_run_tools, can_run_tool_size, cudaMemcpyDeviceToHost);
			for(unsigned int k = 0; k < dev_test_row_jobs[j].sizeof_can_run_tools; ++k){
				cout<<test_can_run_tools[k]<<" ";	
			}
			cout<<endl;
		}
	}
#endif

#ifdef TEST 
	double os_gene;
	double ms_gene;
	size_t can_run_tool_size;
	unsigned int * test_can_run_tools;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_row_pointer, &dev_jobs[i], sizeof(scuJob*), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_jobs, dev_row_pointer, sizeof(scuJob)*NUMOF_JOBS, cudaMemcpyDeviceToHost);
		for(unsigned int j =0; j < NUMOF_JOBS; ++j){
			
			cudaMemcpy(&os_gene, dev_test_row_jobs[j].os_gene, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(&ms_gene, dev_test_row_jobs[j].ms_gene, sizeof(double), cudaMemcpyDeviceToHost);
			cout<<dev_test_row_jobs[j].number << " : "<<dev_test_row_jobs[j].splitValue<<endl;
			printf("ms : %.3f, os : %.3f\n", ms_gene, os_gene);
			can_run_tool_size = dev_test_row_jobs[j].sizeof_can_run_tools * sizeof(unsigned int);
			// printf("size = %zu ", can_run_tool_size);
			test_can_run_tools = (unsigned int *)malloc(can_run_tool_size);
			cudaMemcpy(test_can_run_tools, dev_test_row_jobs[j].can_run_tools, can_run_tool_size, cudaMemcpyDeviceToHost);
			for(unsigned int k = 0; k < dev_test_row_jobs[j].sizeof_can_run_tools; ++k){
				cout<<test_can_run_tools[k]<<" ";	
			}
			cout<<endl;
			printf("Choose Machine = %d, Process Time = %.3f\n\n", dev_test_row_jobs[j].machine_id, dev_test_row_jobs[j].machine_process_time);
		}
	}
#endif


#ifdef TEST
	// scuMachine * dev_machine_row_pointer;
	// scuMachine * dev_machines_row = (scuMachine*)malloc(sizeof(scuMachine) * NUMOF_MACHINES);
	dev_machines_row = (scuMachine*)malloc(sizeof(scuMachine) * NUMOF_MACHINES);
	unsigned int * job_lists;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_machine_row_pointer, &dev_machines[i], sizeof(scuMachine*), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_machines_row, dev_machine_row_pointer, sizeof(scuMachine) * NUMOF_MACHINES, cudaMemcpyDeviceToHost);
		for(unsigned int j = 0; j < NUMOF_MACHINES; ++j){
			printf("Machine Id %d : ", dev_machines_row[j].number);
			job_lists = (unsigned int *)malloc(sizeof(unsigned int) * dev_machines_row[j].job_size);
			cudaMemcpy(job_lists, dev_machines_row[j].job_id_list, sizeof(unsigned int)*dev_machines_row[j].job_size, cudaMemcpyDeviceToHost);
			for(unsigned int k = 0; k < dev_machines_row[j].job_size; ++k){
				printf("%u ", job_lists[k]);
			}
			printf("\n");
		}
		printf("\n\n");
	}	
#endif

	return 0;
}
