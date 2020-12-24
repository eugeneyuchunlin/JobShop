#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <locale>
#include <pthread.h>
#include <random>
#include <string>
#include <map>
#include <strings.h>
#include <type_traits>
#include <vector>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "cuJob.h"
#include "cuMachine.h"
#include "cuChromosome.h"
#include "configure.h"
#include "unit_test.h"

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
 *  @param vector<map<string, string> > rows
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
		double *** genes_dev_array,
		double *** genes_host_array
){
	scuChromosome ** chrs = (scuChromosome **)malloc(sizeof(scuChromosome *) * NUMOF_CHROMOSOMES);
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		chrs[i] = createScuChromosome(i, NUMOF_JOBS);
		genes_dev_array[0][i] = chrs[i]->dev_genes;
		genes_host_array[0][i] = chrs[i]->genes;
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
		jobs[i][j].last = jobs[i][j].next=  NULL;
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
				jobs[x][y].last = jobs[x][y].next = NULL;
				break;
			}
		} 
	}
}

__global__ void machine_selection_part2_kernel(scuJob ** jobs, scuMachine ** machines, unsigned int NUMOF_JOBS, unsigned int NUMOF_MACHINES, unsigned int NUMOF_CHROMOSOMES){
	unsigned int index = 0;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x < NUMOF_CHROMOSOMES && y < NUMOF_MACHINES){
		// for(unsigned int i = 0; i < machines[x][y].job_size; ++i){
		// 	machines[x][y].job_id_list[i] = machines[x][y].job_lists[i]->number;
		// }
		for(int i = 0; i < NUMOF_JOBS; ++i){
			if(machines[x][y].number == jobs[x][i].machine_id){
				machines[x][y].job_lists[index] = &(jobs[x][i]);
				++index;
			}
		}
		machines[x][y].job_size = index;
	}
	// if(x < NUMOF_CHROMOSOMES && y < NUMOF_MACHINES){
	// 	machines[x][y].job_size = 0;
	// 	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
	// 		if(machines[x][y].number == jobs[x][i].machine_id){
	// 			machines[x][y].job_lists[index] = &jobs[x][i];
	// 			++index;
	// 		}
	// 	}
	// 	machines[x][y].job_size = 2;
	// 	// printf("index = %d\n", index);
	// }
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

__device__ void setup_startT_endT(double recover_time,unsigned int size,  scuJob ** jobList, unsigned int ** SETUP_TIME){
	// setup start time and end time
	double lastFinishedTime, setup_t;
	double temp_t;
	scuJob * temp;
	unsigned int i;
	jobList[0]->start_time = (jobList[0]->arrive_t >= recover_time ? jobList[0]->arrive_t : recover_time);
	jobList[0]->end_time = jobList[0]->start_time + jobList[0]->machine_process_time;
	lastFinishedTime = jobList[0]->end_time;
	for(i = 1; i < size; ++i){
		temp = jobList[i - 1];
		setup_t = SETUP_TIME[temp->number][jobList[i]->number];
		temp_t = lastFinishedTime + setup_t;
		jobList[i]->start_time = (jobList[i]->arrive_t > temp_t ? jobList[i]->arrive_t : temp_t);
		lastFinishedTime = jobList[i]->end_time = jobList[i]->start_time + jobList[i]->machine_process_time;
	}
}


__global__ void machine_sort_the_jobs(scuMachine ** machines, unsigned int ** SETUP_TIME ,unsigned int NUMOF_MACHINES, unsigned int NUMOF_CHROMOSOMES){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	scuJob ** jobList;
	scuJob * temp;
	int size;
	int i, j, k;
	double lastFinishedTime;
	unsigned int setup_t;
	double temp_t;
	if(x < NUMOF_CHROMOSOMES && y < NUMOF_MACHINES){
		jobList = machines[x][y].job_lists;
		size = machines[x][y].job_size;

		for(i = 0; i < size - 1; ++i){
			for(j = 0; j < size - 1; ++j){
				// if(jobList[j]->number > jobList[j+1]->number){
				if(*(jobList[j]->os_gene) < *(jobList[j + 1]->os_gene)){ // swap 
					temp = jobList[j];
					jobList[j] = jobList[j + 1];
					jobList[j + 1] = temp;
				}
			}
		}
		

		
		if(size > 0){
			// setup start time and end time
			jobList[0]->start_time = (jobList[0]->arrive_t >= machines[x][y].recover_time ? jobList[0]->arrive_t : machines[x][y].recover_time);
			jobList[0]->end_time = jobList[0]->start_time + jobList[0]->machine_process_time;
			lastFinishedTime = jobList[0]->end_time;
			for(i = 1; i < size; ++i){
				temp = jobList[i - 1];
				setup_t = SETUP_TIME[temp->number][jobList[i]->number];
				temp_t = lastFinishedTime + setup_t;
				jobList[i]->start_time = (jobList[i]->arrive_t > temp_t ? jobList[i]->arrive_t : temp_t);
				lastFinishedTime = jobList[i]->end_time = jobList[i]->start_time + jobList[i]->machine_process_time;
			}
			machines[x][y].total_time = lastFinishedTime;
		}
		// double interval_t;
		// if(size > 1){
		// 	for(i = 1;	i < size; ++i){
		// 		interval_t = jobList[0]->start_time - machines[x][y].recover_time; // recover time <-> jobList[0]->start_time
		// 		lastFinishedTime = machines[x][y].recover_time;
		// 		for(j = 0; j <= i; ++j){
		// 			if(jobList[i]->arrive_t >= lastFinishedTime && jobList[i]->arrive_t < jobList[j]->start_time){ // check if jobList[i] is avaliable at the select interval
		// 				if(lastFinishedTime + jobList[i]->machine_process_time < jobList[j]->start_time){ // check if jobList[i] can be put into the interval
		// 					// revise the sequence
		// 					temp = jobList[i]; 
		// 					for(k = i - 1; k >= j; --k){
		// 						jobList[k + 1] = jobList[k];
		// 					}
		// 					jobList[j] = temp;

		// 				}
		// 			}	
		// 		}
		// 	}
		// }
		

	}	
}

__global__ void machine_clear_kernel(scuMachine ** machines, unsigned int NUMOF_MACHINES, unsigned int NUMOF_CHROMOSOMES){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < NUMOF_CHROMOSOMES && y < NUMOF_MACHINES){
		machines[x][y].job_size = 0;
	}
	
}

__global__ void mutation_kernel(
		scuChromosome * chromosomes,
		unsigned int * parent,
		unsigned int * position,
		double * genes,
		unsigned int NUMOF_MUTATIONS,
		unsigned int START
){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int p, pos;
	unsigned int child;
	if(idx < NUMOF_MUTATIONS){
		p = parent[idx];
		pos = position[idx];
		child = idx + START;
		memcpy(chromosomes[child].dev_genes, chromosomes[p].dev_genes, sizeof(double)*chromosomes[p].size);
		chromosomes[child].dev_genes[pos] = genes[idx];
	}
}


__global__ void crossover_kernel(
		scuChromosome * chromosomes, 
		unsigned int * parent1, 
		unsigned int * parent2, 
		unsigned int * cutPoints, 
		unsigned int * size, 
		unsigned int NUMOF_CROSOVRS, 
		unsigned int START
){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int p1, p2;
	unsigned int c1, c2;
	unsigned int cut, range;
	if(idx < NUMOF_CROSOVRS){
		p1 = parent1[idx];
		p2 = parent2[idx];
		
		c1 = idx * 2 + START;
		c2 = idx * 2 + START + 1;

		cut = cutPoints[idx];
		range = size[idx];

		// copy data from parent
		memcpy(chromosomes[c1].dev_genes, chromosomes[p1].dev_genes, sizeof(double)*chromosomes[c1].size);
		memcpy(chromosomes[c2].dev_genes, chromosomes[p2].dev_genes, sizeof(double)*chromosomes[c2].size);
		
		// crossover
		memcpy(chromosomes[c1].dev_genes + cut, chromosomes[p2].dev_genes + cut, sizeof(double)*range);
		memcpy(chromosomes[c2].dev_genes + cut, chromosomes[p1].dev_genes + cut, sizeof(double)*range);
		
	}
}

__global__ void setup_fitness_value(
		scuMachine ** machines,
		scuChromosome * chromosomes,
		unsigned int NUMOF_MACHINES,
		unsigned int NUMOF_CHROMOSOMES
){
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	double max = 0;
	if(x < NUMOF_CHROMOSOMES){
		for(unsigned int i = 0; i < NUMOF_MACHINES; ++i){
			if(machines[x][i].total_time > max){
				max = machines[x][i].total_time;
			}		
		}	
		chromosomes[x].fitnessValue = max;
	}
}

__global__ void sort_chromosomes(
		scuChromosome * chromosomes,
		unsigned int NUMOF_CHROMOSOMES
){
	int i, j;
	scuChromosome *temp;
	for(i = 0; i < NUMOF_CHROMOSOMES - 1; ++i){
		for(j = 0; j < NUMOF_CHROMOSOMES - 1; ++j){
			if(chromosomes[j].fitnessValue > chromosomes[j].fitnessValue){
				temp = &chromosomes[j];
				chromosomes[j] = chromosomes[j + 1];
				chromosomes[j + 1] = *temp;
			}
		}
	}
	
}

int random_int(int start, int end, int different_num){
	if(different_num < 0){
		return start + rand() % (end - start);
	}else{
		int rnd = start + (rand() % (end - start));

		while(rnd == different_num){
			rnd = start + (rand() % (end + start));
		}
		return rnd;
	}
}

float random(unsigned int end){
	return (float)rand()/((float)RAND_MAX/end);
}

void copy_chromosome_content_from_device_to_host(double ** dev_genes, double ** host_genes,unsigned int NUMOF_JOBS, unsigned int NUMOF_CHROMOSOMES){
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i)
		cudaMemcpy(host_genes[i], dev_genes[i], sizeof(double)*NUMOF_JOBS*2, cudaMemcpyDeviceToHost);
}

void copy_chromosome_content_from_host_to_device(double ** dev_genes, double ** host_genes,unsigned int NUMOF_JOBS, unsigned int NUMOF_CHROMOSOMES){
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i)
		cudaMemcpy(dev_genes[i], host_genes[i], sizeof(double)*NUMOF_JOBS*2, cudaMemcpyHostToDevice);
}


void crossover_core(scuChromosome * c1, scuChromosome * c1Child, scuChromosome * c2, scuChromosome * c2Child){
	memcpy(c1Child->genes, c1->genes, sizeof(double)*c1->size);
	memcpy(c2Child->genes, c2->genes, sizeof(double)*c2->size);
	unsigned int cut1, cut2, temp;
	cut1 = random_int(0, c1->size, -1);
	cut2 = random_int(0, c2->size, cut1);
	
	if(cut1 < cut2){
		temp = cut1;
		cut1 = cut2;
		cut2 = temp;
	}
	temp = cut1 - cut2;

	memcpy(c1Child->genes + cut1, c2->genes + cut1, sizeof(double)*(temp));
	memcpy(c2Child->genes + cut1, c1->genes + cut1, sizeof(double)*(temp)); 
}

unsigned int crossover(scuChromosome ** chromosomes, unsigned int AMOUNT, unsigned int START, unsigned int NUMOF_CHROMOSOMES){
	scuChromosome * c1, * c2;
	unsigned int m, n;
	unsigned int index = START;
	for(unsigned int i = 0; i < AMOUNT; i+=2){
		m = random_int(0, NUMOF_CHROMOSOMES, -1);
		n = random_int(0, NUMOF_CHROMOSOMES, m);
		c1 = chromosomes[m];
		c2 = chromosomes[n];
		crossover_core(c1, chromosomes[index], c2, chromosomes[index + 1]);	
		index += 2;
	}
	return START + AMOUNT;
}

void mutation_core(scuChromosome * c, scuChromosome * child){
	unsigned int point = random_int(0, c->size, -1);
	memcpy(child->genes, c->genes, sizeof(double)*c->size);
	child->genes[point] = random(1);
}

unsigned int mutation(scuChromosome ** chromosomes, unsigned int AMOUNT, unsigned int START, unsigned int NUMOF_CHROMOSOMES){
	unsigned int index = START;
	unsigned int m;
	for(unsigned int i = 0; i < AMOUNT; ++i){
		m = random_int(0, NUMOF_CHROMOSOMES, -1);
		mutation_core(chromosomes[m], chromosomes[index]);
		++index;
	}
	return START + AMOUNT;
}

void initialize_crossover_vectors(
		unsigned int ** parent1,
		unsigned int ** parent2,
		unsigned int ** cutpoints,
		unsigned int ** size,
		unsigned int PARENT_SIZE
){
	*parent1 = (unsigned int*)malloc(sizeof(unsigned int)*PARENT_SIZE);
	*parent2 = (unsigned int *)malloc(sizeof(unsigned int)*PARENT_SIZE);
	*cutpoints = (unsigned int*)malloc(sizeof(unsigned int)*PARENT_SIZE);
	*size = (unsigned int*)malloc(sizeof(unsigned int)*PARENT_SIZE);
}

void initialize_dev_crossover_vectors(
		unsigned int ** parent1,
		unsigned int ** parent2,
		unsigned int ** cutpoints,
		unsigned int ** size,
		unsigned int PARENT_SIZE
){
	size_t mem_size = sizeof(unsigned int)*PARENT_SIZE;		
	cudaMalloc((void**)parent1, mem_size);
	cudaMalloc((void**)parent2, mem_size);
	cudaMalloc((void**)cutpoints, mem_size);
	cudaMalloc((void**)size, mem_size);
}

void generate_crossover_vectors(
		unsigned int *parent1,
		unsigned int *parent2,
		unsigned int *cut,
		unsigned int *size,
		unsigned int NUMOF_CHROMOSOMES,
		unsigned int AMOUNT
){
	unsigned int m, n;
	unsigned int cut1, cut2, temp;
	for(unsigned int i = 0; i < AMOUNT; ++i){
		m = random_int(0, NUMOF_CHROMOSOMES, -1);
		n = random_int(0, NUMOF_CHROMOSOMES, m);
		parent1[i] = m;
		parent2[i] = n;
		
		cut1 = random_int(0, NUMOF_CHROMOSOMES, -1);
		cut2 = random_int(0, NUMOF_CHROMOSOMES, cut1);
		if(cut1 < cut2){
			temp = cut1;
			cut1 = cut2;
			cut2 = temp;
		}
		cut[i] = cut2;
		size[i] = cut1 - cut2;
	}	

}

void initialize_mutation_vectors(
		unsigned int ** parent,
		unsigned int ** position,
		double ** genes,
		unsigned int NUMOF_MUTATIONS
){
	*parent = (unsigned int *)malloc(sizeof(unsigned int)*NUMOF_MUTATIONS);
	*position = (unsigned int *)malloc(sizeof(unsigned int)*NUMOF_MUTATIONS);
	*genes = (double*)malloc(sizeof(double)*NUMOF_MUTATIONS);
}

void initialize_device_mutation_vectors(
		unsigned int ** parent,
		unsigned int ** position,
		double ** genes,
		unsigned int NUMOF_MUTATIONS
){
	cudaMalloc((void**)parent, sizeof(unsigned int)*NUMOF_MUTATIONS);
	cudaMalloc((void**)position, sizeof(unsigned int)*NUMOF_MUTATIONS);
	cudaMalloc((void**)genes, sizeof(double)*NUMOF_MUTATIONS);
}



void generate_mutation_vectors(
		unsigned int * parent,
		unsigned int * position,
		double * genes,
		unsigned int NUMOF_JOBS,
		unsigned int NUMOF_CHROMOSOMES,
		unsigned int NUMOF_MUTATIONS
){
	for(unsigned int i = 0; i < NUMOF_MUTATIONS; ++i){
		parent[i] = random_int(0, NUMOF_CHROMOSOMES, -1);
		position[i] = random_int(0, NUMOF_JOBS*2, -1);
		genes[i] = random(1);
	}		
}

unsigned int ** convert_vector_setup_time_to_array(
		vector<vector<int> > setup_time
){
	unsigned int ** array = (unsigned int **)malloc(sizeof(unsigned int *)*setup_time.size());
	for(unsigned int i = 0; i < setup_time.size(); ++i){
		array[i] = (unsigned int *)malloc(sizeof(unsigned int)*setup_time[i].size());
		for(unsigned int j = 0; j < setup_time[i].size(); ++j){
			array[i][j] = setup_time[i][j];
		}
	}
	return array;
}

void copy_setup_time(unsigned int ** host, unsigned int ** dev_array);


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
		
	const unsigned int NUMOF_CHROMOSOMES = atoi(argv[1]);
	const unsigned int MAX_NUMOF_CHROMOSOMES =  2 * NUMOF_CHROMOSOMES;
	const unsigned int NUMOF_JOBS = wipData.size();
	const unsigned int NUMOF_MACHINES = Status.size();
	

	unsigned int ** HOST_SETUP_TIME = convert_vector_setup_time_to_array(setup_time);
	unsigned int ** CAN_RUN_TOOLS = (unsigned int**)malloc(sizeof(unsigned *) * NUMOF_JOBS);
	double ** PROCESS_TIME = (double **)malloc(sizeof(double *) * NUMOF_JOBS);
	double ** CHROMOSOMES_2D_DEV_ARRAY_POINTER = (double **)malloc(sizeof(double *)*MAX_NUMOF_CHROMOSOMES);
	double ** CHROMOSOMES_2D_HOST_ARRAY_POINTER = (double **)malloc(sizeof(double *)*MAX_NUMOF_CHROMOSOMES);
	unsigned int *** JOB_ID_LIST = (unsigned int ***)malloc(sizeof(unsigned int **) * MAX_NUMOF_CHROMOSOMES);


	scuJob *** jobs = createJobs(
			MAX_NUMOF_CHROMOSOMES, 
			NUMOF_JOBS, 
			&CAN_RUN_TOOLS,
			&PROCESS_TIME,
			wipData, Data
	);

	

	
	scuMachine *** machines = create_machines(
			MAX_NUMOF_CHROMOSOMES,
			NUMOF_MACHINES,
			NUMOF_JOBS,
			&JOB_ID_LIST,
			Status
	);

	scuChromosome ** chromosomes = createChromosomes(
			MAX_NUMOF_CHROMOSOMES,
			NUMOF_JOBS,
			&CHROMOSOMES_2D_DEV_ARRAY_POINTER,
			&CHROMOSOMES_2D_HOST_ARRAY_POINTER
	);

	// copy setup time
	unsigned int ** DEV_SETUP_TIME;
	unsigned int * setup_temp;
	cudaMalloc((void**)&DEV_SETUP_TIME, sizeof(unsigned int *)*NUMOF_JOBS); 
	for(unsigned int i = 0; i < NUMOF_JOBS; ++i){
		cudaMalloc((void**)&setup_temp, sizeof(unsigned int)*NUMOF_JOBS);
		cudaMemcpy(setup_temp, HOST_SETUP_TIME[i], sizeof(unsigned int)*NUMOF_JOBS, cudaMemcpyHostToDevice);
		cudaMemcpy(&(DEV_SETUP_TIME[i]), &setup_temp, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	}

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
	if(test_can_run_tools_successfully_copy(CAN_RUN_TOOLS, DEV_CAN_RUN_TOOLS, jobs, NUMOF_JOBS)) printf("pass test can run tools successfully copy !\n");
	if(test_process_time_successfully_copy(PROCESS_TIME, DEV_PROCESS_TIME, jobs, NUMOF_JOBS)) printf("pass test process time successfully copy !\n");
	if(test_setupt_time_successfully_copied(HOST_SETUP_TIME, DEV_SETUP_TIME, NUMOF_JOBS)) printf("pass test copy setup time!\n");
#endif


	/** Copy job to cuda Memory **/
	scuJob ** dev_jobs;
	scuJob *dev_jobs_row;
	cudaMalloc((void**)&dev_jobs, MAX_NUMOF_CHROMOSOMES * sizeof(scuJob **));
	for(unsigned int i = 0; i < MAX_NUMOF_CHROMOSOMES; ++i){
		cudaMalloc((void **)&dev_jobs_row, sizeof(scuJob)*NUMOF_JOBS);
		for(unsigned int j = 0; j < NUMOF_JOBS; ++j){
			cudaMemcpy(&(dev_jobs_row[j]), jobs[i][j], sizeof(scuJob), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(&dev_jobs[i], &dev_jobs_row, sizeof(scuJob *), cudaMemcpyHostToDevice);
	}

#ifdef TEST	
	if(test_job_successfully_copy(jobs, dev_jobs, MAX_NUMOF_CHROMOSOMES, NUMOF_JOBS))printf("pass test job successfully copy !\n");
#endif
	//*******************End of copying Jobs**************************//
	


	/** Copy Machine from host to device **/
	scuMachine ** dev_machines;
	scuMachine * dev_machines_row;

	cudaMalloc((void**)&dev_machines, sizeof(scuMachine **)*MAX_NUMOF_CHROMOSOMES);

	for(unsigned int i = 0; i < MAX_NUMOF_CHROMOSOMES; ++i){
		cudaMalloc((void**)&dev_machines_row, sizeof(scuMachine)*NUMOF_MACHINES);
		for(unsigned int j = 0; j < NUMOF_MACHINES; ++j){
			cudaMemcpy(&(dev_machines_row[j]), machines[i][j], sizeof(scuMachine), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(&dev_machines[i], &dev_machines_row, sizeof(scuMachine*), cudaMemcpyHostToDevice);
	}

#ifdef TEST
	if(test_machines_successfully_copy(machines, dev_machines, NUMOF_MACHINES, MAX_NUMOF_CHROMOSOMES)) printf("pass test machines successfully copied!\n");
	else exit(-1);
#endif
	/********************End of copying Machines*******************/
	


	/** Copy Chromosomes **/
	scuChromosome *dev_chromosomes;
	double ** DEV_CHROMOSOMES_2D_ARRAY_POINTER;
	cudaMalloc((void **)&DEV_CHROMOSOMES_2D_ARRAY_POINTER, sizeof(double *)*MAX_NUMOF_CHROMOSOMES);
	cudaMemcpy(DEV_CHROMOSOMES_2D_ARRAY_POINTER, CHROMOSOMES_2D_DEV_ARRAY_POINTER, sizeof(double *) * MAX_NUMOF_CHROMOSOMES, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_chromosomes, MAX_NUMOF_CHROMOSOMES * sizeof(scuChromosome));
	for(unsigned int i = 0; i < MAX_NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_chromosomes[i], chromosomes[i], sizeof(scuChromosome), cudaMemcpyHostToDevice);
	}
	
	// init chromosome
	curandState * d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	setup_kernel<<<1, MAX_NUMOF_CHROMOSOMES>>>(d_state);
	chromosome_initialize_kernel<<<MAX_NUMOF_CHROMOSOMES / 20, 20>>>(dev_chromosomes, d_state);

#ifdef TEST
	test_chromosome_initialize(dev_chromosomes, NUMOF_JOBS, MAX_NUMOF_CHROMOSOMES);
	// test_copying_genes(dev_chromosomes, CHROMOSOMES_2D_DEV_ARRAY_POINTER, CHROMOSOMES_2D_HOST_ARRAY_POINTER, NUMOF_JOBS, MAX_NUMOF_CHROMOSOMES);
#endif
	
	dim3 job_initialize_kernel_thread_dim(16, 16);
	dim3 job_initialize_kernel_block_dim(60, 60);
	dim3 machine_kernel_thread_dim(16, 16);
	dim3 machine_kernel_block_dim(60, 60);

	jobs_initialize_kernel<<<job_initialize_kernel_block_dim, job_initialize_kernel_thread_dim>>>(dev_jobs, DEV_CAN_RUN_TOOLS, DEV_PROCESS_TIME, NUMOF_JOBS, MAX_NUMOF_CHROMOSOMES);
	jobs_binding_chromosome_kernel<<<job_initialize_kernel_block_dim, job_initialize_kernel_thread_dim>>>(dev_jobs, DEV_CHROMOSOMES_2D_ARRAY_POINTER, NUMOF_JOBS, MAX_NUMOF_CHROMOSOMES);
	
	// unsigned int START_INDEX;
	unsigned int CROSOVR_RATE = NUMOF_CHROMOSOMES * 0.6;
	unsigned int MUT_RATE = NUMOF_CHROMOSOMES * 0.4;
	unsigned int * parent1, *parent2, *dev_parent1, *dev_parent2;
	unsigned int * cutPoints,*crossover_range, *dev_cutPoints, *dev_crossover_range;
	initialize_crossover_vectors(&parent1, &parent2, &cutPoints, &crossover_range, CROSOVR_RATE);
	initialize_dev_crossover_vectors(&dev_parent1, &dev_parent2, &dev_cutPoints, &dev_crossover_range, CROSOVR_RATE);
	size_t cro_cpySize = sizeof(unsigned int)*CROSOVR_RATE;

	size_t mut_cpySize = sizeof(unsigned int)*MUT_RATE;
	unsigned int * parent, *position, *dev_parent, *dev_position;
	double * replace_genes, *dev_replace_genes;
	initialize_mutation_vectors(&parent, &position, &replace_genes, MUT_RATE);
	initialize_device_mutation_vectors(&dev_parent, &dev_position, &dev_replace_genes, MUT_RATE);


	clock_t start = clock();
	for(unsigned int i = 0; i < 1000; ++i){
		machine_clear_kernel<<<machine_kernel_block_dim, machine_kernel_thread_dim>>>(dev_machines, NUMOF_MACHINES, MAX_NUMOF_CHROMOSOMES);
		
		// generate crossover data
		generate_crossover_vectors(parent1, parent2, cutPoints, crossover_range, NUMOF_CHROMOSOMES, CROSOVR_RATE);
		// copy to GPU
		cudaMemcpy(dev_parent1, parent1, cro_cpySize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_parent2, parent2, cro_cpySize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_cutPoints, cutPoints, cro_cpySize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_crossover_range, crossover_range, cro_cpySize, cudaMemcpyHostToDevice);
		// do crossover
		crossover_kernel<<<1, CROSOVR_RATE>>>(dev_chromosomes, dev_parent1, dev_parent2, dev_cutPoints, dev_crossover_range, CROSOVR_RATE / 2, NUMOF_CHROMOSOMES);
	
		// generate mutation data
		generate_mutation_vectors(parent, position, replace_genes, NUMOF_JOBS, NUMOF_CHROMOSOMES, MUT_RATE);
		// copy to GPU
		cudaMemcpy(dev_parent, parent, mut_cpySize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_position, position, mut_cpySize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_replace_genes, replace_genes, sizeof(double)*MUT_RATE, cudaMemcpyHostToDevice);
		// do mutation
		mutation_kernel<<<1, MUT_RATE>>>(dev_chromosomes, dev_parent, dev_position, dev_replace_genes, MUT_RATE, NUMOF_CHROMOSOMES + CROSOVR_RATE);

		machine_selection_kernel<<<job_initialize_kernel_block_dim, job_initialize_kernel_thread_dim>>>(dev_jobs, NUMOF_JOBS, MAX_NUMOF_CHROMOSOMES);
		machine_selection_part2_kernel<<<machine_kernel_block_dim, machine_kernel_thread_dim>>>(dev_jobs, dev_machines,	NUMOF_JOBS, NUMOF_MACHINES, MAX_NUMOF_CHROMOSOMES);
		machine_sort_the_jobs<<<machine_kernel_block_dim, machine_kernel_thread_dim>>>(dev_machines, DEV_SETUP_TIME,  NUMOF_MACHINES, MAX_NUMOF_CHROMOSOMES);

		setup_fitness_value<<<1, MAX_NUMOF_CHROMOSOMES>>>(dev_machines, dev_chromosomes, NUMOF_MACHINES, MAX_NUMOF_CHROMOSOMES);
		// sort_chromosomes<<<1, 1>>>(dev_chromosomes, MAX_NUMOF_CHROMOSOMES);
	}
	clock_t end = clock();
	machine_set_job_list_id<<<machine_kernel_block_dim, machine_kernel_thread_dim>>>(dev_machines, NUMOF_MACHINES, MAX_NUMOF_CHROMOSOMES);

	unsigned int * test_can_run_tools;
	size_t can_run_tool_size;
	scuJob * dev_row_pointer;
	scuJob * dev_test_row_jobs = (scuJob *)malloc(sizeof(scuJob)*NUMOF_JOBS);
	can_run_tool_size = NUMOF_MACHINES * sizeof(unsigned int);
	test_can_run_tools = (unsigned int *)malloc(can_run_tool_size);
	
	FILE * testFile;
#ifndef TEST	 // test machine selection part 1
	for(unsigned int i = 0; i < MAX_NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_row_pointer, &dev_jobs[i], sizeof(scuJob*), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_jobs, dev_row_pointer, sizeof(scuJob)*NUMOF_JOBS, cudaMemcpyDeviceToHost);

		testFile = fopen("MS_ch%u.log", "w");
		for(unsigned int j =0; j < NUMOF_JOBS; ++j){
			fprintf(testFile, "%u : %.3f", dev_test_row_jobs[j].number, dev_test_row_jobs[j].splitValue); 
			// cout<<dev_test_row_jobs[j].number << " : " <<dev_test_row_jobs[j].splitValue<<" ";
			can_run_tool_size = dev_test_row_jobs[j].sizeof_can_run_tools * sizeof(unsigned int);
			fprintf(testFile, "size = %zu ", can_run_tool_size);
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
	// size_t can_run_tool_size;
	// unsigned int * test_can_run_tools;
	// can_run_tool_size = NUMOF_MACHINES* sizeof(unsigned int);
	// test_can_run_tools = (unsigned int *)malloc(can_run_tool_size);
	char fileName[100];
	for(unsigned int i = 0; i < MAX_NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_row_pointer, &dev_jobs[i], sizeof(scuJob*), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_test_row_jobs, dev_row_pointer, sizeof(scuJob)*NUMOF_JOBS, cudaMemcpyDeviceToHost);
		sprintf(fileName, "MS_ch%u.log", i);
		testFile = fopen(fileName, "w");
		for(unsigned int j =0; j < NUMOF_JOBS; ++j){
			
			cudaMemcpy(&os_gene, dev_test_row_jobs[j].os_gene, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(&ms_gene, dev_test_row_jobs[j].ms_gene, sizeof(double), cudaMemcpyDeviceToHost);
			// cout<<dev_test_row_jobs[j].number << " : "<<dev_test_row_jobs[j].splitValue<<endl;
			fprintf(testFile, "Job number : %u\n", dev_test_row_jobs[j].number);
			fprintf(testFile, "ms : %.3f, os : %.3f ", ms_gene, os_gene);
			can_run_tool_size = dev_test_row_jobs[j].sizeof_can_run_tools * sizeof(unsigned int);
			fprintf(testFile, "size = %zu\n", can_run_tool_size);
			cudaMemcpy(test_can_run_tools, dev_test_row_jobs[j].can_run_tools, can_run_tool_size, cudaMemcpyDeviceToHost);
			for(unsigned int k = 0; k < dev_test_row_jobs[j].sizeof_can_run_tools; ++k){
				fprintf(testFile, "%u ", test_can_run_tools[k]);
			}
			fprintf(testFile, "Choose Machine = %d, Process Time = %.3f\n\n", dev_test_row_jobs[j].machine_id, dev_test_row_jobs[j].machine_process_time);
		}
		fclose(testFile);
	}
#endif

	//return 0;


// #ifdef TEST
	// scuMachine * dev_machine_row_pointer;
	// scuMachine * dev_machines_row = (scuMachine*)malloc(sizeof(scuMachine) * NUMOF_MACHINES);
	scuChromosome * test_chromosomes = (scuChromosome*)malloc(sizeof(scuChromosome) * NUMOF_CHROMOSOMES);
	cudaMemcpy(test_chromosomes, dev_chromosomes, sizeof(scuChromosome)*NUMOF_CHROMOSOMES, cudaMemcpyDeviceToHost);

	scuMachine * dev_machine_row_pointer;
	dev_machines_row = (scuMachine*)malloc(sizeof(scuMachine) * NUMOF_MACHINES);
	unsigned int * job_lists;
	unsigned int total = 0;
	for(unsigned int i = 0; i < NUMOF_CHROMOSOMES; ++i){
		cudaMemcpy(&dev_machine_row_pointer, &dev_machines[i], sizeof(scuMachine*), cudaMemcpyDeviceToHost);
		cudaMemcpy(dev_machines_row, dev_machine_row_pointer, sizeof(scuMachine) * NUMOF_MACHINES, cudaMemcpyDeviceToHost);
		total  = 0;
		sprintf(fileName, "MS_machine_%u.log", i);
		testFile = fopen(fileName, "w");
		for(unsigned int j = 0; j < NUMOF_MACHINES; ++j){
			fprintf(stdout, "Machine Id %d : %.3f || ", dev_machines_row[j].number, dev_machines_row[j].total_time);
			job_lists = (unsigned int *)malloc(sizeof(unsigned int) * dev_machines_row[j].job_size);
			cudaMemcpy(job_lists, dev_machines_row[j].job_id_list, sizeof(unsigned int)*dev_machines_row[j].job_size, cudaMemcpyDeviceToHost);
			total += dev_machines_row[j].job_size;
			for(unsigned int k = 0; k < dev_machines_row[j].job_size; ++k){
				fprintf(stdout, "%u ", job_lists[k]);
			}
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "size = %u\nfitness value = %.3f\n\n", total, test_chromosomes[i].fitnessValue);

	}	
// #endif

	printf("Elapsed Time = %.3f", (double)(end - start) / (double)(CLOCKS_PER_SEC));

	return 0;
}
