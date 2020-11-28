#include <iostream>
#include <locale>
#include <string>
#include <map>
#include <strings.h>
#include <vector>
#include <ctime>
#include "cuJob.h"
#include "cuChromosome.h"
#include "configure.h"
#include <cuda.h>
#include <cuda_runtime.h> 

using namespace std;

__global__ void machine_seletion(double * splitValues, double * chromosomes, int * machineIndexes){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int times = 0;
	double gene = chromosomes[index];
	double i = 0.0;
	machineIndexes[index] = 0;
	for(i = splitValues[index]; i <= 1.0; i += splitValues[index]){
		if(gene < i){
			machineIndexes[index] = times;
			break;
		}
		++times;
	}
}

vector<vector<cuJob *> > create_jobs(
		unsigned int numberOfChromosomes, 
		unsigned int numberOfJobs,
		map<string, map<string, int> > eqp_receipe, 
		vector<map<string, string> > wip_data,
		double * splitValues,
		int * machineIndexes
){
	vector<vector<cuJob *> > jobs;
	unsigned int i,j,size;
	cuJob * tempcujob;
	size = numberOfJobs;


	for(i = 0; i < numberOfChromosomes; ++i){
		vector<cuJob *> temp;
		for(j = 0; j < size; ++j){
			tempcujob = new cuJob(i, &machineIndexes[i*numberOfJobs + j], wip_data[j], eqp_receipe);
			splitValues[i*numberOfJobs+j] = tempcujob->get_split_value();
			machineIndexes[i*numberOfJobs+j] = 0;
			temp.push_back(tempcujob);
		}
		jobs.push_back(temp);

	}
	return jobs;

}


int main(int argc, const char * argv[]){
	
	clock_t read_file = clock();
	srand(time(NULL));
	map<string, map<string, int> >Data;
	Data = EQP_TIME("./semiconductor-scheduling-data/EQP_RECIPE.txt");
	
	map<string, vector<string> >Status;
	Status = STATUS("./semiconductor-scheduling-data/Tool.txt");

	vector<map<string, string> > wipData;
	wipData = WIP("./semiconductor-scheduling-data/WIP.txt");

	vector<vector<int> > setup_time;
	setup_time = SETUP_TIME("./semiconductor-scheduling-data/Setup_time.txt");
	clock_t endof_reading = clock();

	unsigned int numberOfChromosomes = atoi(argv[1]);
	unsigned int numberOfJobs = wipData.size();
	int machineIndexes[numberOfChromosomes][numberOfJobs];
	double splitValues[numberOfChromosomes][numberOfJobs];
	double machineSelectionGenes[numberOfChromosomes][numberOfJobs];
	
	vector<vector<cuJob *> > jobs = create_jobs(
			numberOfChromosomes, 
			numberOfJobs,
			Data, wipData, 
			(double*)splitValues, 
			(int*)machineIndexes
	);
	clock_t endof_cloning_jobs = clock();
	vector<cuChromosome *> chromosomes;

	for(unsigned int i = 0; i < jobs.size(); ++i){
		chromosomes.push_back(new cuChromosome(jobs[i]));
	}


	for(unsigned int i = 0; i < numberOfChromosomes; ++i){
		chromosomes[i]->get_machine_selection_genes(machineSelectionGenes[i]);
	}

	clock_t endof_creating_chromosomes = clock();


	// testing variables
	

	/** Setup cuda variables*/
	double * d_splitValues, *d_machineSelectionGenes;
	int *d_machineIndexes;
	cudaMalloc((void**)&d_splitValues, numberOfChromosomes*numberOfJobs*sizeof(double));
	cudaMalloc((void**)&d_machineSelectionGenes, numberOfChromosomes*numberOfJobs*sizeof(double));
	cudaMalloc((void**)&d_machineIndexes, numberOfChromosomes*numberOfJobs*sizeof(int));
	cudaMemcpy(d_splitValues, splitValues, numberOfChromosomes*numberOfJobs*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_machineSelectionGenes, machineSelectionGenes, numberOfChromosomes*numberOfJobs*sizeof(double), cudaMemcpyHostToDevice);

	clock_t cuda_computing = clock();
	machine_seletion<<<numberOfJobs, numberOfChromosomes>>>(d_splitValues, d_machineSelectionGenes, d_machineIndexes);

	clock_t endof_cuda_computing = clock();
	cudaMemcpy(machineIndexes, d_machineIndexes, numberOfChromosomes*numberOfJobs*sizeof(int), cudaMemcpyDeviceToHost);

	for(unsigned int i = 0; i < jobs.size(); ++i){
		for(unsigned int j = 0; j < 10; ++j){
			printf("%s ", jobs[i][j]->_canRunTools[*jobs[i][j]->_machineIndexAddress].c_str());
		}
		printf("\n");
	}
	

	FILE * file = fopen("machine.txt", "w");
	for(int i = 0; i < numberOfChromosomes; ++i){
		for(int j = 0; j < 20; ++j){
			fprintf(file, "%d ", machineIndexes[i][j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);

	// file = fopen("gene.txt", "w");
	// for(int i = 0; i < numberOfChromosomes; ++i){
	// 	for(int j = 0; j < 10; ++j){
	// 		fprintf(file, "%.3f ", machineSelectionGenes[i][j]);
	// 	}
	// 	fprintf(file, "\n");
	// }
	// fclose(file);

	// file = fopen("split.txt", "w");
	// for(int i = 0; i < 10; ++i){
	// 	fprintf(file, "%.3f ", splitValues[0][i]);
	// }
	// fprintf(file, "\n");
	
	printf("read file = %f\n", (double)(endof_reading - read_file) /(double)(CLOCKS_PER_SEC));
	printf("clone jobs = %f\n", (double)(endof_cloning_jobs - endof_reading) / (double)(CLOCKS_PER_SEC));
	printf("create chromosomes = %f\n", (double)(endof_creating_chromosomes - endof_cloning_jobs)/(double)(CLOCKS_PER_SEC));
	printf("cuda computing = %f\n",(double)(endof_cuda_computing - cuda_computing)/(double)(CLOCKS_PER_SEC));
	
	return 0;
}


