#ifndef __CU_CHROMOSOME_H__
#define __CU_CHROMOSOME_H__
#include <vector>
#include <random>
#include <memory.h>
#include <stdio.h>
#include "cuJob.h"



struct scuChromosome{
	unsigned int number;
	double * genes;
	double * dev_genes;
	double * machineSelectionGenes; 
	double * arrangementGenes;
	unsigned int size;
	double fitnessValue;
};

scuChromosome * createScuChromosome(
	unsigned int number,
	unsigned int NUMOF_JOBS
);

class cuChromosome{
private:
	double * _genes;
	double * _machineSelectionGenes;
	double * _arrangementGenes;
	unsigned int _size;
	double * _out_machineSelectionGenes;
	double * _out_arrangementGenes;
private:
	double _random();
public:
	cuChromosome(std::vector<cuJob *>);
	void link_cuJobs(std::vector<cuJob *>);	
	double * get_machine_selection_genes(double *);
	double * get_job_arrangement_genes();
};

#endif
