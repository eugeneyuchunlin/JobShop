#include "cuChromosome.h"
#include "cuJob.h"
#include <cstdlib>

cuChromosome::cuChromosome(std::vector<cuJob *> jobs){
	this->_size = jobs.size();
	this->_genes = (double *)malloc(this->_size * 2 * sizeof(double));
	this->_machineSelectionGenes = _genes;
	this->_arrangementGenes = _genes + this->_size;
	this->link_cuJobs(jobs);	
	for(unsigned int i = 0, size = this->_size * 2; i < size; ++i){
		this->_genes[i] = _random();
	}

	this->_out_machineSelectionGenes = (double *)malloc(this->_size * sizeof(double));
	this->_out_arrangementGenes = (double *)malloc(this->_size * sizeof(double));
}


void cuChromosome::link_cuJobs(std::vector<cuJob *>jobs){
	for(unsigned int i = 0; i < this->_size; ++i){
		jobs[i]->_machineSelectionGene = &this->_machineSelectionGenes[i];	
		jobs[i]->_arrangementGene = &this->_arrangementGenes[i];
	}
}

double cuChromosome::_random(){
	return (double)rand() / (RAND_MAX + 1.0);
}

double * cuChromosome::get_machine_selection_genes(double *outer){
	memcpy(outer, this->_machineSelectionGenes, sizeof(double)*this->_size);
	return outer;
}
