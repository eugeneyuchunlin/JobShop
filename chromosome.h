#ifndef CHROMOSOME_H
#define CHROMOSOME_H

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ostream>
#include <vector>

class Chromosome;

class ChromosomeLinker{
public:
	int link_num;
	double makespan;
	double quality;
	int deadJobs;
	int tooLate;
	Chromosome * linkChromosome;
	ChromosomeLinker();
	ChromosomeLinker(int link_num, double quality, int deadJobs, double makespan, int otolate, Chromosome *);
};

class Chromosome{
private:
	double * _chromosome;
	double * _machine;
	double * _order;
	int _size;
	double _random(); 
	std::vector<double> _chromosome_temp;
public:
	int value;
	Chromosome(int size);
	Chromosome(const Chromosome &);
	Chromosome();
	~Chromosome();
	double getMachine(int jobNumber);
	double getOrder(int jobNumner);
	std::vector<Chromosome> operator*(Chromosome ); // operator * means intercross
	Chromosome & operator=(const Chromosome &);
	Chromosome operator!();		
	void __repr__();
	friend std::ostream &  operator<<(std::ostream & out, const Chromosome &);
};

bool chromosome_comparator(Chromosome & c1, Chromosome & c2);
bool chromosomelinker_makespan_comparator(ChromosomeLinker c1, ChromosomeLinker c2);
bool chromosomelinker_deadJobs_comparator(ChromosomeLinker c1, ChromosomeLinker c2);
bool chromosomelinker_tooLate_comparator(ChromosomeLinker c1, ChromosomeLinker c2);
bool Chromosomelinker_quality_comparator(ChromosomeLinker c1, ChromosomeLinker c2);

std::ostream &  operator<<(std::ostream & out,const Chromosome &);

#endif
