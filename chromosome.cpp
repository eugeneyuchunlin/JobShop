#include "chromosome.h"
#include <cstddef>
#include <ostream>
#include <vector>

Chromosome::Chromosome(int size){
	_size = size;
	_chromosome = NULL;
	_chromosome = new double[size * 2];
	// _chromosome = (double *)malloc(size * 2 * sizeof(double));
	_machine = _chromosome;
	_order = _chromosome + size;

	for(int i = 0, size = 2 * _size; i < size; ++i){
		_chromosome[i] = _random();	
	}
	
	for(int i = 0; i < size * 2; ++i)
		_chromosome_temp.push_back(_chromosome[i]);
	
}

Chromosome::Chromosome(){
	_chromosome = _machine = _order = NULL;
	_size = 0;
}


Chromosome::Chromosome(const Chromosome & otherChromsome){
	// std::cout<<"size = "<<otherChromsome<<std::endl;
	this->_size = otherChromsome._size;
	this->_chromosome = new double[this->_size * 2];
	// this->_chromosome = (double *)malloc(this->_size * 2 * sizeof(double));
	this->_machine = this->_chromosome;
	this->_order = this->_chromosome + this->_size;
	this->_chromosome_temp = otherChromsome._chromosome_temp;
	this->value = otherChromsome.value;
	for(int i = 0, size = 2 * this->_size; i < size; ++i){
		this->_chromosome[i] = otherChromsome._chromosome[i];
	}
}
Chromosome & Chromosome::operator=(const Chromosome & otherChromsome){
	// free the memory
	if(otherChromsome._size != this->_size && _chromosome != NULL){
		delete _chromosome;
		_chromosome = _machine = _order = NULL;
		this->_size = otherChromsome._size;
		// this->_chromosome = new double[this->_size * 2];
	}

	if(_chromosome == NULL){
		this->_size = otherChromsome._size;
		this->_chromosome = new double[this->_size * 2];
		this->_machine = this->_chromosome;
		this->_order = this->_chromosome + this->_size;

	}

	// copy the information
	this->value = otherChromsome.value;
	this->_chromosome_temp = otherChromsome._chromosome_temp;
	for(int i = 0, size = 2 * this->_size; i < size; ++i){
		this->_chromosome[i] = otherChromsome._chromosome[i];
	}

	return *this;
}


double Chromosome::_random(){
	return (double)rand() / (RAND_MAX + 1.0);
}

double Chromosome::getMachine(int jobNumber){
	return _machine[jobNumber];
}

double Chromosome::getOrder(int jobNumber){
	return _order[jobNumber];	
}

std::vector<Chromosome> Chromosome::operator*(Chromosome & chro){
	if(chro._size != this->_size)
		throw "the chromosomes are not the same species";
	int lower = rand() % this->_size * 2;
	int upper = this->_size * 2 - lower;
	upper = rand() % upper;
	upper += lower;
	// printf("(lower, upper) = (%d, %d)\n", lower, upper);
	double temp;
	Chromosome child(*this);
	// intercross
	for(int i = lower; i < upper; ++i){
		temp = child._chromosome[i];
		child._chromosome[i] = chro._chromosome[i];
		chro._chromosome[i] = temp;
	}
	std::vector<Chromosome> children;
	children.push_back(child);
	children.push_back(chro);
	return children;
}

Chromosome Chromosome::operator!(){
	Chromosome child(*this);
	// std::cout<<rand()<<std::endl;
	// std::cout<<this->_size<<std::endl;
	int choosedGene = (rand() % this->_size);
	// std::cout<<"safe"<<std::endl;
	child._chromosome[choosedGene] = _random();
	return child;
}

Chromosome::~Chromosome(){
	// std::cout<<"----------------------"<<std::endl;
	// std::cout<<"chromosome destructor"<<std::endl;
	// std::cout<<this->_chromosome<<std::endl;
	if(this->_chromosome != NULL){
		// std::cout<<"it is freed"<<std::endl;
		delete this->_chromosome;
		this->_chromosome = NULL;
		// std::cout<<this->_chromosome<<std::endl;
	}
	// std::cout<<"-----------------------"<<std::endl;
}

void Chromosome::__repr__(){
	for(int i = 0, size = 2 * _size; i < size; ++i){
		printf("%.2f ", _chromosome[i]);
	}
	printf("\n");
}

ChromosomeLinker::ChromosomeLinker(){
	link_num = 0;
	value = 0;
	linkChromosome = NULL;
}

ChromosomeLinker::ChromosomeLinker(int link_num, int value, Chromosome * linkChromosome){
	this->link_num = link_num;
	this->value = value;
	this->linkChromosome = linkChromosome; 
}

bool chromosome_comparator(Chromosome & c1, Chromosome & c2){
	return c1.value > c2.value;
}

bool chromosomelinker_comparator(ChromosomeLinker c1, ChromosomeLinker c2){
	return c1.value < c2.value;
}

std::ostream & operator<<(std::ostream & out, const Chromosome & chromosome){
	return out<<chromosome._size;	
}

