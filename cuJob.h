#ifndef __CU_JOB_H__
#define __CU_JOB_H__
#include <string>
#include <map>
#include <vector>
#include "job_base.h"

struct scuJob{
	double * ms_gene;
	double * os_gene;
	double * process_time; // shared clone
	double start_time;
	double end_time;
	double splitValue;
	unsigned int machine_id;
	double machine_process_time;
	unsigned int * can_run_tools; // shared clone
	unsigned int number; // clone
	unsigned int sizeof_can_run_tools; // clone
	unsigned int capacityof_can_run_tools; // clone
	unsigned int sizeof_process_time; // clone
	unsigned int capacityof_process_time; // clone
	std::string job_id;
};

scuJob * createScuJob(
		int, 
		std::map<std::string, std::string>, 
		std::map<std::string,  std::map<std::string, int> >
);

scuJob * shared_clone(scuJob *);

class cuJob;
class cuChromosome;

class cuJob : public Job_base{

/** Job property*/
private:
    double _startTime;
    double _endTime;
    double _R_QT;
    double _ARRIVE_T;
    double _URGENT;
    double _quality;

    int _number;
    int _durationTime;

    double _splitValue;
	
	std::string _jobID;
	std::string _RECIPE;
    std::vector<std::string> _canRunTools;
	std::vector<int> _canRunToolsInNumbers;
	std::vector<double> _processTime;

/** memory pointer, make the algorithm run efficiently*/
private:
    double * _machineSelectionGene; // point to chromosome's gene
    double * _arrangementGene; // point to chromosome's gene
	int * _machineIndexAddress; // get the machine index directly from memory

public:
	cuJob();
	cuJob(
			int number,
			int * machineIndex,
			std::map<std::string, std::string>, 
		    std::map<std::string, std::map<std::string, int> > eqp_recipe
	);

	double get_start_time();
	double get_end_time();
	double get_split_value();
	void set_machine_index(int * index_address);
	void set_start_time(double time);
	void clear();
	friend class cuChromosome;
	friend int main(int argc, const char * argv[]);

};


#endif
