#ifndef __CU_JOB_H__
#define __CU_JOB_H__
#include <string>
#include <map>
#include <vector>
#include "job_base.h"

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
