#include "cuJob.h"
#include <cstdlib>
#include <iostream>

scuJob * createScuJob(
		int number, 
		std::map<std::string, std::string> row,
		std::map<std::string, std::map<std::string, int > > eqp_recipe
){
	scuJob * job = (scuJob *)malloc(sizeof(scuJob));
	std::string recipe = row["RECIPE"];
	std::string can_run_tools = row["CANRUN_TOOL"];
	double qty = std::stof(row["QTY"]) / 25.0;
	job->number = number;
	std::string lotId = row["LOT_ID"];	
	// job->job_id = row["LOT_ID"];
	job->ms_gene = nullptr;
	job->os_gene = nullptr;
	job->sizeof_can_run_tools = 0;
	job->capacityof_can_run_tools = 20;
	job->can_run_tools = (unsigned int *)malloc(job->capacityof_can_run_tools * sizeof(unsigned int));
	
	job->capacityof_process_time = 10;
	job->process_time = (double *)malloc(job->capacityof_process_time*sizeof(double));
	
	
	size_t startPos = 0;
	std::string temp;
	unsigned int i = 0,
				 j = 0;
	do{
		temp = can_run_tools.substr(startPos, 6);
		job->can_run_tools[i] = std::stoi(&temp[3]);
		job->process_time[j] = (double)eqp_recipe[recipe][temp] * qty;
		++i;
		++j;
		if(i >= job->capacityof_can_run_tools){
			realloc(job->can_run_tools, 20*sizeof(unsigned int) + job->capacityof_process_time*sizeof(unsigned int));	
		}

		if(j >= job->capacityof_process_time){
			realloc(job->process_time, 20*sizeof(double) + job->capacityof_process_time*sizeof(double));
		}
		startPos += 6;
	}while(startPos != can_run_tools.length());

	job->sizeof_can_run_tools = i;
	job->sizeof_process_time = j;
	job->splitValue = 0;

	return job;

}

struct scuJob * shared_clone(scuJob * src){
	scuJob * njob = (scuJob *)malloc(sizeof(scuJob));
	njob->number = src->number;
	njob->sizeof_can_run_tools = src->sizeof_can_run_tools;
	njob->sizeof_process_time = src->sizeof_process_time;
	njob->start_time = njob->end_time = 0;
	njob->splitValue = 0;
	return njob;
}


cuJob::cuJob(){
	
}

cuJob::cuJob(
		int number,
		int * machineIndex,
		std::map<std::string, std::string>  row,
		std::map<std::string, std::map<std::string, int> > eqp_recipe
){
	_jobID = row["LOT_ID"];	
	_RECIPE = row["RECIPE"];
	_R_QT = std::stof(row["R_QT"]) * 60;
	_ARRIVE_T = std::stof(row["ARRIV_T"]);

	_startTime = _endTime = -1;
	_number = number;
	_URGENT = std::stof(row["URGENT_W"]);
	std::string canRunMachines = row["CANRUN_TOOL"];
	double QTY = std::stof(row["QTY"]) / 25;
	
	size_t startPos = 0;
	std::string temp;

	do{
		temp = canRunMachines.substr(startPos, 6);
		_canRunTools.push_back(temp);
		_canRunToolsInNumbers.push_back(std::stoi(&temp[3]));
		_processTime.push_back((double)eqp_recipe[_RECIPE][temp] * QTY);
		startPos += 6;
	}while(startPos != canRunMachines.length());
	
	_splitValue = 1.0 /(double)_canRunTools.size();
	this->_machineIndexAddress = machineIndex;
		
}

void cuJob::set_machine_index(int * index_address){
	this->_machineIndexAddress = index_address;
}

double cuJob::get_split_value(){
	return this->_splitValue;
}

double cuJob::get_start_time(){
	return _startTime;
}

double cuJob::get_end_time(){
	return _endTime;
}

void cuJob::set_start_time(double time){
	this->_startTime = time;
}

void cuJob::clear(){

}
