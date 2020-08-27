#include "job.h"

Job::Job(int number, std::map<int, int> duration){
	_number = number;
	_duration = duration;
	_startTime = -1;
	_endTime = -1;
	_machineNo = -1;
	_duration_time = 0;
	double split = 1.0 / (double)_duration.size();
	double part = split;
	for(std::map<int, int>::iterator it = _duration.begin(); it != _duration.end(); it++, part += split)
		_machinCircle[it->first] = part;
}


void Job::assign_machine_number(int machineNumber){
	_machineNo = machineNumber;	
}

void Job::assign_machine_number(double gene){
	/*
	std::cout<<"Job "<<this->_number<<std::endl;
	for(std::map<int, double>::iterator it = _machinCircle.begin(); it != _machinCircle.end(); it++){
		std::cout<<it->first<<" "<<it->second<<"	,";
	}
	std::cout<<std::endl;
	std::cout<<"gene = "<<gene<<std::endl;
	*/
	for(std::map<int, double>::iterator it = _machinCircle.begin(); it != _machinCircle.end(); it++){
		if(gene < it->second){
			_machineNo = it->first;
			break;
		}
	}
	/*
	std::cout<<"assign to machine "<<_machineNo<<std::endl;
	std::cout<<"====================="<<std::endl;	
	*/
}

void Job::assign_machine_order(double gene){
	this->_gene_order = gene;
}

void Job::assign_machine_order(int order){
	this->_real_order = order;
}



double Job::get_gene_order(){
	return _gene_order;
}

int Job::get_start_time(){
	return _startTime;
}

int Job::get_number(){
	return _number;
}

int Job::get_end_time(){
	return _endTime;
}

void Job::set_start_time(int time){
	_startTime = time;
	_endTime = _startTime + _duration[_machineNo];
	_duration_time = _endTime - _startTime;
}

int Job::get_machine_number(){
	return _machineNo;
}

int Job::get_real_order(){
	return _real_order;
}

bool compare_job_order(Job * job1, Job * job2){
	return job1->_gene_order > job2->_gene_order;	
}

int Job::get_duration(){
	return _duration_time;
}

void Job::clear(){
	_startTime = _endTime = 0;
	_machineNo = _real_order = -1;
	_gene_order = -1.0;

}
