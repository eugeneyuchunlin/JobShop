#include "cuJob.h"

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
