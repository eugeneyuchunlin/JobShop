#include "setup_time_job.h"
#include "job_base.h"
#include <chrono>

SetupTimeJob::SetupTimeJob(int duration):Job_base(){
	_duration = duration;
	_startTime = _endTime = 0;
}

double SetupTimeJob::get_start_time(){
	return _startTime;
}

double SetupTimeJob::get_end_time(){
	return _endTime;
}

void SetupTimeJob::set_start_time(double time){
	_startTime = time;
	_endTime = _startTime + _duration;
}

void SetupTimeJob::clear(){
	_duration = _startTime = _duration = 0;
}
