#ifndef SETUP_TIME_JOB_H
#define SETUP_TIME_JOB_H
#include "job_base.h"

class SetupTimeJob : public Job_base{
private:
	double _startTime;
	double _endTime;
	double _duration;
public:
	SetupTimeJob(int duration);
	double get_start_time();
	double get_end_time();
	void set_start_time(double time);
	void clear();
};

#endif
