#ifndef JOB_H
#define JOB_H

#include <iostream>
#include <vector>
#include <map>
#include <string>

class Job;

bool compare_job_order(Job * job1, Job * job2);

class Job{
private:
	int _startTime;
	int _endTime;
	int _number;
	int _machineNo;
	int _real_order;	
	double _gene_order;
	int _duration_time;
	std::map<int, int > _duration;
	std::map<int, double> _machinCircle;
public:
	Job(int number, std::map<int, int> duration);
	void assign_machine_number(int machinNumber);
	void assign_machine_number(double gene);
	void assign_machine_order(double gene);
	void assign_machine_order(int order);
	int get_start_time();
	int get_end_time();
	int get_number();
	int get_machine_number();
	int get_duration();
	double get_gene_order();
	int get_real_order();
	void set_start_time(int time);
	void clear();
	friend bool compare_job_order(Job *, Job *);
};

#endif
