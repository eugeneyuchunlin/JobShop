#ifndef JOB_H
#define JOB_H

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "job_base.h"

class Job;

bool compare_job_order(Job * job1, Job * job2);
bool compare_job_order_quality(Job * job1, Job * job2);

class Job:public Job_base{
private:
	double _startTime;
	double _endTime;
	double _gene_order;
	double _R_QT;
	double _ARRIVE_T;
	double _quality;
	double _urgent;

	bool _isArrive;

	int _number;
	int _real_order;	
	int _duration_time;

	std::string _machineID;
	std::string _recipe;
	std::string _jobID;

	std::map<int, int > _duration;
	std::map<int, double> _machinCircle;	
	std::map<std::string, double> _processTime;
	std::map<std::string, double> _machineIDCircle;
	std::map<std::string, std::string>_row_data;
public:
	bool RQ_T_LEGAL;
	bool ARRIVE_T_LEGAL;
	Job(int number, std::map<int, int> duration);
	Job(int number, std::map<std::string, std::string> row, std::map<std::string, std::map<std::string, int> > eqp_recipe);
	// void assign_machine_number(int machinNumber);
	void assign_machine_id(std::string machineID);
	void assign_machine_number(double gene);
	void assign_machine_order(double gene);
	void assign_machine_order(int order);
	double get_start_time();
	double get_end_time();
	int get_number();
	// int get_machine_number();
	std::string get_machine_id();
	std::string get_recipe();
	int get_duration();
	double get_gene_order();
	double get_arrive_time();
	int get_real_order();
	void set_start_time(double time);
	void clear();
	bool is_arrive();
	friend bool compare_job_order(Job *, Job *);
	friend bool compare_job_order_quality(Job *, Job *);
};

#endif
