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
	// int _machineNo;
	int _real_order;	
	double _gene_order;
	int _duration_time;
	std::map<int, int > _duration;
	std::map<int, double> _machinCircle;

	
	std::string _machineID;
	std::string _recipe;
	std::string _jobID;	
	std::map<std::string, std::string>_row_data;
	std::map<std::string, int> _processTime;
	std::map<std::string, double> _machineIDCircle;
public:
	Job(int number, std::map<int, int> duration);
	Job(int number, std::map<std::string, std::string> row, std::map<std::string, std::map<std::string, int> > eqp_recipe);
	// void assign_machine_number(int machinNumber);
	void assign_machine_id(std::string machineID);
	void assign_machine_number(double gene);
	void assign_machine_order(double gene);
	void assign_machine_order(int order);
	int get_start_time();
	int get_end_time();
	int get_number();
	// int get_machine_number();
	std::string get_machine_id();
	int get_duration();
	double get_gene_order();
	int get_real_order();
	void set_start_time(int time);
	void clear();
	friend bool compare_job_order(Job *, Job *);
};

#endif
