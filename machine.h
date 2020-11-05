#ifndef MACHINE_H
#define MACHINE_H

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <map>
#include "gantt.h"
#include "job.h"
#include "job_base.h"
#include "setup_time_job.h"
#include <string>

class Machine{
private:
	static int ARRIVE_PENALTY;
	static int R_QT_PENALTY;

private:
	std::vector<Job *> _jobs;
	std::vector<Job_base *> _setup_time_jobs;
	static std::map<std::string, cv::Scalar>::iterator _colorIt;
	const static std::map<std::string, cv::Scalar>::iterator _colorItEnd;
	std::vector<std::vector<int> > _setup_time;
	int _number;
	Job_base * _jobs_start;
	Job_base * _current_job;
	std::string _machineID;
	std::string _status;
	int _recoverTime;
	cv::Scalar tempScalar;
	int _colorCode[3];
	int _totalTime;
	void generate_color_code();
	std::map<std::string, int> _processTime;
	int _quality;

	void insert_setup_time();
public:
	Machine(int number);
	Machine(int number, std::string machineID, std::vector<std::string> status, std::vector<std::vector<int> >setup_times);
	Machine();
	void add_job(Job *);
	void sort_job(bool rule=false);
	void demo();
	void clear();
	void add_into_gantt_chart(GanttChart & gantt);
	int get_total_time();
	int get_quality();
	int penalty_function(Job *);
	int get_dead_jobs_amount();
	int get_too_late_job_amount();
};

#endif
