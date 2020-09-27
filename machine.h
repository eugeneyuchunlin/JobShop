#ifndef MACHINE_H
#define MACHINE_H

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <map>
#include "gantt.h"
#include "job.h"
#include <string>

class Machine{
private:
	std::vector<Job *> _jobs;
	static std::map<std::string, cv::Scalar>::iterator _colorIt;
	const static std::map<std::string, cv::Scalar>::iterator _colorItEnd;
	int _number;
	std::string _machineID;
	std::string _status;
	int _recoverTime;
	cv::Scalar tempScalar;
	int _colorCode[3];
	int _totalTime;
	void generate_color_code();
	std::map<std::string, int> _processTime;
public:
	Machine(int number);
	Machine(int number, std::string machineID, std::vector<std::string> status);
	Machine();
	void add_job(Job *);
	void sort_job();
	void demo();
	void clear();
	void add_into_gantt_chart(GanttChart & gantt);
	int get_total_time();
};

#endif
