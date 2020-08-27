#ifndef MACHINE_H
#define MACHINE_H

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "gantt.h"
#include "job.h"

class Machine{
private:
	std::vector<Job *> _jobs;
	static std::map<std::string, cv::Scalar>::iterator _colorIt;
	const static std::map<std::string, cv::Scalar>::iterator _colorItEnd;
	int _number;
	cv::Scalar tempScalar;
	int _colorCode[3];
	int _totalTime;
	void generate_color_code();
public:
	Machine(int number);
	Machine();
	void add_job(Job *);
	void sort_job();
	void demo();
	void clear();
	void add_into_gantt_chart(GanttChart & gantt);
	int get_total_time();
};

#endif
