#ifndef __GANTT_H__
#define __GANTT_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <map>

#include "def.h"


class GanttChart{
private:
	cv::Mat * _chart;
	int _machineNumber;
	int _maxTime;
	std::map<int, int> _position_x_mapping;
	std::map<int, int> _position_y_mapping;	
	void _draw_frame(int width);
		
public:
	static std::map<std::string, cv::Scalar> COLORMAP;
	
	cv::Mat get_img();	
	GanttChart(int max);
	~GanttChart();
	void set_machine_number(int); 
	void draw_job(int machine, int jobNumber, int timeStart, int timeEnd, cv::Scalar);
	void set_time(int machine, int time);
	
};


#endif
