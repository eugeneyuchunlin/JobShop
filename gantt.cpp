#include "gantt.h"
#include "def.h"
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>



GanttChart::GanttChart(int max, int machineNumber){
	_chart = new cv::Mat(FRAME_HEIGHT * 7, FRAME_WIDTH + max * 8, CV_8UC3,cv::Scalar(255, 255, 255) );
	_draw_frame(max);
	set_machine_number(machineNumber); 
}


GanttChart::~GanttChart(){
	delete _chart;
}

void GanttChart::draw_job(int machine,int jobNumber,  int timeStart, int timeEnd, cv::Scalar color){
	int startX = _position_x_mapping[machine];
	int startY = _position_y_mapping[machine];
	int gap = 7 * (timeEnd - timeStart);
	int textLength = std::to_string(jobNumber).length();
	textLength -= 1;
	// std::cout<<textLength<<std::endl;
	cv::rectangle(* _chart, cv::Rect(startX + 7 * timeStart, startY - 100 , gap , 200) , color, -1);
	cv::rectangle(* _chart, cv::Rect(startX + 7 * timeStart, startY - 100 , gap , 200) , COLORMAP["black"], 3);
	cv::putText(* _chart, std::to_string(jobNumber), cv::Point(startX + 7 * timeStart + gap / 2 - 30 - textLength * 10, startY + 20), cv::FONT_HERSHEY_DUPLEX, 3, this->COLORMAP["lightgrey"],3,  cv::LINE_8);

}

void GanttChart::set_time(int machine, int time){
	int startY = _position_y_mapping[machine];
	int startX = _position_x_mapping[machine];
	cv::putText(* _chart, std::to_string(time), cv::Point(startX + 7 * time, startY + 20), cv::FONT_HERSHEY_DUPLEX, 3, this->COLORMAP["blue"],3,  cv::LINE_8);


	
}

void GanttChart::_draw_frame(int width){
	// draw the chart frame
	cv::rectangle(*_chart, cv::Rect(FRAME_START_X, FRAME_START_Y,600 * 2 + width * 8, 800 * 5), cv::Scalar(119, 119, 119), 2);	
	int temp = 0;
	// for(int i = FRAME_START_X + 150, length = FRAME_START_X + 150 + 7 * width; i < length; i += 140, temp += 20)
	//	cv::putText(*_chart, std::to_string(temp), cv::Point(i , 2000), cv::FONT_HERSHEY_DUPLEX, 2, this->COLORMAP["black"],3, cv::LINE_8); 	

	/*
	for(int i = 1200; i < 1200 + 7*width; i += 70){
		cv::putText(*_chart, std::to_string(i / 70), cv::Point(i, 1700),cv::FONT_HERSHEY_DUPLEX, 3, this->COLORMAP["black"],3, cv::LINE_8); 	
	}
	*/
}



cv::Mat GanttChart::get_img(){
	return * _chart;
}

void GanttChart::set_machine_number(int number){
	_machineNumber = number;
	float gap = 4000.0 / (number + 1);

	for(float i = FRAME_START_Y + gap; i <= 4000.0; i += gap){
		cv::line(*_chart, cv::Point(FRAME_START_X - 100, i), cv::Point(FRAME_START_X + 100, i), cv::Scalar(0, 0, 0), 2);
		cv::putText(*_chart, "M " + std::to_string(number), cv::Point(FRAME_START_X - 400, i + 50), cv::FONT_HERSHEY_DUPLEX, 3, cv::Scalar(0, 0, 0), 10, cv::LINE_8);
		_position_x_mapping[number] = FRAME_START_X + 150;
		_position_y_mapping[number] = i;
		number -= 1;
	}
}

std::map<std::string, cv::Scalar> GanttChart::COLORMAP = 
{
	{"red", cv::Scalar(0, 0, 255)}, 
	{"lightcoral", cv::Scalar(128, 128, 240)},
	{"lightyellow", cv::Scalar(224, 255, 255)},
	{"lemonchiffon", cv::Scalar(205, 250, 255)},
	{"yellow", cv::Scalar(0, 255, 255)},
	{"lime", cv::Scalar(0, 255, 0)},
	{"green", cv::Scalar(0, 128, 0)},
	{"lightgrean", cv::Scalar(144, 238, 144)},
	{"olive", cv::Scalar(0, 128, 128)},
	{"lightcyan", cv::Scalar(255, 255, 224)},
	{"cyan", cv::Scalar(255, 255, 0)},
	{"lightblue", cv::Scalar(230, 216, 173)},
	{"lightskyblue", cv::Scalar(250, 206, 135)},
	{"skyblue", cv::Scalar(235, 206, 135)},
	{"deepskyblue", cv::Scalar(255, 191, 0)},
	{"blue", cv::Scalar(255, 0, 0)},
	{"navy", cv::Scalar(128, 0, 0)},
	{"lavenler", cv::Scalar(250, 230, 230)},
	{"pink", cv::Scalar(203, 192, 255)},
	{"white", cv::Scalar(255, 255, 255)},
	{"lightgrey", cv::Scalar(160, 149, 143)},
	// {"snow", cv::Scalar(250, 250, 255)},
	// {"honeydew", cv::Scalar(240, 255, 240)},
	// {"mintcream", cv::Scalar(250, 255, 245)},
	// {"azure", cv::Scalar(255, 255, 240)},
	// {"aliceblue", cv::Scalar(255, 248, 240)},
	// {"ghostwhite", cv::Scalar(255, 248, 248)},
	// {"whitesmoke", cv::Scalar(245, 245, 245)},
	{"seashell", cv::Scalar(238, 245, 255)},
	{"beige", cv::Scalar(220, 245, 245)},
	{"ivory", cv::Scalar(240, 255, 255)},
	{"black", cv::Scalar(0, 0, 0)}
	
};
