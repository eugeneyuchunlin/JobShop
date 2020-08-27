#include "machine.h"
#include "gantt.h"
#include <opencv2/core/types.hpp>

std::map<std::string, cv::Scalar>::iterator Machine::_colorIt = GanttChart::COLORMAP.begin();

const std::map<std::string, cv::Scalar>::iterator Machine::_colorItEnd = GanttChart::COLORMAP.end();

Machine::Machine(int number){
	_number = number;
	generate_color_code();
	_totalTime = 0;
}

Machine::Machine(){
	_number = 0;
	_totalTime = 0;
	generate_color_code();
}

void Machine::add_job(Job * job){
	_jobs.push_back(job);
}

void Machine::sort_job(){
	sort(_jobs.begin(), _jobs.end(), compare_job_order);
	int lastFinishTime = 0;
	for(unsigned int i = 0; i < _jobs.size(); ++i){
		_jobs[i]->assign_machine_order((int)i);
		_jobs[i]->set_start_time(lastFinishTime);
		lastFinishTime = _jobs[i]->get_end_time();
	}
	_totalTime = lastFinishTime;
}

void Machine::generate_color_code(){
	for(int i = 0; i < 3; ++i){
		_colorCode[i] = rand() / 256;
	}
}

void Machine::add_into_gantt_chart(GanttChart & gantt){
	

	for(unsigned int i = 0; i < _jobs.size(); ++i){
		if(_colorIt != 	_colorItEnd){
			tempScalar = _colorIt->second;
			_colorIt++;
		}else{
			generate_color_code();
			tempScalar = cv::Scalar(_colorCode[0], _colorCode[1], _colorCode[2]);		
		}
		gantt.draw_job(this->_number + 1, _jobs[i]->get_number() + 1 , _jobs[i]->get_start_time(), _jobs[i]->get_end_time(), tempScalar);	
	}
}

int Machine::get_total_time(){
	return _totalTime;
}
			
void Machine::clear(){
	this->_jobs.clear();
	_colorIt = GanttChart::COLORMAP.begin();
	_totalTime = 0;
}


void Machine::demo(){
	printf("+++++++++Machine %2d+++++++++++\n", _number);
	for(unsigned int i = 0; i < _jobs.size(); ++i){
		printf("|          Job %2d            |\n", _jobs[i]->get_number() + 1);
		printf("|start time = %3d            |\n", _jobs[i]->get_start_time());
		printf("|end time   = %3d            |\n", _jobs[i]->get_end_time());
		printf("------------------------------\n");
	}
	printf("\033[A");
		printf("\r                             ");
	printf("\r++++++++++++++++++++++++++++++\n");
		
}

