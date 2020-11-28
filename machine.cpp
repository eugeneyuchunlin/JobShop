#include "machine.h"
#include "gantt.h"
#include "job.h"
#include "job_base.h"
#include "setup_time_job.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>

std::map<std::string, cv::Scalar>::iterator Machine::_colorIt = GanttChart::COLORMAP.begin();

const std::map<std::string, cv::Scalar>::iterator Machine::_colorItEnd = GanttChart::COLORMAP.end();

int Machine::ARRIVE_PENALTY = 30;
int Machine::R_QT_PENALTY = 300;

Machine::Machine(int number){
	_number = number;
	generate_color_code();
	_totalTime = 0;
}

Machine::Machine(int number, std::string machineID, std::vector<std::string> statuses,std::vector<std::vector<int> > setup_time){
	_number = number;
	_machineID = machineID;
	_status = statuses[0];
	_recoverTime = std::stoi(statuses[1]);
	generate_color_code();
	_totalTime = 0;
	_setup_time = setup_time;
}

Machine::Machine(){
	
	_number = 0;
	_totalTime = 0;
	generate_color_code();
	_jobs_start = nullptr;
	_current_job = nullptr;
}

void Machine::add_job(Job * job){
	_jobs.push_back(job);
}

void Machine::insert_setup_time(){
	int job_number;
	int lastFinishTime;
	int setup_time;
	Job_base * setupTimeJob;
	std::string recipe = _jobs[0]->get_recipe();
	this->_jobs_start = _jobs[0];
	this->_current_job = _jobs[0];
	job_number = _jobs[0]->get_number();
	_jobs[0]->set_start_time(this->_recoverTime);
	lastFinishTime = _jobs[0]->get_end_time();
	for(unsigned int i = 1; i < _jobs.size(); ++i){
		_jobs[i]->assign_machine_order(int(i));
		setup_time = _setup_time[job_number][_jobs[i]->get_number()];
		job_number = _jobs[i]->get_number();
		if(setup_time == 0){ // directly connect
			this->_current_job->set_next(_jobs[i]);
			_jobs[i]->set_last(this->_current_job);
			_jobs[i]->set_start_time(lastFinishTime);
		}else{ // insert the setup time;
			
			setupTimeJob = new SetupTimeJob(setup_time);
			setupTimeJob->set_last(this->_current_job);
			setupTimeJob->set_start_time(lastFinishTime);
			this->_current_job->set_next(setupTimeJob);
			setupTimeJob->set_next(_jobs[i]);
			_jobs[i]->set_last(_jobs[i]);
			_jobs[i]->set_start_time(setupTimeJob->get_end_time());	
			this->_setup_time_jobs.push_back(_jobs[i]);
		}
		this->_current_job = _jobs[i];
		lastFinishTime = this->_current_job->get_end_time();
		_quality += penalty_function(_jobs[i]);
	}
	_totalTime = lastFinishTime;
}

void Machine::sort_job(bool rule){
	if(!rule){ // follow the algorithm's rule
		sort(_jobs.begin(), _jobs.end(), compare_job_order);
		int lastFinishTime = 0;
		int job_number;
		int setup_time;
		Job_base * setupTimeJob;
		// std::string recipe = _jobs[0]->get_recipe();
		// this->_jobs_start = _jobs[0];
		// this->_current_job = _jobs[0];
		// job_number = _jobs[0]->get_number();
		// _jobs[0]->set_start_time(this->_recoverTime);
		// lastFinishTime = _jobs[0]->get_end_time();
		insert_setup_time();
		// for(unsigned int i = 1; i < _jobs.size(); ++i){
		// 	_jobs[i]->assign_machine_order((int)i);
		// // check the setup time and link the jobs
		// 	setup_time = _setup_time[job_number][_jobs[i]->get_number()];
		// 	job_number = _jobs[i]->get_number();
		// 	if(setup_time == 0){
		// 		this->_current_job->set_next(_jobs[i]);
		// 		_jobs[i]->set_last(this->_current_job);
		// 		_jobs[i]->set_start_time(lastFinishTime);
		// 	}else{ // insert the setup time
		// 		setupTimeJob = new SetupTimeJob(setup_time);
		// 		setupTimeJob->set_last(this->_current_job);
		// 		setupTimeJob->set_start_time(lastFinishTime);
		// 		this->_current_job->set_next(setupTimeJob);
		// 		setupTimeJob->set_next(_jobs[i]);
		// 		_jobs[i]->set_last(setupTimeJob);
		// 		_jobs[i]->set_start_time(setupTimeJob->get_end_time());
		// 		this->_setup_time_jobs.push_back(setupTimeJob);
		// 	}
		// 	this->_current_job = _jobs[i];
		// 	lastFinishTime = _jobs[i]->get_end_time();
		// 	_quality += penalty_function(_jobs[i]);
		// }
		// 	_totalTime = lastFinishTime;
		_quality += _totalTime;	
	}else{ // follow my rule

		/* if job is arriving -> arrange it
		 * 
		 */
		sort(_jobs.begin(), _jobs.end(), compare_job_order_quality);	
		insert_setup_time();
		_quality += _totalTime;
	


	}
}

void Machine::generate_color_code(){
	for(int i = 0; i < 3; ++i){
		_colorCode[i] = rand() / 256;
	}
}

void Machine::add_into_gantt_chart(GanttChart & gantt){

	for(unsigned int i = 0; i < _jobs.size(); ++i){
		// if(_colorIt != 	_colorItEnd){
		// 	tempScalar = _colorIt->second;
		// 	_colorIt++;
		// }else{
		// 	generate_color_code();
		// 	tempScalar = cv::Scalar(_colorCode[0], _colorCode[1], _colorCode[2]);		
		// }
		if(!_jobs[i]->ARRIVE_T_LEGAL && !_jobs[i]->RQ_T_LEGAL) // 
			gantt.draw_job(this->_number + 1, _jobs[i]->get_number() + 1 , _jobs[i]->get_start_time(), _jobs[i]->get_end_time(), GanttChart::COLORMAP["blue"]);
		else if(!_jobs[i]->ARRIVE_T_LEGAL) // not ready
			gantt.draw_job(this->_number + 1, _jobs[i]->get_number() + 1 , _jobs[i]->get_start_time(), _jobs[i]->get_end_time(), GanttChart::COLORMAP["red"]);
		else if(!_jobs[i]->RQ_T_LEGAL) // cargo is dead
			gantt.draw_job(this->_number + 1, _jobs[i]->get_number() + 1 , _jobs[i]->get_start_time(), _jobs[i]->get_end_time(), GanttChart::COLORMAP["black"]);
		else // acceptable
			gantt.draw_job(this->_number + 1, _jobs[i]->get_number() + 1 , _jobs[i]->get_start_time(), _jobs[i]->get_end_time(), GanttChart::COLORMAP["white"]);
		
	}
}

int Machine::get_total_time(){
	return _totalTime;
}

int Machine::get_quality(){
	return _quality;
}

int Machine::penalty_function(Job * job){
	int penalty = 0;
	if(!job->ARRIVE_T_LEGAL)
		penalty += Machine::ARRIVE_PENALTY;
	if(!job->RQ_T_LEGAL)
		penalty += Machine::R_QT_PENALTY;

	return penalty;
}
			
void Machine::clear(){
	this->_jobs.clear();
	_colorIt = GanttChart::COLORMAP.begin();
	_totalTime = 0;
	_quality = 0;
}

int Machine::get_dead_jobs_amount(){
	int amount = 0;
	for(unsigned int i = 0; i < _jobs.size(); ++i){
		if(!_jobs[i]->RQ_T_LEGAL)
			amount += 1;
	}
	return amount;
}

int Machine::get_too_late_job_amount(){
	int amount = 0;
	for(unsigned int i = 0; i < _jobs.size(); ++i){
		if(!_jobs[i]->ARRIVE_T_LEGAL)
			amount += 1;
	}
	return amount;

}


void Machine::demo(){
	printf("+++++++++Machine %2d++++++++++++\n", _number);
	for(unsigned int i = 0; i < _jobs.size(); ++i){
		printf("|          Job %2d            |\n", _jobs[i]->get_number() + 1);
		printf("|start time  = %3.3f         |\n", _jobs[i]->get_start_time());
		printf("|arrive time = %3.3f         |\n", _jobs[i]->get_arrive_time()); 
		printf("|end time   = %3.3f         |\n", _jobs[i]->get_end_time());
		printf("------------------------------\n");
	}
	printf("\033[A");
		printf("\r                             ");
	printf("\r+++++++++++++++++++++++++++++++\n");
		
}

