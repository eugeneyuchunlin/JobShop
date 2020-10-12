#include "job_base.h"

Job_base::Job_base(){
	last = next = nullptr;
}

void Job_base::set_last(Job_base * nlast){
	last = nlast;
}

Job_base * Job_base::get_last(){
	return last;
}

void Job_base::set_next(Job_base * nnext){
	next = nnext;
}

Job_base * Job_base::get_next(){
	return next;
}
