#ifndef JOB_BASE_H
#define JOB_BASE_H

class Job_base{
private:
	Job_base * last;
	Job_base * next;	

public:
	Job_base();
	void set_last(Job_base *);
	void set_next(Job_base *);
	Job_base * get_next();
	Job_base * get_last();

	virtual double get_start_time()=0;
	virtual double get_end_time()=0;
	virtual void set_start_time(double time)=0;
	virtual void clear()=0;
};


#endif
