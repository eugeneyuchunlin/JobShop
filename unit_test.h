#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "cuChromosome.h"
#include "cuJob.h"
#include "cuMachine.h"
#include <cctype>


/**	bool test_can_run_tools_successfully_copy
 * 	Test DEV_CAN_RUN_TOOLS and CAN_RUN_TOOLS are the same
 * 	DEV_CAN_RUN_TOOLS is copied from host to device
 * 	In this test function, it will perform testing DEV_CAN_RUN_TOOLS and CAN_RUN_TOOLS are the same.
 *	
 *	@param unsigned int ** CAN_RUN_TOOLS
 *	@param unsigned int ** DEV_CAN_RUN_TOOLS
 *	@param scuJob *** jobs,
 *	@param unsigned int NUMOF_JOBS
 *	
 *	@return true if test is successfully finished
 */
bool test_can_run_tools_successfully_copy(
		unsigned int ** CAN_RUN_TOOLS, 
		unsigned int ** DEV_CAN_RUN_TOOLS, 
		scuJob ***jobs, 
		unsigned int NUMOF_JOBS
);


/** bool test_process_time_successfully_copy
 *	Test PROCESS_TIME and DEV_PROCESS_TIME are the same
 *	DEV_PROCESS_TIME is copied from host to device
 *	In this test function, it will perform testing DEV_PROCESS_TIME and PROCESS_TIME are the same.
 *	
 *	@param double ** PROCESS_TIME
 *	@param double ** DEV_PROCESS_TIME
 *	@param scuJob ** jobs
 *	@param unsigned int NUMOF_JOBS
 *
 *	@return true if test is successfully finished
 *
 */
bool test_process_time_successfully_copy(
		double ** PROCESS_TIME, 
		double ** DEV_PROCESS_TIME, 
		scuJob *** jobs, 
		unsigned int NUMOF_JOBS
);

/** bool test_jobs_are_the_same
 *	test two jobs are the same
 *	test the following information:
 *		job->number
 *		job->sizeof_can_run_tools
 *		job->sizeof_process_time
 *
 * @param scuJob * job1
 * @param scuJob * job2
 *
 * @return true if test is successfully finished
 */
bool test_jobs_are_the_same(scuJob * job1, scuJob*job2);



/** bool test_job_successfully_copy
 *	test jobs are successfully copied from host to device
 *
 * 	@param scuJob *** jobs
 * 	@param scuJob ** dev_jobs
 * 	@param unsigned int NUMOF_CHROMOSOMES
 * 	@param unsigned int NUMOF_JOBS
 *
 * 	@return true if test is successfully finished
 *
 */
bool test_job_successfully_copy(
		scuJob *** jobs,
		scuJob ** dev_jobs,
		unsigned int NUMOF_CHROMOSOMES,
		unsigned int NUMOF_JOBS
);


/**	bool test_machines_are_the_same
 * 	test two machines are the same
 * 	tes the following information:
 * 		scuMachine->recover_time
 * 		scuMachine->number
 *	
 *	@param scuMachine * m1
 *	@param scuMachine * m2
 *
 *	@return true if those data are the same
 *
 */
bool test_machines_are_the_same(scuMachine * m1, scuMachine * m2);


/** bool test_machines_successfully_copy
 * 	test machines are successfully copied from host to device
 *	
 *	@param scuMachine *** machines
 *	@param sucMachine ** dev_machines
 *	@param unsigned int NUMOF_MACHINES
 *	@unsigned int NUMOF_CHROMOSOMES
 *
 *	@return true if test is successfully finished
 *
 */
bool test_machines_successfully_copy(
		scuMachine *** machines,
		scuMachine ** dev_machines,
		unsigned int NUMOF_MACHINES,
		unsigned int NUMOF_CHROMOSOMES
);

/** bool test chromosomes initialization
 *	look up the chromosomes are initialized successfully
 *	
 *	@param scuChromosome * dev_chromosomes
 *	@param unsigned int NUMOF_JOBS
 *	@param unsigned int NUMOF_CHROMOSOMES
 *
 *	@return true if pass the test
 */
bool test_chromosome_initialize(
		scuChromosome * dev_chromosomes,
		unsigned int NUMOF_JOBS,
		unsigned int NUMOF_CHROMOSOMES	
);

/** void testing copy genes
 *
 *
 */
void test_copying_genes(
		scuChromosome * dev_chromosomes,
		double ** DEV_ARRAY,
		double ** HOST_ARRAY,
		unsigned int NUMOF_JOBS,
		unsigned int NUMOF_CHROMOSOMES
);

bool test_setupt_time_successfully_copied(
		unsigned int ** HOST_SETUP_TIME,
		unsigned int ** DEV_SETUP_TIME,
		unsigned int NUMOF_JOBS
);

#endif
