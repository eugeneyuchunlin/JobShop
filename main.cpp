#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <map>
#include <opencv2/videoio.hpp>
#include <pthread/qos.h>
#include <string>
#include <sys/signal.h>
#include <vector>
#include <fstream>
#include <ctime>

#include "job.h"
#include "chromosome.h"
#include "machine.h"
#include "gantt.h"

using namespace cv;
using namespace std;

void genetic_algorithm_initialize(
		vector<ChromosomeLinker> & linkers, 
		int size
);

Chromosome genetic_algorithm(
		vector<Chromosome> & Chromosomes, 
		vector<Job *> & Jobs, 
		map<int, Machine> & Machines, 
		const int JOB_AMOUNT, 
		const int MACHINE_AMOUNT,
		const double INTERCROSS_RATE,
		const double MUTATION_RATE,
		const double ELITIST_RATE,
		const double RANK_SELECTION_RATE,
		const clock_t MAXTIME
);

void progress_bar(double progress);

void crossover(const double RATE, Chromosome* parants,int parants_size, vector<Chromosome> & temp);

void mutation(const double RATE, Chromosome* parants,int parants_size, vector<Chromosome> & temp);

vector<ChromosomeLinker> RouletteWheelSelection(vector<ChromosomeLinker> & chromosomeLinker, int amounts, int nextGenerationAmount);

int random_int(int start, int end, int different_num=-1);


int main(int argc, const char * argv[]) {

	// usage : ./Main config.txt CHROMOSOME_AMOUNT GATIMES CROSSOVER_RATE MUTATION_RATE 
	
	const int CHROMOSOME_AMOUNT = atoi(argv[2]);
	const int GA_TIMES = atoi(argv[3]);	
	const double GA_INTERCROSS_RATE = atof(argv[4]);
	const double GA_MUTATE_RATE = atof(argv[5]);
	
	// Declare Objects
	vector<Job *> Jobs;
	map<int, Machine>Machines;
	vector<Chromosome> Chromosomes;
	clock_t t1 = clock();
	// load configuration file
	vector<map<int, int> > configs;
	map<int,int> *tempVector;
	int temp;
	int lineElementAmount;
	srand(time(NULL));
	FILE * file;	
	file = fopen(argv[1], "r");
	fscanf(file, "%d", &lineElementAmount);	
	for(int i = 0; fscanf(file, "%d", &temp) != EOF; ++i){
		if(i == 0)
			tempVector = new map<int, int> ();

		if(temp != -1){
			tempVector->operator[](i) = temp;
		}

		if(i == (lineElementAmount - 1)){
			i = -1;
			configs.push_back(*tempVector);
		}
	}
	const int JOB_AMOUNT = configs.size();
	const int MACHINE_AMOUNT = lineElementAmount;

	// cout<<"create all instances"<<endl;
	Chromosome temp_chromosome;
	// Step 1. create all instances
	for(int i = 0; i < CHROMOSOME_AMOUNT; ++i){
		temp_chromosome = Chromosome(JOB_AMOUNT);
		Chromosomes.push_back(temp_chromosome);
	}

	// cout<<"end of create all instances"<<endl;

	// create Jobs instance
	for(unsigned int i = 0; i < configs.size(); ++i){
		Jobs.push_back(new Job(i, configs[i]));	
	}
	
	for(int i = 0; i < lineElementAmount; ++i){
		Machines[i] = Machine(i);			
	}
	// Chromosomes[0].__repr__();
	// Chromosome ch = !Chromosomes[0];
	// cout<<"=++++++++++++++"<<endl;
	// Chromosomes[0].__repr__();
	// ch.__repr__();

		

	// start GA
	Chromosome bestSolution;
	bestSolution = genetic_algorithm(
			Chromosomes, 
			Jobs, 
			Machines, 
			JOB_AMOUNT, 
			MACHINE_AMOUNT,
			GA_INTERCROSS_RATE,
			GA_MUTATE_RATE,
			0.2,
			0.8,
			GA_TIMES * CLOCKS_PER_SEC
	);
	
	// reconstructing
	// machine clear
	for(int j = 0; j < MACHINE_AMOUNT; ++j){
		Machines[j].clear();
	}
	for(int j = 0; j < JOB_AMOUNT; ++j){
		Jobs[j]->clear();
		Jobs[j]->assign_machine_number(bestSolution.getMachine(Jobs[j]->get_number()));
		Jobs[j]->assign_machine_order(bestSolution.getOrder(Jobs[j]->get_number()));
	}
	// Step 4. Machines get the job
	for(int j = 0; j < JOB_AMOUNT; ++j){
		Machines[Jobs[j]->get_machine_number()].add_job(Jobs[j]);
	}
	
	int maxMachineTime = 0;
	int whichMachine;
	for(int i = 0; i < lineElementAmount; ++i){
		Machines[i].sort_job();
		temp = Machines[i].get_total_time();
		if(temp > maxMachineTime){
			maxMachineTime = temp;
			whichMachine = i + 1;
		}
	}
	cout<<"max time = "<<maxMachineTime<<endl;
	
	GanttChart chart(maxMachineTime); // Gantt Chart
	for(int i = 0; i < lineElementAmount; ++i){
		Machines[i].add_into_gantt_chart(chart);
		// Machines[i].demo();
		chart.set_time(i + 1, Machines[i].get_total_time());
	}
	
	clock_t t2 = clock();
	cout<< t2 - t1<<endl;
	cout<<(double)(t2 - t1) / (double)CLOCKS_PER_SEC<<endl;
	
	Mat mat = chart.get_img();
	namedWindow("Gantt Chart", WINDOW_AUTOSIZE );
    imshow("Gantt Chart", mat);
	waitKey(0);

	return 0;
}

void genetic_algorithm_initialize(vector<ChromosomeLinker> &linkers, int size){
	ChromosomeLinker temp;	
	for(int i = 0; i < size; ++i){
		linkers.push_back(ChromosomeLinker());
		linkers[i].link_num = i;
		linkers[i].value = 65535;
	}
}

void progress_bar(double progress){
	printf("\033[A");
	printf("\r                                                                                                   ");	

	printf("\r[");
	for(int i = 0, max = progress - 2; i < max; i += 2){
		printf("= ");	
	}
	if(progress < 100.0)
		printf("=>");
	else
		printf("= ");
	for(int i = progress + 2; i <= 100; i += 2){
		printf("· ");
	}

	printf("]%.2f", progress);
	cout<<"%";
	printf("\n");
}

Chromosome genetic_algorithm(
		vector<Chromosome> & Chromosomes, 
		vector<Job *> & Jobs, 
		map<int, Machine> & Machines, 
		const int JOB_AMOUNT,
		const int MACHINE_AMOUNT, 
		const double INTERCROSS_RATE,
		const double MUTATION_RATE, 
		const double ELITIST_RATE,
		const double RANK_SELECTION_RATE,
		const clock_t MAXTIME		
){
	int chromosomes_size = Chromosomes.size();
	vector<ChromosomeLinker> linkers;
	genetic_algorithm_initialize(linkers, chromosomes_size * 2);
	vector<Chromosome> temp;
	vector<Chromosome> children;
	Chromosome * chromosomes_ptr = new Chromosome[Chromosomes.size()];
	Chromosome bestSolution;
	int minTime = 65535;
	clock_t startTime, time;
	clock_t endTime = clock() + MAXTIME;
	cout<<endl;
	
	for(int i = 0; i < chromosomes_size; ++i){
		chromosomes_ptr[i] = Chromosomes[i];
	}
	
	for(startTime = time = clock(); time < endTime; time = clock()){
		
		temp.clear();
		
		// Step 1. crossover and mutation
		crossover(INTERCROSS_RATE, chromosomes_ptr, chromosomes_size, temp);
		mutation(MUTATION_RATE, chromosomes_ptr, chromosomes_size, temp);
		for(int i = 0; i < chromosomes_size; ++i){
			temp.push_back(chromosomes_ptr[i]);
		}

		// Step 2. assign machine number and order
		int childrenAmmount = temp.size();
		int maxTime, tempTime;
		for(int i = 0; i < childrenAmmount;  ++i){

			// machine clear
			for(int j = 0; j < MACHINE_AMOUNT; ++j){
				Machines[j].clear();
			}
			maxTime = 0;
			for(int j = 0; j < JOB_AMOUNT; ++j){
				Jobs[j]->clear();
				Jobs[j]->assign_machine_number(temp[i].getMachine(Jobs[j]->get_number()));
				Jobs[j]->assign_machine_order(temp[i].getOrder(Jobs[j]->get_number()));
			}
			// Step 4. Machines get the job
			for(int j = 0; j < JOB_AMOUNT; ++j){
				Machines[Jobs[j]->get_machine_number()].add_job(Jobs[j]);
			}

			for(int j = 0; j < MACHINE_AMOUNT; ++j){
				Machines[j].sort_job();
				tempTime = Machines[j].get_total_time();
				if(tempTime > maxTime)
					maxTime = tempTime;
			}

			linkers[i].value = maxTime;
			linkers[i].linkChromosome = &temp[i];
			linkers[i].link_num = i;
			temp[i].value = maxTime;
		}

		
		progress_bar(100 * ((double)(time - startTime) / (double)(endTime - startTime)));
		sort(linkers.begin(), linkers.end(), chromosomelinker_comparator);
		RouletteWheelSelection(linkers, childrenAmmount, chromosomes_size);
		for(int j = 0; j < chromosomes_size; ++j){
			chromosomes_ptr[j] = *linkers[j].linkChromosome;
		}

		bestSolution = chromosomes_ptr[0];
		minTime = linkers[0].value;

	}	


	progress_bar(100.0);
	delete [] chromosomes_ptr;
	return bestSolution;	
}

void crossover(const double RATE, Chromosome* parants,int parants_size,  vector<Chromosome> & temp){
	// random_shuffle(parants.begin(), parants.end());
	int newSize = round(parants_size * RATE / 2.0);
	vector<Chromosome> children;
	int num1, num2;
	for(int i = 0; i < newSize; ++i){
		num1 = random_int(0, parants_size);
		num2 = random_int(0, parants_size, num1);
		children = parants[num1] * parants[num2];
		temp.push_back(children[0]);
		temp.push_back(children[1]);
	}

	
	
}


void mutation(const double RATE, Chromosome* parants,int parants_size, vector<Chromosome> & temp){
	int newSize = round(parants_size * RATE);
	Chromosome child;
	int num1;
	for(int i = 0; i < newSize; ++i){
		num1 = random_int(0, parants_size);
		// random_shuffle(parants.begin(), parants.end());
		child = !parants[num1];
		temp.push_back(child);
	}
}

int random_int(int start, int end, int different_num){
	if(different_num < 0){
		return start + rand() % (end - start);
	}else{
		int rnd = start + (rand() % (end - start));

		while(rnd == different_num){
			rnd = start + (rand() % (end + start));
		}
		return rnd;
	}
}

vector<ChromosomeLinker> RouletteWheelSelection(vector<ChromosomeLinker> &  chromosomeLinker, int childrenAmounts, int nextGenerationAmount){
	// vector<double> values;
	double * values = new double[childrenAmounts];
	vector<ChromosomeLinker> nextGeneration;
	vector<double> randomValues;

	double total = 0.0;
	double Max = 0.0;	

	for(unsigned int i = 0, size = childrenAmounts; i < size; ++i){
		if(chromosomeLinker[i].value > Max){
			Max = chromosomeLinker[i].value;
		}
	}

	for(unsigned int i = 0, size = childrenAmounts; i < size; ++i){
		total += (Max - chromosomeLinker[i].value);
	}

	for(unsigned int i = 0, size = childrenAmounts; i < size; ++i){
		values[i] = (Max - chromosomeLinker[i].value) / total;
	}

	for(unsigned int i = 1, size =  childrenAmounts; i < size; ++i){
		values[i] += values[i - 1];
	}

		
	for(int i = 0; i < nextGenerationAmount; ++i){
		randomValues.push_back((double)rand() / (RAND_MAX + 1.0));	
	}

	for(int i = 0; i < nextGenerationAmount; ++i){
		for(int j = 0; j < childrenAmounts; ++j){
			if(randomValues[i] < values[j]){
				nextGeneration.push_back(chromosomeLinker[j]);
				break;
			}
		} 		
	}

	for(int i = nextGenerationAmount; i < childrenAmounts; ++i)
		nextGeneration.push_back(chromosomeLinker[i]);
	
	delete[] values;
	return nextGeneration;

}
