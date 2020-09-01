#include "configure.h"

vector<map<int, int> > Configure(const char * filename,int & lineElementAmount){
	vector<map<int, int> > configs;
	map<int,int> *tempVector;
	int temp;
	FILE * file;
	file = fopen(filename, "r");
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

	return configs;	
}
