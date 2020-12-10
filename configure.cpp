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

map<string, map<string, int> > EQP_TIME(const char * filename){
	map<string, map<string, int> > Data;
	fstream file;
	file.open(filename);
	string temp;
	string EQP, RECIPE;
	int processTime;
	if(!file.good()){
		cout<<"fail to open file"<<endl;	
	}else{
		file>>temp>>temp>>temp;
		while(!file.eof()){
			file>>EQP>>RECIPE>>processTime;
			// cout<<EQP<<RECIPE<<processTime<<endl;
			Data[RECIPE][EQP] = processTime;
		}
	}
	return Data;
}

map<string, vector<string> > STATUS(const char * filename){
	map<string, vector<string> > Data;
	fstream file;
	file.open(filename);
	string temp;
	string EQP, STATUS, recorverTime;
	if(!file.good()){
		cout<<"fail to open file"<<endl;	
	}else{
		file>>temp>>temp>>temp;
		while(!file.eof()){
			file>>EQP>>STATUS>>recorverTime;
			// cout<<EQP<<RECIPE<<recorverTime<<endl;
			Data[EQP].push_back(STATUS);
			Data[EQP].push_back(recorverTime);
		}
	}
	return Data;
}

vector<vector<string> > vSTATUS(const char * filename){
	map<string, vector<string> > Data = STATUS(filename);
	vector<vector<string> > vData;
	for(map<string, vector<string> >::iterator it = Data.begin(); it != Data.end(); it++){
		vData.push_back(it->second);
	}
	return vData;
}



vector<string> split_string(string headLine, string delimiter){
	vector<string> splitStrings;
	string substr;
	size_t startPos = 0;
	size_t found;
	
	do{
		found = headLine.find(delimiter, startPos);	
		substr = headLine.substr(startPos, found - startPos);
		startPos = found + 1;
		splitStrings.push_back(substr);
	}while(found != string::npos);
	
	return splitStrings;
}


vector<map<string, string> > WIP(const char * filename){
	vector<map<string, string> > Data;
	vector<string> head;
	vector<string> content;
	map<string, string> temp;
	string line;
	string headline;
	fstream file;
	file.open(filename);
	if(!file.good()){
		cout<<"fail to open file"<<endl;
	}else{
		getline(file, headline);	
		head = split_string(headline, "\t");
		while(getline(file, line)){
			content = split_string(line, "\t");
			temp.clear();
			for(unsigned int i = 0; i < head.size(); ++i){
				temp[head[i]] = content[i];	
			}
			Data.push_back(temp);
		}
	
	}	
	return Data;
}

vector<vector<int> > SETUP_TIME(const char * filename){
	vector<vector<int> > data;
	vector<int> temp;
	vector<string> lineData;
	string headline;
	string line;
	fstream file;
	file.open(filename);
	if(!file.good()){
		cout<<"fail to open file"<<endl;
	}else{
		getline(file, headline);	
		while(getline(file, line)){
			lineData = split_string(line, "\t");
			lineData.erase(lineData.begin());
			temp.clear();
			for(unsigned int i = 0; i < lineData.size(); ++i){
				temp.push_back(stoi(lineData[i]));	
			}
			data.push_back(temp);
		}		
		
	}


	return data;
}



