#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <fstream>

using namespace std;

vector<map<int, int> > Configure(const char * filename, int & lineElementAmount);

map<string, map<string, int> > EQP_TIME(const char * filename);

map<string, vector<string> > STATUS(const char * filename);
vector<vector<string> > vSTATUS(const char * filename);

vector<map<string, string> > WIP(const char * filename);

vector<vector<int> > SETUP_TIME(const char * filename);

