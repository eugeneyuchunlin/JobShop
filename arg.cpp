#include "./arg.h"

Arguments::Arguments(std::map<string, int> initArguments){
	this->_arguments = initArguments
}

void Arguments::parseArgument(int argc, const char * argv[]){
	for(int i = 0; i < argc; ++i){
		std::cout<<argv[i]<<std::endl;
	}	
}
