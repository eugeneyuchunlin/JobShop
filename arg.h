#include <iostream>
#include <string>
#include <map>

class Arguments{
private:
	std::map<string, int> _arguments;	
public:
	void parseArgument(int argc, const char * argv[]);
	Arguments(std::map<string, int> initArguments);
};
