#include<iostream>
#include <string>

using namespace std;

int main(){
	
	string text = "aaaa\tbbbb\tcccc";
	string substr;
	size_t startPos, found;
	startPos = 0;
	do{
		found = text.find("\t", startPos);
		substr = text.substr(startPos, found - startPos);
		cout<<substr<<endl;
		startPos = found + 1;
	}while(found != string::npos);

}
