#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float randomf(unsigned int end){
	return (float)rand()/((float)RAND_MAX/end);
}

int main(){
	srand(time(NULL));
	for(unsigned int i = 0; i < 10; ++i){
		printf("%.3f\n", randomf(1));
	}
}
