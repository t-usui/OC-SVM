#include "main.h"

int main(int argc, char **argv){
	char *save_file_name = NULL;
	char *load_file_name = NULL;
	char *training_file_name;
	char *test_file_name;

/*
	for(int i=1;i<argc;i++){
		if(strcmp(argv[i], "-s") == 0){
			save_file_name = argv[++i];
		}else if(strcmp(argv[i], "-l") == 0){
			load_file_name = argv[++i];
		}

		if(i == argc-2){
			training_file_name = argv[i];
			test_file_name = argv[++i];
		}
	}
*/
			
	classifier::SVC *svc = new classifier::SVC;

//	if(load_file_name != NULL){
//		svc->LoadModel(load_file_name);
//	}else{
//		svc->InputTrainingData();
//	}

//	if(save_file_name != NULL){
//		svc->SaveModel(save_file_name);
//	}

	svc->InputTrainingData("training.txt");
	svc->InputTestData("test.txt");
	svc->Classify();

	delete svc;

	return 0;
}

