#ifndef __SVC_H__
#define __SVC_H__

#include "svm.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace classifier{
	class SVC{
		private:
			// static const int kSvmProblemNumber = 100;
			int svm_problem_number_;
			double result_;

			svm_parameter svm_parameter_;
			svm_problem svm_problem_;
			svm_model *svm_model_;
			struct svm_node **training_data_;
			struct svm_node **test_data_;

			void SetupSvmParameter();
			void CheckSvmParameter();
			int CountData(char *file_name);
			void ParseData(std::string line, int *label, std::vector<int> *index, std::vector<double> *value);
			struct svm_node **InputData(char *file_name);
			void BuildSvmProblem();
			void BuildSvmModel();
			void PredictResult();
			void FinishSvm();

		public:
			SVC();
			~SVC();

			void InputTrainingData(char *file_name);
			void InputTestData(char *file_name);
			void SaveModel(char *filename);
			double Classify();
	};
}

#endif /* __SVC_H__ */

