#include "svc.h"

// #define DEBUG

namespace classifier{
	SVC::SVC(){
		std::cout << "Starting up Beelzebub's Support Vector Classifier...";
		this->SetupSvmParameter();

		std::cout << " Done." << std::endl ;
		std::cout << "LibSVM version: " << libsvm_version << std::endl;
	}


	SVC::~SVC(){
		std::cout << "Exitting Support Vector Classifier." << std::endl;
		delete this->svm_problem_.x;
		delete this->svm_problem_.y;
	}


	void SVC::SetupSvmParameter(){
		std::cout << "Setting up parameters... ";

		this->svm_parameter_.svm_type		= ONE_CLASS;
		this->svm_parameter_.kernel_type	= RBF;
		this->svm_parameter_.degree			= 3;
		this->svm_parameter_.gamma			= 0.5;
		this->svm_parameter_.coef0			= 0;
		this->svm_parameter_.nu				= 0.1;
		this->svm_parameter_.cache_size		= 100;
		this->svm_parameter_.C				= 1;
		this->svm_parameter_.eps			= 0.001;
		this->svm_parameter_.p				= 0.1;
		this->svm_parameter_.shrinking		= 1;
		this->svm_parameter_.probability	= 0;
		this->svm_parameter_.nr_weight		= 0;
		this->svm_parameter_.weight_label	= NULL;
		this->svm_parameter_.weight			= NULL;

/*
		this->svm_parameter_.svm_type		= NU_SVC;
		this->svm_parameter_.kernel_type	= RBF;
		this->svm_parameter_.degree		= 2;
		this->svm_parameter_.gamma		= 0.2;
		this->svm_parameter_.coef0		= 1;
		this->svm_parameter_.nu			= 0.2;
		this->svm_parameter_.cache_size		= 100;
		this->svm_parameter_.C			= 100;
		this->svm_parameter_.eps		= 0.00001;
//		this->svm_parameter_.eps		= 1e-3;
		this->svm_parameter_.p			= 0.1;
		this->svm_parameter_.shrinking		= 0.1;
		this->svm_parameter_.probability	= 0;
		this->svm_parameter_.nr_weight		= 0;
		this->svm_parameter_.weight_label	= NULL;
*/
		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::CheckSvmParameter(){
		std::string ret = "return value of svm_check_parameter.";

		std::cout << "Checking parameters... ";

		if(svm_check_parameter(&this->svm_problem_, &this->svm_parameter_) == NULL){
			std::cout << "Done. NO problem." << std::endl;
		}else{
			std::cerr << "Error: checking parameter" << ret << std::endl;
			exit(EXIT_FAILURE);
		}

		return;
	}


	// Seekなどを用いて高速化すること
	int SVC::CountData(char *file_name){
		int count;
		std::ifstream ifs;
		std::string line;

		ifs.open(file_name);

		for(count=0;getline(ifs, line)!=NULL;count++);

		return count;
	}


	void SVC::ParseData(std::string line, int *label, std::vector<int> *index, std::vector<double> *value){
		std::vector<std::string> space_split_result;
		std::vector<std::string> colon_split_result;

		std::cout << line << std::endl;

		// ex. split "0 1:2 3:4 5:6" into { "0", "1:2", "3:4", "5:6" }
		boost::algorithm::split(space_split_result, line, boost::is_any_of(" "));

		try{
			for(int i=0;i<(int)space_split_result.size();i++){
				// ex. split "1:2" into { "1", "2" }
				boost::algorithm::split(colon_split_result, space_split_result[i], boost::is_any_of(":"));
				if(colon_split_result.size() == 1){
					*label = boost::lexical_cast<int>(colon_split_result[0]);
				}else{
					index->push_back(boost::lexical_cast<int>(colon_split_result[0]));
					value->push_back(boost::lexical_cast<double>(colon_split_result[1]));
				}
			}
		}catch(boost::bad_lexical_cast){
			std::cout << "Error: lexical_cast";
			exit(EXIT_FAILURE);
		}

		return;	
	}


	// struct svm_node **dataを用いるように作り替え（未デバッグ）
	struct svm_node **SVC::InputData(char *file_name){
		struct svm_node **data;

		std::ifstream ifs;
		std::string line;

		int label;
		std::vector<int> index;
		std::vector<double> value;

		ifs.open(file_name);

		this->svm_problem_number_ = this->CountData(file_name);

		data = new struct svm_node*[this->svm_problem_number_];

		for(int i=0;getline(ifs, line)!=NULL;i++){
			index.clear();
			value.clear();
			this->ParseData(line, &label, &index, &value);
		
			#ifdef DEBUG
				//std::cout << "Label: " << label << std::endl;
				for(int j=0;j<(int)index.size();j++) std::cout << index[j] << ":" << value[j] << " ";
				std::cout << std::endl;
			#endif

			data[i] = new struct svm_node[index.size() + 1];

			for(int j=0;j<(int)index.size();j++){
				data[i][j].index = index[j];
				data[i][j].value = value[j];
			}
			data[i][index.size()].index = -1;
		}
		
		return data;
	}


	void SVC::InputTrainingData(char *file_name){
		std::cout << "Inputting training data... ";
		this->training_data_ = this->InputData(file_name);
		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::InputTestData(char *file_name){
		std::cout << "Inputting test data... ";
		this->test_data_ = this->InputData(file_name);
		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::BuildSvmProblem(){
		std::cout << "Building problems... ";

		this->svm_problem_.l = this->svm_problem_number_;
		this->svm_problem_.y = new double[this->svm_problem_.l];

		for(int i=0;i<this->svm_problem_.l;i++){
			this->svm_problem_.y[i] = 1;
		//	this->svm_problem_.y[i] = this->training_label_[i];
		}


		this->svm_problem_.x = new svm_node*[this->svm_problem_.l];
		for(int i=0;i<this->svm_problem_.l;i++){
			this->svm_problem_.x[i] = this->training_data_[i];
		}

		std::cout << "Done." << std::endl;

		return;
	}


	void SVC::BuildSvmModel(){
		int class_number = 0;

		std::cout << "Building model... ";

		this->svm_model_ = svm_train(&this->svm_problem_, &this->svm_parameter_);
		class_number =  svm_get_nr_class(this->svm_model_);

		std::cout << "Done. The number of class: " << class_number << std::endl;

		return;
	}


	void SVC::SaveModel(char *filename){
		svm_save_model(filename, this->svm_model_);

		return;
	}


	void SVC::PredictResult(){
		this->result_ = svm_predict(this->svm_model_, this->test_data_[0]);

		return;
	}


	double SVC::Classify(){
		this->BuildSvmModel();
		this->BuildSvmProblem();
		this->CheckSvmParameter();
		this->PredictResult();

		#ifdef DEBUG
			std::cout << "***************************" << std::endl;
			//std::cout << "Result of classification: " << result << std::endl;
			std::cout << "***************************" << std::endl;
		#endif

		std::cout << this->result_ << std::endl;

		return this->result_;
	}
}

