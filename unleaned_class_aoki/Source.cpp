#include"Header.h"
#include"utils.h"
#include<string>
#include<fstream>
#include<random>
#include<algorithm>
#include<iostream>
#include<stdexcept>
#include<chrono>
#include<thread>
int main() {
	// �f�[�^����荞�݁@�w�K�f�[�^�C���̐����N���X�̃f�[�^
	// NOTE:�w�K�f�[�^�̓f�[�^���~���͎����A�w�K�f�[�^�̃��x����one-hot�Ńf�[�^���~�N���X��
	//      �e�X�g�f�[�^�̓f�[�^���~���͎����A�e�X�g�f�[�^�̃��x����one-hot�Ńf�[�^���~(�N���X��+1)�B
	//      �������e�X�g�f�[�^�̃��x���ɂ��Ă�1��ڂ����w�K�f�[�^��2��ڂ���N���X1,2,3�Ƒ���
	const string data_folder = "C:\\Users\\uekus\\Documents\\FPGA\\OVRFPGA\\jupyter\\";
	const int tryId = 4;
	const int dataId = 0;
	const string learn_data_path = data_folder + "train_data.csv";
	const string learn_cls_path = data_folder  + "train_cls.csv";
	const string test_data_path = data_folder  + "test_data.csv";
	const string test_cls_path = data_folder   + "test_cls.csv";
	vector<vector<double>> learn_data   = get_vector_from_file<double>(learn_data_path);
	vector<vector<int>> class_data      = get_vector_from_file<int>(learn_cls_path);
	vector<vector<double>> test_data    = get_vector_from_file<double>(test_data_path);
	vector<vector<int>> test_class_data = get_vector_from_file<int>(test_cls_path);
	// ���f���̏���
	std::random_device rnd;     // �񌈒�I�ȗ���������ŃV�[�h�����@�𐶐�
	std::mt19937 engine(rnd());     //  �����Z���k�c�C�X�^�[��32�r�b�g�ŁA�����͏����V�[�h
	std::uniform_real_distribution<double> beta_0(0.001, 0.01);
	// Unlearn(int class_num, int component_num, double beta, double unlearn_mixing_degree,
	//     double normalize_unlearn, double beta_threshold, double delta_beta, double complementary_covar_coef);
	// beta:����, unlearn_mixing_degree:alpha_0 ����͓K���D����0.01,  normalize_unlearn:���w�K�N���X�̐��K�����D����͘_���̒l
	// beta_threshold:�������߂�Ƃ���臒l�D�_���̒l, beta_delta:�������߂�Ƃ��̕ω��ʁD�_���̒l, 
	// complementary_covar_coef:h_gauss���Z�o����Ƃ��̒l�D�_���̒l, output_dir:�p�����[�^�Ƃ����ʌ��ʂ��o�͂���f�B���N�g��
	const int    class_num         = 3;
	const int    component_num     = 2;
	const int    data_size         = 3;
	const double unlearn_mix_deg   = 0.01;
	const double normalize_unlearn = 0.5;
	const double beta_threshold    = 0.99;
	const double delta_beta        = 0.01;
	const double complementary_covar_coef = 20.0;

	const int repeat_num = 5;
	for (int re = 0; re < repeat_num; ++re) {
		// �f�[�^����荞�݁@�w�K�f�[�^�C���̐����N���X�̃f�[�^
		vector<vector<double>> learn_data = get_vector_from_file<double>(learn_data_path);
		vector<vector<int>> class_data = get_vector_from_file<int>(learn_cls_path);
		vector<vector<double>> test_data = get_vector_from_file<double>(test_data_path);
		vector<vector<int>> test_class_data = get_vector_from_file<int>(test_cls_path);
		Unlearn uln(class_num, component_num, data_size, beta_0(engine), unlearn_mix_deg,
			normalize_unlearn, beta_threshold, delta_beta, complementary_covar_coef,
			"C:\\Users\\uekus\\Documents\\FPGA\\OVRFPGA\\jupyter\\try" + to_string(tryId) + "\\data"+to_string(dataId)+"\\no_app\\");
		// �w�K�f�[�^���@beta�o�����߂̌��؃f�[�^�ƕ��ʂ̍Ŗޖ@�̂��߂̊w�K�f�[�^�ɕ�����D
		// �Ƃ肠�����w�K�f�[�^�̔��������ؗp�f�[�^�Ɏg��
		// NOTE:�O��Ƃ��Ċw�K�f�[�^�ɂ����ē����N���X�̃f�[�^��1��ɘA�����Ă���Ƃ���D
		vector<int> verification_index;
		vector<int> cls_data_num(class_num);
		for (int d = 0; d < learn_data.size(); d++) {
			cls_data_num[distance(class_data[d].begin(), max_element(class_data[d].begin(), class_data[d].end()))]++;
		}
		int class_first_index = 0;
		for (int cls = 0; cls < class_num; ++cls) {
			vector<int> class_index = make_rand_array_unique(cls_data_num[cls] / 2, 0, cls_data_num[cls] - 1);
			for (const int idx : class_index) {
				verification_index.push_back(class_first_index + idx);
			}
			class_first_index += cls_data_num[cls];
		}
		vector<vector<double>> verification_data;
		vector<vector<int>> verification_class_data;
		for (int i : verification_index) {
			verification_data.push_back(learn_data[i]);
			verification_class_data.push_back(class_data[i]);
		}
		// NOTE:�����r����indexError�o�Ȃ��悤�Ɍ�납������Ă܂��D
		for (int i = verification_index.size() - 1; i >= 0; --i) {
			learn_data.erase(learn_data.begin() + verification_index[i]);
			class_data.erase(class_data.begin() + verification_index[i]);
		}
		// �w�K�f�[�^���N���X���Ƃ�k-means�@�ɕ��荞��ŃN���X�����C
		// �N���X�C�R���|�[�l���g���ƂɃp�����[�^�̒l���o���D
		uln.calc_params(learn_data, class_data);
		uln.out_file_mean();
		uln.out_file_covar();
		uln.out_file_mix_deg();
		// �p�����[�^���g���Č��؃f�[�^�ƂƂ���beta���o���D
		uln.learn_beta(verification_data, verification_class_data);
		uln.out_file_params();
		uln.evaluate(test_data, test_class_data, true);
	}
	// �o�͂��ꂽ�p�����[�^���g���Ď��ʂ��s��
	// NOTE:�����͏o�͕��Ƃ̑Ή������Ă��邩�m���߂邱�ƁD
	//const string folder_path = "C:\\Users\\watanabe\\Desktop\\uekusa\\unlearned_files\\EMG_signals_for_forearm_classification\\try" + to_string(tryId) + "\\data"+to_string(dataId)+"\\no_app";
	//vector<string> file_names;
	//if (!getFileNames(folder_path, file_names)) {
	//	cerr << "Cannot find file names." << endl;
	//}
	//for (int par = 0; par < file_names.size(); ++par) {
	//	this_thread::sleep_for(std::chrono::seconds(1)); // NOTE:�����b�Ŏ��s�����folder�̖��O����error�f������
	//	const string file_directory = file_names[par];
	//	// prepare variables
	//	string filename;
	//	ifstream ifs;
	//	string line;
	//	// load parameters and beta, then create new instance.
	//	filename = file_directory + "\\params.csv";
	//	ifs.open(filename); if (ifs.fail()) { cerr << "can't opne " + filename << endl; exit(-1); }
	//	getline(ifs, line); getline(ifs, line); // NOTE: Pass headers.
	//	vector<string> load_params = split(line, ',');
	//	const int    class_num = stoi(load_params[0]);
	//	const int    component_num = stoi(load_params[1]);
	//	const int    data_size = stoi(load_params[2]);
	//	const double unlearn_mix_deg = stod(load_params[3]);
	//	const double normalize_unlearn = stod(load_params[4]);
	//	const double beta_threshold = stod(load_params[5]);
	//	const double delta_beta = stod(load_params[6]);
	//	const double ccomplementary_covar_coef = stod(load_params[7]);
	//	getline(ifs, line); getline(ifs, line);
	//	load_params = split(line, ',');
	//	vector<double> class_beta(class_num);
	//	for (int cls = 0; cls < class_num; ++cls) {
	//		class_beta[cls] = stod(load_params[cls]);
	//	}
	//	ifs.close();

	//	Unlearn uln(class_num, component_num, data_size, class_beta, unlearn_mix_deg, normalize_unlearn,
	//		beta_threshold, delta_beta, ccomplementary_covar_coef,
	//		"C:\\Users\\watanabe\\Desktop\\uekusa\\unlearned_files\\EMG_signals_for_forearm_classification\\try" + to_string(tryId) + "\\data" + to_string(dataId) + "\\pr_app\\");
	//	uln.load_file_mean_covar_mixdeg(file_directory);
	//	uln.set_approximate(true);

	//	uln.evaluate(test_data, test_class_data, true);
	//}
	for(int re=0;re<repeat_num;++re){
		// �f�[�^����荞�݁@�w�K�f�[�^�C���̐����N���X�̃f�[�^
		vector<vector<double>> learn_data = get_vector_from_file<double>(learn_data_path);
		vector<vector<int>> class_data = get_vector_from_file<int>(learn_cls_path);
		vector<vector<double>> test_data = get_vector_from_file<double>(test_data_path);
		vector<vector<int>> test_class_data = get_vector_from_file<int>(test_cls_path);
		Unlearn uln(class_num, component_num, data_size, beta_0(engine), unlearn_mix_deg,
			normalize_unlearn, beta_threshold, delta_beta, complementary_covar_coef,
			"C:\\Users\\uekus\\Documents\\FPGA\\OVRFPGA\\jupyter\\try" + to_string(tryId) + "\\data" + to_string(dataId) + "\\bo_app\\");
		// uln.set_approximate(true); // NOTE:�����ŋߎ��̂���Ȃ���؂�ւ���

		// �w�K�f�[�^���@beta�o�����߂̌��؃f�[�^�ƕ��ʂ̍Ŗޖ@�̂��߂̊w�K�f�[�^�ɕ�����D
		// �Ƃ肠�����w�K�f�[�^�̔��������ؗp�f�[�^�Ɏg��
		// NOTE:�O��Ƃ��Ċw�K�f�[�^�ɂ����ē����N���X�̃f�[�^��1��ɘA�����Ă���Ƃ���D
		vector<int> verification_index;
		vector<int>cls_data_num(class_num);
		for (int d = 0; d < learn_data.size(); d++) {
			cls_data_num[distance(class_data[d].begin(), max_element(class_data[d].begin(), class_data[d].end()))]++;
		}
		int class_first_index = 0;
		for (int cls = 0; cls < class_num; ++cls) {
			vector<int> class_index = make_rand_array_unique(cls_data_num[cls] / 2, 0, cls_data_num[cls] - 1);
			for (const int idx : class_index) {
				verification_index.push_back(class_first_index + idx);
			}
			class_first_index += cls_data_num[cls];
		}
		vector<vector<double>> verification_data;
		vector<vector<int>> verification_class_data;
		for (int i : verification_index) {
			verification_data.push_back(learn_data[i]);
			verification_class_data.push_back(class_data[i]);
		}
		// �����r����indexError�o�Ȃ��悤�Ɍ�납������Ă܂��D
		for (int i = verification_index.size() - 1; i >= 0; --i) {
			learn_data.erase(learn_data.begin() + verification_index[i]);
			class_data.erase(class_data.begin() + verification_index[i]);
		}
		// �w�K�f�[�^���N���X���Ƃ�k-means�@�ɕ��荞��ŃN���X�����C
		// �N���X�C�R���|�[�l���g���ƂɃp�����[�^�̒l���o���D
		uln.calc_params(learn_data, class_data);
		uln.out_file_mean();
		uln.out_file_covar();
		uln.out_file_mix_deg();
		// �p�����[�^���g���Č��؃f�[�^�ƂƂ���beta���o���D
		uln.learn_beta(verification_data, verification_class_data);
		uln.out_file_params();
		uln.evaluate(test_data, test_class_data, true);
	}

	// GIVE UP:�e�X�g�f�[�^����邱�Ƃ��o���Ȃ������̂ŁCtest()������̂͂�����߂�
	// test()
	//Unlearn uln(3, 2, 0.01, 0.01, 0.5, 0.99, 0.01, 10);
	// uln.k_means_test();
	// uln.gauss_test();
	//uln.hgauss_test();
	//uln.get_mean_covar_test();
	return 0;
}
