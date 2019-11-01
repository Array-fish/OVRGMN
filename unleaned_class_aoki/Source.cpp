/*
2019/06/04
�o�͂܂ł͏���������C�덷�֐����o���Ƃ���������Ăق����D
�܂����̊m�������o���Ă��Ȃ�����ˁD
�ŏI�I�ȖڕW�͊e�p�����[�^���ǂ�Ȗ������ʂ��������邱��
������O���t���o����������mean,covar�Ƃ��̒l���ǂ����̃t�@�C���ɕۑ����Ă�肽��
2019/06/11
�r������energy�����ׂ�0�ɂȂ邩����̊w�K���o���ĂȂ��D
������T��C���͂ƕ��ς͑��v���������番�U������������
2018/06/18
�}�c����ɂǂ����Ȃ�S��Python�ŏ��������ƌ���ꂽ�̂ł��̃R�[�h���ێʂ��������C
����ɔ����C���t�@�N�^�����O�����܂��D(�������������������ĂȂ����ǁD)
���Ƃ��Ƃ̃R�[�h�̓t�@�C���ɂ܂Ƃ߂ăv���W�F�N�g�̃f�B���N�g���̂ǂ�������ɂ����Ă����܂��D
*/
#include"Header.h"
#include<string>
#include<fstream>
#include<random>
#include<algorithm>
#include<iostream>
vector<vector<double>> get_vector_from_file(const string filename);
vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);
int main() {
	// �f�[�^����荞�݁@�w�K�f�[�^�C���̐����N���X�̃f�[�^
	vector<vector<double>> learn_data = get_vector_from_file("lea_sig.csv");
	vector<vector<double>> class_data = get_vector_from_file("lea_class.csv");
	vector<vector<double>> test_data = get_vector_from_file("dis_sig2.csv");
	vector<vector<double>> test_class_data = get_vector_from_file("dis_class.csv");
	// �w�K�f�[�^���@beta�o�����߂̌��؃f�[�^�ƕ��ʂ̍Ŗޖ@�̂��߂̊w�K�f�[�^�ɕ�����D
	// �Ƃ肠�����w�K�f�[�^�̔��������ؗp�f�[�^�Ɏg��
	vector<int> verification_index = make_rand_array_unique(learn_data.size() / 2, 0, learn_data.size() - 1);
	vector<vector<double>> verification_data;
	vector<vector<double>> verification_class_data;
	for (int i : verification_index) {
		verification_data.push_back(learn_data[i]);
		verification_class_data.push_back(class_data[i]);
	}
	// �����r����indexError�o�Ȃ��悤�Ɍ�납������Ă܂��D
	for (int i = verification_index.size() - 1; i >= 0; --i) {
		learn_data.erase(learn_data.begin() + verification_index[i]);
		class_data.erase(class_data.begin() + verification_index[i]);
	}
	// ���f���̏���
	const int class_num = 3;
	const int component_num = 2;
	std::random_device rnd;     // �񌈒�I�ȗ���������ŃV�[�h�����@�𐶐�
	std::mt19937 engine(rnd());     //  �����Z���k�c�C�X�^�[��32�r�b�g�ŁA�����͏����V�[�h
	std::uniform_real_distribution<double> beta_0(0.001, 0.01);
	// Unlearn(int class_num, int component_num, double beta, double zeta, double psi, double beta_threshold, double delta_beta, double epsilon);
	// beta:����, zeta:alpha_0 ����͓K���D����0.01, psi:���w�K�N���X�̐��K�����D����͘_���̒l
	// beta_threshold:�������߂�Ƃ���臒l�D�_���̒l, beta_delta:�������߂�Ƃ��̕ω��ʁD�_���̒l, epsilon:h_gauss���Z�o����Ƃ��̒l�D�_���̒l,
	//double beta0 = ;
	Unlearn uln(class_num, component_num, beta_0(engine), 0.01, 0.5, 0.99, 0.01, 10);
	//Unlearn uln(class_num, component_num, 8, 0.01, 0.5, 0.99, 0.1, 10);// ������ƊȒP�ɂ��邽�߂�beta_0��delta_beta��ς��Ă���̂��{��
	// �w�K�f�[�^���N���X���Ƃ�k-means�@�ɕ��荞��ŃN���X�����C
	// �N���X�C�R���|�[�l���g���ƂɃp�����[�^�̒l���o���D
	// class_data_alt��calc_params()�֐�����������̋C���ŃN���X�̓��͂�index�ɂ��Ă��܂����̂ŕϊ����Ă���
	// ex) [0 0 1] -> [2]
	vector<int> class_data_alt;
	for (int d = 0; d < class_data.size(); ++d) {
		for (int cls = 0; cls < class_num; ++cls) {
			if (class_data[d][cls] == 1) {
				class_data_alt.push_back(cls);
				break;
			}
		}
	}
	uln.calc_params(learn_data,class_data_alt);
	// �p�����[�^���g���Č��؃f�[�^�ƂƂ���beta���o���D
	uln.learn_beta(verification_data,verification_class_data);
	// �w�K�f�[�^���g���Ď��ʂ������D

	/* �]���֐� -T^n*log(probability);�����N���X�̊m���̘a�̕��ρ@DEPRECATE:�]���֐����g���Ď��ʐ��x���o���Ă��d���Ȃ��D
	double j_x = 0;
	for (int d = 0; d < test_class_data.size(); ++d) {
		vector<double> prob;
		uln.calc_probability(test_data[d], prob);
		for (int cls = 0; cls < test_class_data[0].size(); ++cls) {
			j_x += test_class_data[d][cls] * prob[cls];
		}
	}
	cout << "J(x)= " << j_x/test_data.size() << endl;
	*/

	// �����̃N���X/�S�Ẵf�[�^�Ŏ��ʗ����݂�
	int positive = 0;
	for (int d = 0; d < test_class_data.size(); ++d) {
		vector<double> prob;
		uln.calc_probability(test_data[d], prob);
		int max_class_index = distance(test_class_data[d].begin(), max_element(test_class_data[d].begin(), test_class_data[d].end()));
		int max_prob_index = distance(prob.begin(), max_element(prob.begin(), prob.end()));
		if ( max_class_index == max_prob_index ) {
			positive++;
		}
	}
	// TODO:�e�N���X���Ƃ�accuracy, recall, precise���o���D
	cout << "accuracy = " << static_cast<double>(positive) / test_data.size() << endl; 
	// ���ؗp�ɎZ�o�������ϒl�ƕ��U���t�@�C���ɏo�͂���D
	uln.out_file_mean_covar_params("test_007");

	// GIVE UP:�e�X�g�f�[�^����邱�Ƃ��o���Ȃ������̂ŁCtest()������̂͂�����߂�
	// test()
	//Unlearn uln(3, 2, 0.01, 0.01, 0.5, 0.99, 0.01, 10);
	// uln.k_means_test();
	// uln.gauss_test();
	//uln.hgauss_test();
	//uln.get_mean_covar_test();
	return 0;
}
// csv�t�@�C���̓��e��2����vector<double>�ɕϊ�����
vector<vector<double>> get_vector_from_file(const string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		throw "Can't open " + filename;
	}
	string str, str1;
	vector<vector<double>> data;
	while (getline(ifs, str)) {
		stringstream ss{ str };
		vector<double> tmp;
		while (getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max) {
	if (rand_min > rand_max) std::swap(rand_min, rand_max);
	const size_t max_min_diff = static_cast<size_t>(rand_max - rand_min + 1);
	if (max_min_diff < size) throw std::runtime_error("�������ُ�ł�");

	std::vector<int> tmp;
	std::random_device rnd;     // �񌈒�I�ȗ���������ŃV�[�h�����@�𐶐�
	std::mt19937 engine(rnd());     //  �����Z���k�c�C�X�^�[��32�r�b�g�ŁA�����͏����V�[�h
	std::uniform_int_distribution<int> distribution(rand_min, rand_max);

	const size_t make_size = static_cast<size_t>(size * 1.2);

	while (tmp.size() < size) {
		while (tmp.size() < make_size) tmp.push_back(distribution(engine));
		std::sort(tmp.begin(), tmp.end());
		auto unique_end = std::unique(tmp.begin(), tmp.end());

		if (size < std::distance(tmp.begin(), unique_end)) {
			unique_end = std::next(tmp.begin(), size);
		}
		tmp.erase(unique_end, tmp.end());
	}

	return std::move(tmp);
}