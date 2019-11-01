#pragma once
#ifndef HEADER_H
#define HEADER_H
#include<vector>
#include<Eigen/Dense>
#include<fstream>
using namespace std;
using namespace Eigen;
class Unlearn {
private:
	const int class_num;
	const int component_num;
	// ���ϒl�@mean[class][component][data]
	vector<vector<vector<double>>> mean;
	// �����U�s�� covar[class][component][data][data]
	vector<vector<vector<vector<double>>>> covar;
	// �����x alpha[class][component]
	vector<vector<double>> mix_deg;
	// ���U�ɑ΂���␳��
	double beta;
	// ���w�K�N���X�ɑ΂��鍬���xalpha
	const double unlearn_mix_deg;
	// ���w�K�N���X�ɑ΂����̐��K�����H
	const double normalize_unlearn;
	vector<vector<double>> input_data;
	// beta��臒l
	const double beta_threshold;
	// beta�����肷��Ƃ��̑�����
	const double delta_beta;
	
	// �f�[�^�ɑ΂���N���X teacher[data]
	vector<int> teacher_data;
	// �f�[�^�ɑ΂��鏊���R���|�[�l���g component_data[data]
	vector<int> component_data;
	// �e�N���X�ɑ΂���Z�o�m��
	vector<double> class_probability;
	// �]���ۊm���̕��U�����߂����
	const double complementary_covar_coef;
	// �f�o�b�N�p
	ofstream ofs;

	double gauss(vector<double>& input, const VectorXd& mean, const MatrixXd& covar);
	void k_means(const vector<vector<double>>& input, vector<int>& class_label, const int class_num, const int max_times);
	double hgauss(vector<double>& input, VectorXd& mean, MatrixXd& covar);
	void get_mean_covar(vector<vector<double>>& input, vector<double>& mean, vector<vector<double>>& covar);
	void set_beta(double beta);
	// �w�K���Ɏg��
	void calc_probability_learn(vector<double>& input_data, vector<double>& rtn_cls_probability);
	void test();
	void open_csv(const string filename);
public:
	Unlearn(int class_num, int component_num, double beta, double unlearn_mixing_degree, double normalize_unlearn, double beta_threshold, double delta_beta, double complementary_covar_coef);
	int k_means_test();
	int gauss_test();
	int hgauss_test();
	int get_mean_covar_test();
	void calc_params(const vector<vector<double>>& input_data, vector<int>& class_data);
	void learn_beta(vector<vector<double>>& verification_data, vector<vector<double>>& verification_class_data);
	// ���͂���m�����v�Z����Dinput_data�͊w�K�f�[�^��, class_probability��class_num+1�̒�����index0�����w�K�N���X�D
	void calc_probability(vector<double>& input_data, vector<double>& rtn_cls_probability);
	// �L�^�p�@���ςƕ��U�Ƃ��̂��Z�o�ɕK�v�ȃp�����[�^����ꂽ
	void out_file_mean_covar_params(const string file_prefix) const;
	void load_file_mean_covar_params(const string file_prefix);
};
inline void Unlearn::set_beta(double beta) {
	this->beta = beta;
}
#endif HEADER_H

