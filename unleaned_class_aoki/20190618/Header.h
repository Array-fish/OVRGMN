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
	vector<vector<double>> alpha;
	// ���w�K�N���X�ɑ΂��鍬���xalpha
	const double zeta;
	// ���w�K�N���X�ɑ΂����̐��K�����H
	const double psi;
	vector<vector<double>> input_data;
	// ���U�ɑ΂���␳��
	double beta;
	// beta�����肷��Ƃ��̑�����
	const double delta_beta;
	const double beta_threshold;
	// �f�[�^�ɑ΂���N���X teacher[data]
	vector<int> teacher_data;
	// �f�[�^�ɑ΂��鏊���R���|�[�l���g component_data[data]
	vector<int> component_data;
	// �e�N���X�ɑ΂���Z�o�m��
	vector<double> class_probability;
	// �]���ۊm���̂Ȃ�
	const double epsilon;
	// �f�o�b�N�p
	ofstream ofs;
public:
	Unlearn(int class_num, int component_num, double beta, double zeta, double psi, double beta_threshold, double delta_beta, double epsilon);
	void k_means(vector<vector<double>>& input, vector<int>& class_label, int class_num, int max_times);
	int k_means_test(vector<vector<double>>& input, vector<int>& class_label, int class_num, int max_times);
	double gauss(vector<double>& input, VectorXd& mean, MatrixXd& sigma);
	int gauss_test(vector<double>& input, VectorXd& mean, MatrixXd& sigma);
	double hgauss(vector<double>& input, VectorXd& mean, MatrixXd& sigma);
	double hgauss_test(vector<double>& input, VectorXd& mean, MatrixXd& sigma);
	void get_mean_covar(vector<vector<double>>& input, vector<double>& mean, vector<vector<double>>& sigma);
	void get_mean_covar_test(vector<vector<double>>& input, vector<double>& mean, vector<vector<double>>& sigma);
	void calc_params(vector<vector<double>>& input_data, vector<double>& class_data);
	void set_beta(double beta);
	void learn_beta(vector<vector<double>>& verification_data, vector<vector<double>>& verification_class_data);
	// ���͂���m�����v�Z����Dinput_data�͊w�K�f�[�^��, class_probability��class_num+1�̒�����index0�����w�K�N���X�D
	void calc_probability(vector<double>& input_data, vector<double>& rtn_cls_probability);
	void calc_probability_learn(vector<double>& input_data, vector<double>& rtn_cls_probability);
	void test();
	void open_csv(const string filename);
	// �L�^�p�@���ςƕ��U�Ƃ��̂��Z�o�ɕK�v�ȃp�����[�^����ꂽ
	void out_file_mean_covar_params(const string file_prefix) const;
	
};
inline void Unlearn::set_beta(double beta) {
	this->beta = beta;
}
#endif HEADER_H

