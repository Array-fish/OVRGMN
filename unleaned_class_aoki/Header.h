#pragma once
#ifndef HEADER_H
#define HEADER_H
#include<vector>
#include<Eigen/Dense>
#include<fstream>
#include<utility>
using namespace std;
using namespace Eigen;
class Unlearn {
private:
	// ���ʂ���N���X��
	const int class_num;
	// ��N���X���̃R���|�[�l���g��
	const int component_num;
	// data_size(demension?) of one data. will use only creating instance from file.
	int data_size;
	// ���ϒl�@mean[class][component]
	vector<vector<VectorXd>> mean;
	// �����U�s�� covar[class][component]
	vector<vector<MatrixXd>> covar;
	// �����x alpha[class][component]
	vector<vector<double>> mix_deg;
	// ���U�ɑ΂���␳���̏����l
	const double init_beta;
	// ���U�ɑ΂���N�����Ƃ̕␳��
	vector<double> class_beta;
	// ���w�K�N���X�ɑ΂��鍬���xalpha
	const double unlearn_mix_deg;
	// ���w�K�N���X�ɑ΂����̐��K�����H
	const double normalize_unlearn;
	// ���̓f�[�^������2�����z��
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

	// �ߎ��t���O
	bool is_approximate;
	// �f�o�b�N�p, Flor use ofstream through multiple methods.
	ofstream class_ofs;
	// for logging release
	ofstream log_ofs;
	// For file output, time prefix
	const string time_prefix;
	//
	// const string output_dir;
	// set data size to create instance from file.
	void set_data_size(int data_size);
	/**
	 * �K�E�X�m���̎Z�o
	 * 
	 * @param input ����1�f�[�^��
	 * @param mean ����Eigen��VecterXd�^
	 * @param covar ���U�s��Eigen��MatrixXd�^
	 */
	double gauss(vector<double>& input, const VectorXd& mean,MatrixXd& covar);
	void k_means(const vector<vector<double>>& input, vector<int>& class_label, const int class_num, const int max_times);
	double hgauss(vector<double>& input, VectorXd& mean, MatrixXd& covar);
	void get_mean_covar(vector<vector<double>>& input, VectorXd& mean, MatrixXd& covar);
	// void test();GIVE UP
	void open_class_csv(const string filename);
	// NOTE:approximate function
	double approximate_exp(double val);
	double approximate_sqrt(double val);
	void print_mat(const MatrixXd& mat);
public:
	Unlearn(
		const int    class_num, 
		const int    component_num, 
		const int    data_size,
		const double init_beta,
		const double unlearn_mixing_degree, 
		const double normalize_unlearn, 
		const double beta_threshold, 
		const double delta_beta, 
		const double complementary_covar_coef,
		const string output_dir
	);
	Unlearn(
		const int            class_num, 
		const int            component_num, 
		const int            data_size, 
		const vector<double> class_beta,
		const double         unlearn_mixing_degree, 
		const double         normalize_unlearn,
		const double         beta_threshold, 
		const double         delta_beta, 
		const double         complementary_covar_coef,
		const string         output_dir
	);
	virtual ~Unlearn();
	// static Unlearn& newinstance_from_file(const string file_directory);GIVE UP
	int k_means_test();
	int gauss_test();
	int hgauss_test();
	// int get_mean_covar_test();
	void calc_params(const vector<vector<double>>& input_data, vector<vector<int>>& class_data);
	void learn_beta(vector<vector<double>>& verification_data, vector<vector<int>>& verification_class_data);
	// NOTE:���͂��玖��m�����v�Z����Dinput_data�͊w�K�f�[�^��, class_probability��class_num+1�̒�����index0�����w�K�N���X�D
	void calc_probability(vector<double>& input_data, vector<double>& rtn_cls_probability);
	// NOTE:�e�퐸�x���Z�o����D
	vector<vector<double>>& evaluate(vector<vector<double>> &test_data, const vector<vector<int>> &class_data, const bool output2cs = false);
	// NOTE:�L�^�p�@���ςƕ��U�Ƃ��̂��Z�o�ɕK�v�ȃp�����[�^����ꂽ
	void out_file_mean() const;
	void out_file_covar() const;
	void out_file_params() const;
	void out_file_mix_deg() const;
	void out_file_data() const;
	void load_file_mean_covar_mixdeg(const string file_directory); 
	void set_approximate(bool is_approximate);
	string get_time_prefix() const;
	void set_output_dir(const string directory_path);
};
inline void Unlearn::set_data_size(int data_size) {
	this->data_size = data_size;
}
inline void Unlearn::set_approximate(bool is_approximate) {
	this->is_approximate = is_approximate;
}
inline string Unlearn::get_time_prefix() const {
	return this->time_prefix;
}
#endif HEADER_H

