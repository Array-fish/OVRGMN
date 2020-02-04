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
	// 識別するクラス数
	const int class_num;
	// 一クラス事のコンポーネント数
	const int component_num;
	// data_size(demension?) of one data. will use only creating instance from file.
	int data_size;
	// 平均値　mean[class][component]
	vector<vector<VectorXd>> mean;
	// 共分散行列 covar[class][component]
	vector<vector<MatrixXd>> covar;
	// 混合度 alpha[class][component]
	vector<vector<double>> mix_deg;
	// 分散に対する補正項の初期値
	const double init_beta;
	// 分散に対するクラごとの補正項
	vector<double> class_beta;
	// 未学習クラスに対する混合度alpha
	const double unlearn_mix_deg;
	// 未学習クラスに対する謎の正規化項？
	const double normalize_unlearn;
	// 入力データを入れる2次元配列
	vector<vector<double>> input_data;
	// betaの閾値
	const double beta_threshold;
	// betaを決定するときの増加量
	const double delta_beta;
	
	// データに対するクラス teacher[data]
	vector<int> teacher_data;
	// データに対する所属コンポーネント component_data[data]
	vector<int> component_data;
	// 各クラスに対する算出確率
	vector<double> class_probability;
	// 余事象確率の分散を決めるもの
	const double complementary_covar_coef;

	// 近似フラグ
	bool is_approximate;
	// デバック用, Flor use ofstream through multiple methods.
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
	 * ガウス確率の算出
	 * 
	 * @param input 入力1データ分
	 * @param mean 平均EigenのVecterXd型
	 * @param covar 分散行列EigenのMatrixXd型
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
	// NOTE:入力から事後確率を計算する．input_dataは学習データ長, class_probabilityはclass_num+1の長さでindex0が未学習クラス．
	void calc_probability(vector<double>& input_data, vector<double>& rtn_cls_probability);
	// NOTE:各種精度を算出する．
	vector<vector<double>>& evaluate(vector<vector<double>> &test_data, const vector<vector<int>> &class_data, const bool output2cs = false);
	// NOTE:記録用　平均と分散とそのた算出に必要なパラメータを入れた
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

