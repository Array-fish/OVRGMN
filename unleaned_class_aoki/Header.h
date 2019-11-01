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
	// 平均値　mean[class][component][data]
	vector<vector<vector<double>>> mean;
	// 共分散行列 covar[class][component][data][data]
	vector<vector<vector<vector<double>>>> covar;
	// 混合度 alpha[class][component]
	vector<vector<double>> mix_deg;
	// 分散に対する補正項
	double beta;
	// 未学習クラスに対する混合度alpha
	const double unlearn_mix_deg;
	// 未学習クラスに対する謎の正規化項？
	const double normalize_unlearn;
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
	// デバック用
	ofstream ofs;

	double gauss(vector<double>& input, const VectorXd& mean, const MatrixXd& covar);
	void k_means(const vector<vector<double>>& input, vector<int>& class_label, const int class_num, const int max_times);
	double hgauss(vector<double>& input, VectorXd& mean, MatrixXd& covar);
	void get_mean_covar(vector<vector<double>>& input, vector<double>& mean, vector<vector<double>>& covar);
	void set_beta(double beta);
	// 学習時に使う
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
	// 入力から確率を計算する．input_dataは学習データ長, class_probabilityはclass_num+1の長さでindex0が未学習クラス．
	void calc_probability(vector<double>& input_data, vector<double>& rtn_cls_probability);
	// 記録用　平均と分散とそのた算出に必要なパラメータを入れた
	void out_file_mean_covar_params(const string file_prefix) const;
	void load_file_mean_covar_params(const string file_prefix);
};
inline void Unlearn::set_beta(double beta) {
	this->beta = beta;
}
#endif HEADER_H

