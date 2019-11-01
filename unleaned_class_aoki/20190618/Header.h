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
	vector<vector<double>> alpha;
	// 未学習クラスに対する混合度alpha
	const double zeta;
	// 未学習クラスに対する謎の正規化項？
	const double psi;
	vector<vector<double>> input_data;
	// 分散に対する補正項
	double beta;
	// betaを決定するときの増加量
	const double delta_beta;
	const double beta_threshold;
	// データに対するクラス teacher[data]
	vector<int> teacher_data;
	// データに対する所属コンポーネント component_data[data]
	vector<int> component_data;
	// 各クラスに対する算出確率
	vector<double> class_probability;
	// 余事象確率のなんか
	const double epsilon;
	// デバック用
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
	// 入力から確率を計算する．input_dataは学習データ長, class_probabilityはclass_num+1の長さでindex0が未学習クラス．
	void calc_probability(vector<double>& input_data, vector<double>& rtn_cls_probability);
	void calc_probability_learn(vector<double>& input_data, vector<double>& rtn_cls_probability);
	void test();
	void open_csv(const string filename);
	// 記録用　平均と分散とそのた算出に必要なパラメータを入れた
	void out_file_mean_covar_params(const string file_prefix) const;
	
};
inline void Unlearn::set_beta(double beta) {
	this->beta = beta;
}
#endif HEADER_H

