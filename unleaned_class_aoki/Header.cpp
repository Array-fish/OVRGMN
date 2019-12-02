#define _USE_MATH_DEFINES
#define TO_STRING(VariableName) # VariableName
#include"Header.h"
#include"utils.h"
#include<numeric>
#include<Eigen/Dense>
#include<Eigen/LU>
#include<random>
#include<limits>
#include<cmath>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<direct.h>
//#include<cmath>
using namespace Eigen;
Unlearn::Unlearn(const int class_num, const int component_num, const double init_beta, 
	const double unlearn_mixing_degree, const double normalize_unlearn, const double beta_threshold,
	const double delta_beta, const double complementary_covar_coef)
	:class_num(class_num)
	, component_num(component_num)
	, init_beta(init_beta)
	, unlearn_mix_deg(unlearn_mixing_degree)
	, normalize_unlearn(normalize_unlearn)
	, beta_threshold(beta_threshold)
	, delta_beta(delta_beta)
	, complementary_covar_coef(complementary_covar_coef)
	, is_approximate(false)
	, time_prefix("..\\..\\unlearned_files\\" + get_date_sec())
{
	mean = vector<vector<vector<double>>>(class_num, vector<vector<double>>(component_num));
	covar = vector<vector<vector<vector<double>>>>(class_num, vector<vector<vector<double>>>(component_num));
	mix_deg = vector<vector<double>>(class_num, vector<double>(component_num));
	class_beta = vector<double>(class_num, init_beta);
	if (_mkdir(time_prefix.c_str())) {
		cerr << time_prefix << " cannot be created." << endl;
		exit(-1);
	}
}
// create form file version.
Unlearn::Unlearn(const int class_num, const int component_num, const int data_size, const vector<double> class_beta,
	const double unlearn_mixing_degree, const double normalize_unlearn, const double beta_threshold,
	const double delta_beta, const double complementary_covar_coef)
	:class_num(class_num)
	, component_num(component_num)
	, init_beta(0)
	, data_size(data_size)
	, class_beta(class_beta)
	, unlearn_mix_deg(unlearn_mixing_degree)
	, normalize_unlearn(normalize_unlearn)
	, beta_threshold(beta_threshold)
	, delta_beta(delta_beta)
	, complementary_covar_coef(complementary_covar_coef)
	, is_approximate(false)
	, time_prefix("..\\..\\unlearned_files\\" + get_date_sec())
{
	mean = vector<vector<vector<double>>>(class_num, vector<vector<double>>(component_num,vector<double>(data_size)));
	covar = vector<vector<vector<vector<double>>>>(class_num, vector<vector<vector<double>>>(component_num,vector<vector<double>>(data_size,vector<double>(data_size))));
	mix_deg = vector<vector<double>>(class_num, vector<double>(component_num));
	if (_mkdir(time_prefix.c_str())) {
		cerr << time_prefix << " cannot be created." << endl;
		exit(-1);
	}
}
//Unlearn& Unlearn::newinstance_from_file(const string file_directory) {
//	// prepare variables
//	string filename;
//	ifstream ifs;
//	string line;
//	// load parameters and beta, then create new instance.
//	filename = file_directory+"\\params.csv";
//	ifs.open(filename); if (ifs.fail()) { throw "can't opne " + filename; }
//	getline(ifs, line); getline(ifs, line); // NOTE: Pass headers.
//	vector<string> load_params = split(line, ',');
//	const int class_num = stoi(load_params[0]);
//	const int component_num = stoi(load_params[1]);
//	const int data_size = stoi(load_params[2]);
//	const double unlearn_mix_deg = stod(load_params[3]);
//	const double normalize_unlearn = stod(load_params[4]);
//	const double beta_threshold = stod(load_params[5]);
//	const double delta_beta = stod(load_params[6]);
//	const double ccomplementary_covar_coef = stod(load_params[7]);
//	getline(ifs, line); getline(ifs, line);
//	load_params = split(line, ',');
//	const double beta = stod(load_params[0]);
//	ifs.close();
//
//	return Unlearn(class_num, component_num, data_size, beta, unlearn_mix_deg, normalize_unlearn,
//		beta_threshold, delta_beta, ccomplementary_covar_coef);
//}
void Unlearn::load_file_mean_covar_mixdeg(const string file_directory) {
	string filename;
	ifstream ifs;
	string line;
	// load mean
	filename = file_directory+"\\mean.csv";
	ifs.open(filename); if (ifs.fail()) { throw "can't opne " + filename; }
	vector<string> load_means;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			getline(ifs, line);
			load_means = split(line, ',');
			for (int ds = 0; ds < data_size; ++ds) {
				mean[cls][com][ds] = stod(load_means[ds]);
			}
		}
	}
	ifs.close();
	// 共分散の入力
	filename = file_directory+"\\covar.csv";
	ifs.open(filename); if (ifs.fail()) { throw "can't opne " + filename; }
	vector<string> load_covars;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			for (int dsr = 0; dsr < data_size; ++dsr) {
				getline(ifs, line);
				load_covars = split(line, ',');
				for (int dsc = 0; dsc < data_size; ++dsc) {
					covar[cls][com][dsr][dsc] = stod(load_covars[dsc]);
				}
			}
		}
	}
	ifs.close();

	// 混合度の値．
	filename = file_directory+"\\mix_deg.csv";
	ifs.open(filename); if (ifs.fail()) { throw "can't opne " + filename; }
	vector<string> load_mix_deg;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			getline(ifs, line);
			load_mix_deg = split(line, ',');
			mix_deg[cls][com] = stod(load_mix_deg[0]);
		}
	}
	ifs.close();
}
void Unlearn::calc_params(const vector<vector<double>> &input_data,vector<int>& class_data) {
	for (int cls = 0; cls < class_num; ++cls) {
		// k_mean()に突っ込むためにクラスごとに新しい2次元vectorを作ってる
		vector<vector<double>> cls_input;
		for (int d = 0; d < input_data.size(); ++d) {
			if (class_data[d] == cls) {
				cls_input.push_back(input_data[d]);
			}
		}
		vector<int> component_label;
		k_means(cls_input, component_label, component_num, 1000);
		for (int c = 0; c < component_num; ++c) {
			// make_mean_covar()に突っ込むためにコンポーネントごとに2次元vectorを作ってる
			vector<vector<double>> tmp_component;
			for (int d = 0; d < cls_input.size(); ++d) {
				if (component_label[d] == c) {
					tmp_component.push_back(cls_input[d]);
				}
			}
			// クラス　コンポーネントごとのデータを使って平均と分散を作る
			get_mean_covar(tmp_component, mean[cls][c], covar[cls][c]);
			// ついでに混同度alphaを設定する
			mix_deg[cls][c] = (1 - unlearn_mix_deg) * tmp_component.size() / input_data.size();
		}
	}
}
void Unlearn::k_means(const vector<vector<double>> &input,vector<int> &class_label,const int class_num,const int max_loop) {
	// 平均の初期値としてランダムな二つのデータを割り当てている．
	class_label.resize(input.size());
	std::random_device rnd;     // 非決定的な乱数生成器でシード生成機を生成
	std::mt19937 mt(rnd());     //  メルセンヌツイスターの32ビット版、引数は初期シード
	std::uniform_int_distribution<> rand_index(0, input.size()-1);     // [0, ?] 範囲の一様乱数
	vector<vector<double>> cls_mean = vector<vector<double>>(class_num, vector<double>(input[0].size(), 0));
	for (int cls = 0; cls < class_num; ++cls) {
		cls_mean[cls] = input[rand_index(mt)];
		// 同じデータが2回現れていないかチェック
		for (int exist_cls = 0; exist_cls < cls; ++exist_cls) {
			if (cls_mean[cls] == cls_mean[exist_cls]) {
				cls--;
				break;
			}
		}
	}
	// クラス分けが変わらなくなったら終了
	bool change = true;// クラス分けが変わったかのフラグ
	int loop = 0;// 繰り返し回数
	while (change && loop < max_loop) {
		change = false;
		for (int in = 0; in < input.size(); ++in) {
			double min_dist = numeric_limits<double>::max();
			int min_cls = 0;
			// 全てのクラスについて2乗誤差を比較する
			for (int cls = 0; cls < class_num; ++cls) {
				double cls_dist = 0;
				for (int d = 0; d < input[0].size(); ++d) {
					cls_dist+=pow(input[in][d] - cls_mean[cls][d], 2);
				}
				if (cls_dist < min_dist) {
					min_dist = cls_dist;
					min_cls = cls;
				}
			}
			// クラス平均との距離が最小になるクラスが自分の所属していたクラスと異なるとき，所属クラスを変更する．
			// 以上の処理によってあるデータの所属クラスが変化した場合，クラスの平均が変化するので処理を続ける必要がある．
			if (min_cls != class_label[in]) {
				class_label[in] = min_cls;
				change = true;
			}
		}
		// クラスの平均値の更新
		if (change) {
			vector<vector<double>> cls_sum(class_num, vector<double>(input[0].size(), 0));
			vector<int> cls_data_cnt(class_num,0);// 後で平均を取る用，そのクラスに割り振られたデータの数
			for (int in = 0; in < input.size(); ++in) {
				int cls_label = class_label[in];
				cls_data_cnt[cls_label]++;
				for (int d = 0; d < input[0].size(); ++d) {
					cls_sum[cls_label][d] += input[in][d];
				}
			}
			for (int cls = 0; cls < class_num; ++cls) {
				for (int d = 0; d < input[0].size(); ++d) {
					cls_mean[cls][d] = cls_sum[cls][d] / cls_data_cnt[cls];
				}
			}
		}
		loop++;
	}
}
int Unlearn::k_means_test() {
	vector<vector<double>> input = vector<vector<double>>(200, vector<double>(2,0));
	std::random_device seed_gen;     // 非決定的な乱数生成器でシード生成機を生成
	std::mt19937 mt(seed_gen());     //  メルセンヌツイスターの32ビット版、引数は初期シード
	std::normal_distribution<> ndist(5.0,1.0);     // 平均5，標準偏差1.0のガウス分布
	for (int da = 0; da < 100; ++da) {
		input[da][0] = ndist(mt);
		input[da][1] = ndist(mt);
	}
	std::normal_distribution<> ndist1(-5.0, 1.0);     // 平均5，標準偏差1.0のガウス分布
	for (int da = 100; da < 200; ++da) {
		input[da][0] = ndist1(mt);
		input[da][1] = ndist1(mt);
	}
	vector<int> class_label;
	int class_num = 2;
	int max_times = 1000;
	k_means(input, class_label, class_num, max_times);
	cout << accumulate(class_label.begin(), class_label.begin() + 100, 0) << endl;
	cout << accumulate(class_label.begin() + 100, class_label.begin() + 200, 0) << endl;;
	return 0;
}
// ちなみに行列のサイズが大きくなるとdeterminabt()とinverse()が死ぬらしいけど，よくわからん．
double Unlearn::gauss(vector<double>& input, const VectorXd& mean,const MatrixXd& sigma) {
	int input_size = input.size();
	VectorXd inp = Map<VectorXd>(&input[0],input_size);
	VectorXd dif = inp - mean;
	double kappa; // TODO:change the variable name to understandable name.
	if(is_approximate)
		kappa = pow(2 * M_PI, -input_size / 2.0) / approximate_sqrt(sigma.determinant());
	else
		kappa = pow(2 * M_PI, -input_size / 2.0) * pow(sigma.determinant(), -1 / 2.0);
	MatrixXd arg_of_exp = -1 * (dif.transpose() * sigma.inverse() * dif) / 2;
	if (is_approximate) 
		return  kappa * approximate_exp(arg_of_exp(0, 0));
	else 
		return kappa * exp(arg_of_exp(0, 0));
}
int Unlearn::gauss_test() {
	// －1から１まで縦横をそれぞれ100分割した．これをガウス関数に入れて分布が正しいか見る．
	const int grid_div = 100;
	vector<vector<double>> input_list(pow(grid_div+1, 2), vector<double>(2, 0));
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			input_list[x * (grid_div + 1) + y][0] = static_cast<double>(2 * x - grid_div) / grid_div;
			input_list[x * (grid_div + 1) + y][1] = static_cast<double>(2 * y - grid_div) / grid_div;
		}
	}
	VectorXd mean=VectorXd::Zero(2);
	MatrixXd covar = MatrixXd::Zero(2, 2);
	covar(0, 0) = covar(1, 1) = 1;
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			input_list[x * (grid_div + 1) + y][0] = static_cast<double>(2 * x - grid_div) / grid_div;
			input_list[x * (grid_div + 1) + y][1] = static_cast<double>(2 * y - grid_div) / grid_div;
		}
	}
	vector<double> output_list(pow(grid_div + 1, 2), 0);
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			output_list[x * (grid_div + 1) + y] = gauss(input_list[x * (grid_div + 1) + y], mean, covar);
		}
	}
	// 出てきた数字を目視で確認したけど，正直あってるかわからない．
	// ここの部分もっとちゃんとしたってるか調べる方法が欲しい．
	ofstream ofs(time_prefix + "\\gauss_test_001.csv");
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			ofs << output_list[x * (grid_div + 1) + y] <<",";
			// cout << static_cast<int>(output_list[x * (grid_div + 1) + y]*1000) << ", ";
		}
		ofs << endl;
		// cout << endl;
	}
	return 0;
}
double Unlearn::hgauss(vector<double>& input, VectorXd& mean, MatrixXd& covar) {
	int input_size = input.size();
	VectorXd inp = Map<VectorXd>(&input[0], input_size);
	VectorXd dif = inp - mean;
	double kappa; // TODO:change the variable name to understandable name.
	if (is_approximate)
		kappa = pow(2 * M_PI, -input_size / 2.0) / approximate_sqrt(covar.determinant());
	else
		kappa = pow(2 * M_PI, -input_size / 2.0) * pow(covar.determinant(), -1 / 2.0);
	MatrixXd widened_covar = complementary_covar_coef * covar;
	MatrixXd arg_of_exp_former = -1 * (dif.transpose() * widened_covar.inverse() * dif) / 2;
	MatrixXd arg_of_exp_latter = -1 * (dif.transpose() * covar.inverse() * dif) / 2;
	if(is_approximate)
		return  kappa / (pow(complementary_covar_coef, static_cast<double>(input_size) / 2) - 1)
			* (approximate_exp(arg_of_exp_former(0, 0)) - approximate_exp(arg_of_exp_latter(0, 0)));
	else
		return  kappa / (pow(complementary_covar_coef, static_cast<double>(input_size) / 2) - 1) * (exp(arg_of_exp_former(0,0)) - exp(arg_of_exp_latter(0,0))); 
}
int Unlearn::hgauss_test() {
	// －1から１まで縦横をそれぞれ100分割した．これをガウス関数に入れて分布が正しいか見る．
	const int grid_div = 100;
	const int ratio = 3;
	vector<vector<double>> input_list(pow(grid_div + 1, 2), vector<double>(2, 0));
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			input_list[x * (grid_div + 1) + y][0] = static_cast<double>(2 * x - grid_div) * ratio / grid_div;
			input_list[x * (grid_div + 1) + y][1] = static_cast<double>(2 * y - grid_div) * ratio / grid_div;
		}
	}
	VectorXd mean = VectorXd::Zero(2);
	MatrixXd covar = MatrixXd::Zero(2, 2);
	covar(0, 0) = covar(1, 1) = 1;
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			input_list[x * (grid_div + 1) + y][0] = static_cast<double>(2 * x - grid_div) * ratio / grid_div;
			input_list[x * (grid_div + 1) + y][1] = static_cast<double>(2 * y - grid_div) * ratio / grid_div;
		}
	}
	vector<double> output_list(pow(grid_div + 1, 2), 0);
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			output_list[x * (grid_div + 1) + y] = hgauss(input_list[x * (grid_div + 1) + y], mean, covar);
		}
	}
	// 出てきた数字を目視で確認したけど，正直あってるかわからない．
	// ここの部分もっとちゃんとしたってるか調べる方法が欲しい．
	ofstream ofs(time_prefix + "\\hgauss_test_003.csv");
	for (int x = 0; x <= grid_div; ++x) {
		for (int y = 0; y <= grid_div; ++y) {
			ofs << output_list[x * (grid_div + 1) + y] << ",";
			// cout << static_cast<int>(output_list[x * (grid_div + 1) + y]*1000) << ", ";
		}
		ofs << endl;
		// cout << endl;
	}
	return 0;
}
void Unlearn::get_mean_covar(vector<vector<double>>& input, vector<double> &mean, vector<vector<double>> & sigma) {
	const int input_data_size = input[0].size();
	mean.resize(input_data_size);
	// 平均
	for (int i = 0; i < input_data_size; ++i) {
		mean[i] = accumulate(input.begin(), input.end(), 0.0, [&i](double acc, vector<double>& vec) {return acc + vec[i]; }) / input.size();
	}
	vector<vector<double>> deviation_data = input;// 偏差行列
	for (int row = 0; row < deviation_data.size(); ++row) {
		for (int cols = 0; cols < deviation_data[0].size(); ++cols) {
			deviation_data[row][cols] -= mean[cols];
		}
	}
	// 共分散行列 ここ平均の掛け算で行けるはず　時間があったら変えて
	sigma = vector<vector<double>>(input_data_size, vector<double>(input_data_size, 0));
	for (int row = 0; row < deviation_data[0].size(); ++row) {
		for (int cols = row; cols < deviation_data[0].size(); ++cols) {
			double var = accumulate(deviation_data.begin(), deviation_data.end(), 0.0, [&row, &cols](double acc, vector<double>& vec) {return acc + vec[row] * vec[cols]; }) / deviation_data.size();
			sigma[row][cols] = var;
			sigma[cols][row] = var;
		}
	}
}
int Unlearn::get_mean_covar_test() {
	vector<double> mean;
	vector<vector<double>> covar;
	const int data_num = 200;
	vector<vector<double>> input = vector<vector<double>>(data_num, vector<double>(2, 0));
	std::random_device seed_gen;     // 非決定的な乱数生成器でシード生成機を生成
	std::mt19937 mt(seed_gen());     //  メルセンヌツイスターの32ビット版、引数は初期シード
	std::normal_distribution<> ndist(5.0, 2.0);     // 平均5，標準偏差1.0のガウス分布
	std::normal_distribution<> ndist1(4.0, 1.0);     // 平均5，標準偏差1.0のガウス分布
	for (int da = 0; da < data_num; ++da) {
		input[da][0] = ndist(mt);
		input[da][1] = ndist1(mt);
	}
	get_mean_covar(input, mean, covar);
	for (auto me : mean) {
		cout << me << ",";
	}
	cout << endl;
	for (auto cova : covar) {
		for (auto co : cova) {
			cout << co << ",";
		}
		cout << endl;
	}
	return 0;
}
void Unlearn::learn_beta(vector<vector<double>> &verification_data,vector<vector<double>> &verification_class_data) {
	// デバック用
	for (int cls = 0; cls < class_num; ++cls) {
		ofstream ofs(time_prefix + "\\class"+to_string(cls)+"beta.csv");
		check_open_file(ofs, time_prefix + "\\class" + to_string(cls) + "beta.csv");
		double energy = 0;
		class_beta[cls] -= delta_beta;// 最初に足しちゃうから先にひいてるよ
		int times = 0;
		double previous_energy = 0;
		while (energy <= beta_threshold && (times++) < 150) {
			energy = 0;// エネルギー関数初期化
			class_beta[cls] += delta_beta;// βをちょっと増やす
			cout << "class:"<<cls<<", times:" << times << " beta:" << class_beta[cls] << endl;// デバック用
			for (int d = 0; d < verification_data.size(); ++d) {
				vector<double> prob;
				calc_probability_learn(verification_data[d], prob);
				for (int tmp_cls = 0; tmp_cls < class_num; ++tmp_cls) {
					energy += verification_class_data[d][tmp_cls] * prob[tmp_cls];
				}
			}
			energy /= verification_data.size();
			cout << "times:" << times << " " << TO_STRING(energy) << ":" << energy << endl;
			string rec = to_string(times) + "," + to_string(class_beta[cls]) + "," + to_string(energy);
			ofs << rec << endl;
			if (energy <= previous_energy) 
				break;
			previous_energy = energy;
		}
	}
}
void Unlearn::calc_probability(vector<double> &input_data, vector<double> &rtn_cls_probability) {
	//ofstream uln_ofs(time_prefix + "\\unlearn_hgauss_likelihood.csv");
	//ofstream learn_class_ofs(time_prefix + "\\learn_class_likelihood.csv");
	//check_open_file(uln_ofs,time_prefix + "\\unlearn_hgauss_likelihood.csv");
	//check_open_file(learn_class_ofs, time_prefix + "\\learn_class_likelihood.csv");
	vector<double> class_probability(class_num, 0.0);
	double sum_all_probability = 0.0;

	// 未学習クラスのやつ
	double unlearn_probability = unlearn_mix_deg * normalize_unlearn;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			// クラス確率の計算用，平均と分散の変換
			VectorXd mean_eigen = Map<VectorXd>(&mean[cls][com][0], mean[cls][com].size());
			vector<double> tmp;// eigenのMap<>の機能を
			for (int col = 0; col < covar[cls][com].size(); ++col) {
				for (int row = 0; row < covar[cls][com][0].size(); ++row) {
					tmp.push_back(covar[cls][com][row][col]);
				}
			}
			MatrixXd sigma_eigen = Map<MatrixXd>(&tmp[0], covar[cls][com].size(), covar[cls][com][0].size());
			// 学習クラスの確率算出
			MatrixXd beta_sigma = class_beta[cls] * sigma_eigen;
			class_probability[cls] += mix_deg[cls][com] * gauss(input_data, mean_eigen, beta_sigma);
			// 未学習クラスの確率算出
			unlearn_probability *= hgauss(input_data, mean_eigen, sigma_eigen);
			double output_hga = hgauss(input_data, mean_eigen, sigma_eigen);
			//uln_ofs <<  output_hga <<",";
		}
		/*learn_class_ofs << class_probability[cls] << ",";
		if (cls >= class_num-1) {
			learn_class_ofs << endl;
		}*/
		sum_all_probability += class_probability[cls];
	}
	//uln_ofs << unlearn_probability << endl;
	sum_all_probability += unlearn_probability;
	unlearn_probability /= sum_all_probability;
	for (int cls = 0; cls < class_num; ++cls) {
		class_probability[cls] /= sum_all_probability;
	}
	vector<double> tmp;
	tmp.push_back(unlearn_probability);
	for (int cls = 0; cls < class_num; ++cls) {
		tmp.push_back(class_probability[cls]);
	}
	rtn_cls_probability = tmp;
}
void Unlearn::calc_probability_learn(vector<double>& input_data, vector<double>& rtn_cls_probability) {
	vector<double> class_probability(class_num, 0);
	double sum_all_probability = 0;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			// クラス確率の計算用，平均と分散の変換
			VectorXd mean_eigen = Map<VectorXd>(&mean[cls][com][0], mean[cls][com].size());
			vector<double> tmp;// eigenのMap<>の機能を
			for (int col = 0; col < covar[cls][com].size(); ++col) {
				for (int row = 0; row < covar[cls][com][0].size(); ++row) {
					tmp.push_back(covar[cls][com][row][col]);
				}
			}
			MatrixXd sigma_eigen = Map<MatrixXd>(&tmp[0], covar[cls][com].size(), covar[cls][com][0].size());
			// 学習クラスの確率算出
			MatrixXd beta_sigma = class_beta[cls] * sigma_eigen;
			double tmp_gauss = gauss(input_data, mean_eigen, beta_sigma);
			class_probability[cls] += mix_deg[cls][com] * tmp_gauss;
		}
		sum_all_probability += class_probability[cls];
		if (sum_all_probability == 0) {
			// ここに入るってことは3クラスの算出確率がすべて0になってるってことなんだけど，そんなんある？
			sum_all_probability = 0.0000000001;// コレやっていいのかわからん
		}
	}
	for (int cls = 0; cls < class_num; ++cls) {
		class_probability[cls] /= sum_all_probability;
	}
	rtn_cls_probability = class_probability;
}
//void Unlearn::test() {
//	//VectorXd mu = { 1 };
//}
void Unlearn::open_class_csv(const string filename) {
	class_ofs.open(filename);	if (class_ofs.fail()) {throw "can't opne " + filename;}
}
// 確率分布を出すのに必要な各種の値をファイルに保存する．
void Unlearn::out_file_mean_covar_params() const{
	// 平均の保存
	string filename = time_prefix + "\\mean.csv";
	ofstream ofs(filename); if (ofs.fail()) {throw "Can't open " + filename;}
	const int data_size = mean[0][0].size();
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			for (double mea : mean[cls][com]) {
				ofs << mea << ",";
			}
			ofs << endl;
		}
	}
	ofs.close();
	// 共分散の保存
	filename = time_prefix + "\\covar.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			for (int dat = 0; dat < data_size; ++dat) {
				for (double cov : covar[cls][com][dat]) {
					ofs << cov << ",";
				}
				ofs << endl;
			}
		}
	}
	ofs.close();
	// その他の値の保存
	filename = time_prefix + "\\params.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "class_num,component_num,data_size,unlearn_mixing_degree,normalize_unlearn,beta_threshold,delta_beta,complementary_covar_coef" << endl;
	ofs << class_num << "," << component_num << "," << data_size << "," << unlearn_mix_deg << ","
		<< normalize_unlearn << ","<< beta_threshold<< "," << delta_beta << "," 
		<< complementary_covar_coef << endl;
	ofs << "beta " << endl;
	for (int cls = 0; cls < class_num; ++cls) {
		ofs << class_beta[cls] << ",";
	}
	ofs << endl;
	ofs.close();
	// 混合度の値．
	filename = time_prefix + "\\mix_deg.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			ofs << mix_deg[cls][com] << "," << endl;
		}
	}
	ofs.close();
}
/*
 * NOTE:calculate accuracy and presicion, recall, F-measure
*/
vector<vector<double>>& Unlearn::evaluate(vector<vector<double>>& test_data, const vector<vector<double>>& class_data, 
	const bool output2csv){
	vector<int> class_true_positive(class_num + 1, 0), class_true_negative(class_num + 1, 0),
		        class_false_positive(class_num + 1, 0), class_false_negative(class_num + 1, 0);
	for (int d = 0; d < test_data.size(); ++d) {
		vector<double> prob;
		calc_probability(test_data[d], prob);
		int max_class_index = distance(class_data[d].begin(), max_element(class_data[d].begin(), class_data[d].end()));
		int max_prob_index = distance(prob.begin(), max_element(prob.begin(), prob.end()));
		if (max_class_index == max_prob_index) {
			class_true_positive[max_class_index]++;
			for (int cls = 0; cls <= class_num; ++cls){
				if (cls != max_class_index)
					class_true_negative[cls]++;
			}
		}
		else {
			class_false_positive[max_prob_index]++;
			class_false_negative[max_class_index]++;
			for (int cls = 0; cls < class_num + 1; ++cls) {
				if (cls != max_class_index && cls != max_prob_index)
					class_true_negative[cls]++;
			}
		}
	}
	// NOTE:result[class][indeces] Indeces's order is [accuracy, presicion, recall, F-measure]
	vector<vector<double>> result(class_num + 1, vector<double>(4, 0.0));
	for (int cls = 0; cls < class_num + 1; ++cls) {
		result[cls][0] = static_cast<double>(class_true_positive[cls] + class_true_negative[cls])
			/ (class_true_positive[cls] + class_false_positive[cls] + class_false_negative[cls] + class_true_negative[cls]);
		result[cls][1] = static_cast<double>(class_true_positive[cls]) / (class_true_positive[cls] + class_false_positive[cls]);
		result[cls][2] = static_cast<double>(class_true_positive[cls]) / (class_true_positive[cls] + class_false_negative[cls]);
		result[cls][3] = 2 * result[cls][1] * result[cls][2] / (result[cls][1] + result[cls][2]);
		cout << "class: " << setw(3) << cls 
			<<", accuracy: "   << fixed << setprecision(8) << result[cls][0] 
			<< ", presicion: " << fixed << setprecision(8) << result[cls][1]
			<< ", recall: "    << fixed << setprecision(8) << result[cls][2] 
			<< ", F-measure:"  << fixed << setprecision(8) << result[cls][3] << endl;
	}
	vector<double> macro_average(4, 0.0);
	for (int i = 0; i < 4; ++i) {
		for (int cls = 0; cls < class_num + 1; cls++) {
			macro_average[i] += result[cls][i];
		}
		macro_average[i] /= (class_num + 1);
	}
	cout << "macro_average" << endl
		<< "accuracy: " << fixed << setprecision(8) << macro_average[0] << endl
		<< "presicion: " << fixed << setprecision(8) << macro_average[1] << endl
		<< "recall: " << fixed << setprecision(8) << macro_average[2] << endl
		<< "F-measure(ave): " << fixed << setprecision(8) << macro_average[3] << endl
		<< "F-measure(calc): " << fixed << setprecision(8)
		<< 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;
	double micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();
	cout << "micro_average: " << fixed << setprecision(8) << micro_average << endl;
	if (output2csv) {
		ofstream ofs(time_prefix + "\\evaluate.csv");
		ofs << "class,accuracy,presicion,recall,F-measure" << endl;
		for (int cls = 0; cls < class_num + 1; ++cls) {
			ofs << cls << "," << result[cls][0] << "," << result[cls][1] << "," << result[cls][2] << "," << result[cls][3] << endl;
		}
		ofs << "macro_average" << endl
			<< "accuracy," << macro_average[0] << endl
			<< "presicion," << macro_average[1] << endl
			<< "recall," << macro_average[2] << endl
			<< "F-measure(ave)," << macro_average[3] << endl
			<< "F-measure(calc),"
			<< 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;
		double micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();
		ofs << "micro_average," << micro_average << endl;
	}
	return result;
	
}

double Unlearn::approximate_exp(double val) {
	int floor_val = floor(val);
	double decimal_val = val - floor_val;
	if (floor_val >= 0)
		return (decimal_val + 1) * (1 << floor_val);
	else
		return (decimal_val + 1) * pow(2, floor_val);
		// return (decimal_val + 1) * (1 >> -floor_val); // NOTE:これだと0になる
}
double Unlearn::approximate_sqrt(double val) {
	int shift_cnt = 0;
	if (val >= 0.5) {
		while (val > 2) {
			val /= 4;
			shift_cnt++;
		}
		return (val + 1) / 2 * (1 << shift_cnt);
	}
	else {
		while (val < 0.5) {
			val *= 4;
			shift_cnt++;
		}
		return (val + 1) / 2 * pow(2 , -shift_cnt);
		// return (val + 1) / 2 * (1 >> shift_cnt); // NOTE:これだとゼロになる
	}
}