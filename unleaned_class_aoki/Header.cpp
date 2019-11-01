#define _USE_MATH_DEFINES
#define TO_STRING(VariableName) # VariableName
#include"Header.h"
#include<numeric>
#include<Eigen/Dense>
#include<Eigen/LU>
#include<random>
#include<limits>
#include<cmath>
#include<iostream>
#include<fstream>
//#include<cmath>
using namespace Eigen;
Unlearn::Unlearn(int class_num, int component_num,double beta,double unlearn_mixing_degree,double normalize_unlearn,double beta_threshold,double delta_beta,double complementary_covar_coef)
	:class_num(class_num)
	, component_num(component_num)
	, beta(beta)
	,unlearn_mix_deg(unlearn_mixing_degree)
	, normalize_unlearn(normalize_unlearn)
	,beta_threshold(beta_threshold)
	,delta_beta(delta_beta)
	, complementary_covar_coef(complementary_covar_coef)
{
	mean = vector<vector<vector<double>>>(class_num, vector<vector<double>>(component_num));
	covar = vector<vector<vector<vector<double>>>>(class_num, vector<vector<vector<double>>>(component_num));
	mix_deg = vector<vector<double>>(class_num, vector<double>(component_num));
};
void Unlearn::calc_params(const vector<vector<double>> &input_data,vector<int>& class_data) {
	ofstream ofs;
	ofstream ofs1;
	for (int cls = 0; cls < class_num; ++cls) {
		ofs.open("out_file\\class" + to_string(cls) + "_1.csv");
		// k_mean()に突っ込むためにクラスごとに新しい2次元vectorを作ってる
		vector<vector<double>> cls_input;
		for (int d = 0; d < input_data.size(); ++d) {
			if (class_data[d] == cls) {
				cls_input.push_back(input_data[d]);
			}
		}
		for (auto te_inp : cls_input) {
			for (auto t_i : te_inp) {
				ofs << t_i << ",";
			}
			ofs << endl;
		}
		ofs.close();
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
			ofs1.open("out_file\\class" + to_string(cls) + "component"+to_string(c)+"_1.csv");
			for (auto te_com : tmp_component) {
				for (auto t_c : te_com) {
					ofs1 << t_c << ",";
				}
				ofs1 << endl;
			}
			ofs1.close();
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
	// ここって同じデータが平均になる可能性があるんじゃないかな．
	for (int cls = 0; cls < class_num; ++cls) {
		cls_mean[cls] = input[rand_index(mt)];
	}
	// クラス分けが変わらなくなったら終了
	int change = 1;// クラス分けが変わったかのフラグ
	int loop = 0;// 繰り返し回数
	while (change && loop < max_loop) {
		change = 0;
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
				change = 1;
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
	double kappa = pow(2 * M_PI, -input_size / 2.0) * pow(sigma.determinant(), -1 / 2.0);
	MatrixXd arg_of_exp = -1 * (dif.transpose() * sigma.inverse() * dif) / 2;
	double rtn = kappa * exp(arg_of_exp(0, 0));
	return rtn;
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
	ofstream ofs("out_file\\gauss_test_001.csv");
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
	double kappa = pow(2 * M_PI, -static_cast<double>(input_size) / 2) * pow(covar.determinant(), -static_cast<double>(1) / 2);
	MatrixXd widened_covar = complementary_covar_coef * covar;
	MatrixXd arg_of_exp_former = -1 * (dif.transpose() * widened_covar.inverse() * dif) / 2;
	MatrixXd arg_of_exp_latter = -1 * (dif.transpose() * covar.inverse() * dif) / 2;
	double tmp = kappa / (pow(complementary_covar_coef, static_cast<double>(input_size) / 2) - 1) * (exp(arg_of_exp_former(0,0)) - exp(arg_of_exp_latter(0,0)));
	return tmp;
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
	ofstream ofs("out_file\\hgauss_test_003.csv");
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
	int input_data_size = input[0].size();
	mean.resize(input_data_size);
	// 平均
	for (int i = 0; i < input_data_size; ++i) {
		mean[i] = accumulate(input.begin(), input.end(), 0.0, [&i](double acc, vector<double> & vec) {return acc + vec[i]; }) / input.size();
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
			double var = accumulate(deviation_data.begin(), deviation_data.end(), 0.0, [&row, &cols](double acc, vector<double> & vec) {return acc + vec[row] * vec[cols]; }) / deviation_data.size();
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
	open_csv("out_file\\beta_recode006.csv");
	double energy = 0;
	beta -= delta_beta;// 最初に足しちゃうから先にひいてるよ
	int times = 0;
	while (energy <= beta_threshold && (times++) < 150) {
		energy = 0;// エネルギー関数初期化
		beta += delta_beta;// βをちょっと増やす
		cout << "times:" << times << " beta:" << beta << endl;// デバック用
		for (int d = 0; d < verification_data.size(); ++d) {
			vector<double> prob;
			calc_probability_learn(verification_data[d], prob);
			for (int cls = 0; cls < class_num; ++cls) {
				// 論文ではここはfloor()がついてたけどそんなことしたら全部energyが0になるに決まってんじゃん
				energy += verification_class_data[d][cls] * prob[cls];
			}
			//cout << "times:" << times << " d:" << d <<" " <<TO_STRING(energy) << energy << endl;
		}
		energy = energy / verification_data.size();
		cout << "times:" << times<< " " << TO_STRING(energy) << energy << endl;
		string rec = to_string(times) +","+ to_string(beta)+"," + to_string(energy);
		ofs << rec << endl;

	}
}
void Unlearn::calc_probability(vector<double> &input_data, vector<double> &rtn_cls_probability) {
	vector<double> class_probability(class_num,0);
	double sum_all_probability=0;
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
			MatrixXd beta_sigma = beta * sigma_eigen;
			class_probability[cls] += mix_deg[cls][com] * gauss(input_data, mean_eigen, beta_sigma);
			// 未学習クラスの確率算出
			unlearn_probability *= hgauss(input_data, mean_eigen, sigma_eigen);
		}
		sum_all_probability += class_probability[cls];
	}
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
			MatrixXd beta_sigma = beta * sigma_eigen;
			/*cout << mean_eigen << endl;
			cout << beta_sigma << endl;*/
			double tmp_gauss = gauss(input_data, mean_eigen, beta_sigma);
			/*if (tmp_gauss == 0) {
				getchar();
			}*/
			class_probability[cls] += mix_deg[cls][com] * tmp_gauss;
		}
		sum_all_probability += class_probability[cls];
		if (sum_all_probability == 0) {
			// ここに入るってことは3クラスの算出確率がすべて0になってるってことなんだけど，そんなんある？
			sum_all_probability = 0.00000001;// コレやっていいのかわからん
		}
	}
	for (int cls = 0; cls < class_num; ++cls) {
		class_probability[cls] /= sum_all_probability;
	}
	rtn_cls_probability = class_probability;
}
void Unlearn::test() {
	//VectorXd mu = { 1 };
}
void Unlearn::open_csv(const string filename) {
	ofs.open(filename);	if (ofs.fail()) {throw "can't opne " + filename;}
}

// 確率分布を出すのに必要な各種の値をファイルに保存する．
void Unlearn::out_file_mean_covar_params(const string file_prefix) const{
	// 平均の保存
	string filename = "out_file\\"+file_prefix + "_mean.csv";
	ofstream ofs(filename); if (ofs.fail()) {throw "Can't open " + filename;}
	ofs << "class_num, component_num, data_size" <<endl;
	int data_size = mean[0][0].size();
	ofs << class_num << "," << component_num <<","<<data_size<< endl;
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
	filename ="out_file\\"+ file_prefix + "_covar.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "class_num, component_num, data_size" << endl;
	ofs << class_num << "," << component_num << "," << data_size << endl;
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
	filename = "out_file\\" + file_prefix + "_params.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "beta, unlearn_mixing_degree, normalize_unlearn, complementary_covar_coef" << endl;
	ofs << beta << "," << unlearn_mix_deg << "," << normalize_unlearn << "," << complementary_covar_coef << endl;
	ofs.close();
	// 混合度の値．
	filename = "out_file\\" + file_prefix + "_mix_deg.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "class_num, component_num, data_size" << endl;
	ofs << class_num << "," << component_num << "," << data_size << endl;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			ofs << mix_deg[cls][com] << "," << endl;
		}
	}
	ofs.close();
}
void Unlearn::load_file_mean_covar_params(const string file_prefix) {
	// 平均の入力
	string filename = "in_file\\" + file_prefix + "_mean.csv";
	ifstream ifs(filename); if (ifs.fail()) { throw "Can't open " + filename; }
	ofs << "class_num, component_num, data_size" << endl;
	int data_size = mean[0][0].size();
	ofs << class_num << "," << component_num << "," << data_size << endl;
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
	filename = "out_file\\" + file_prefix + "_covar.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "class_num, component_num, data_size" << endl;
	ofs << class_num << "," << component_num << "," << data_size << endl;
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
	filename = "out_file\\" + file_prefix + "_params.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "beta, unlearn_mixing_degree, normalize_unlearn, complementary_covar_coef" << endl;
	ofs << beta << "," << unlearn_mix_deg << "," << normalize_unlearn << "," << complementary_covar_coef << endl;
	ofs.close();
	// 混合度の値．
	filename = "out_file\\" + file_prefix + "_mix_deg.csv";
	ofs.open(filename); if (ofs.fail()) { throw "can't opne " + filename; }
	ofs << "class_num, component_num, data_size" << endl;
	ofs << class_num << "," << component_num << "," << data_size << endl;
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			ofs << mix_deg[cls][com] << "," << endl;
		}
	}
	ofs.close();
}