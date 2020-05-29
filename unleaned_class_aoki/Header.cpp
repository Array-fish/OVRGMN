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
#include<filesystem>
//#include<cmath>
using namespace Eigen;
Unlearn::Unlearn(const int class_num, const int component_num, const int data_size, const double init_beta,
	const double unlearn_mixing_degree, const double normalize_unlearn, const double beta_threshold,
	const double delta_beta, const double complementary_covar_coef,const string output_dir)
	: class_num(class_num)
	, component_num(component_num)
	, data_size(data_size)
	, init_beta(init_beta)
	, unlearn_mix_deg(unlearn_mixing_degree)
	, normalize_unlearn(normalize_unlearn)
	, beta_threshold(beta_threshold)
	, delta_beta(delta_beta)
	, complementary_covar_coef(complementary_covar_coef)
	, is_approximate(false)
	, time_prefix(output_dir + get_date_sec())
{
	mean = vector<vector<VectorXd>>(class_num, vector<VectorXd>(component_num));
	covar = vector<vector<MatrixXd>>(class_num, vector<MatrixXd>(component_num));
	mix_deg = vector<vector<double>>(class_num, vector<double>(component_num));
	class_beta = vector<double>(class_num, init_beta);
	if (!std::filesystem::create_directories(time_prefix)) {
		cerr << time_prefix << " cannot be created." << endl;
		exit(-1);
	}
	log_ofs.open(log_file_path, ios::app);
	if (!log_ofs) {
		cerr << "log file cannot be opened." << endl;
		exit(-1);
	}
	log_ofs << get_date_sec() + " Unlearn instance is created." << endl;
}
// create form file version.
Unlearn::Unlearn(const int class_num, const int component_num, const int data_size, const vector<double> class_beta,
	const double unlearn_mixing_degree, const double normalize_unlearn, const double beta_threshold,
	const double delta_beta, const double complementary_covar_coef, const string output_dir)
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
	, time_prefix(output_dir + get_date_sec())
{
	mean = vector<vector<VectorXd>>(class_num, vector<VectorXd>(component_num));
	covar = vector<vector<MatrixXd>>(class_num, vector<MatrixXd>(component_num));
	mix_deg = vector<vector<double>>(class_num, vector<double>(component_num));
	if (!std::filesystem::create_directories(time_prefix)) {
		cerr << time_prefix << " cannot be created." << endl;
		exit(-1);
	}
	log_ofs.open(log_file_path, ios::app);
	if (!log_ofs) {
		cerr << "log file cannot be opened." << endl;
		exit(-1);
	}
}
Unlearn::~Unlearn()
{
	if (log_ofs) {
		log_ofs.close();
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
//	const int cluster_num = stoi(load_params[0]);
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
//	return Unlearn(cluster_num, component_num, data_size, beta, unlearn_mix_deg, normalize_unlearn,
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
			mean[cls][com] = VectorXd::Zero(data_size);
			getline(ifs, line);
			load_means = split(line, ',');
			for (int ds = 0; ds < data_size; ++ds) {
				mean[cls][com](ds) = stod(load_means[ds]);
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
			covar[cls][com] = MatrixXd::Zero(data_size, data_size);
			for (int dsr = 0; dsr < data_size; ++dsr) {
				getline(ifs, line);
				load_covars = split(line, ',');
				for (int dsc = 0; dsc < data_size; ++dsc) {
					covar[cls][com](dsr,dsc) = stod(load_covars[dsc]);
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
	log_ofs << get_date_sec() + " load_file_mean_covar_mixdeg() called." << endl;
}
void Unlearn::calc_params(const vector<vector<double>> &input_data,vector<vector<int>>& class_data) {
	for (int cls = 0; cls < class_num; ++cls) {
		// k_mean()に突っ込むためにクラスごとに新しい2次元vectorを作ってる
		vector<vector<double>> cls_input;
		for (int d = 0; d < input_data.size(); ++d) {
			if (class_data[d][cls] == 1) {
				cls_input.push_back(input_data[d]);
			}
		}
		vector<int> component_label;
		k_means(cls_input, component_label, component_num, 1000);
		for (int com = 0; com < component_num; ++com) {
			// make_mean_covar()に突っ込むためにコンポーネントごとに2次元vectorを作ってる
			vector<vector<double>> tmp_component;
			for (int d = 0; d < cls_input.size(); ++d) {
				if (component_label[d] == com) {
					tmp_component.push_back(cls_input[d]);
				}
			}
			// for debug
			/*ofstream ofs(time_prefix +"\\component_datas_cls" + to_string(cls) + "_com" + to_string(c) + ".csv");
			for (int da = 0; da < tmp_component.size(); ++da) {
				for (int dsize = 0; dsize < data_size; ++dsize) {
					ofs << tmp_component[da][dsize] << ",";
				}
				ofs << endl;
			}
			ofs.close();*/
			// クラス　コンポーネントごとのデータを使って平均と分散を作る
			get_mean_covar(tmp_component, mean[cls][com], covar[cls][com]);
			// NOTE:行列式が0だったとき用　体格成分に少し足してごまかす．
			double det_covar = covar[cls][com].determinant();
			//cout << "cls:" << cls << ",com:" << com << ",det:" << setprecision(20) <<det_covar << endl;
			while (det_covar < 10e-15) {
				for (int d = 0; d < data_size; ++d) {
					covar[cls][com](d, d) += 1e-5;
				}
				det_covar = covar[cls][com].determinant();
				/*if (det_covar > 10e-15) {
					cout << "OK fixed det_covar:" <<setprecision(20) << det_covar<< endl;
				}
				else {
					cout << "NG fixed det_covar:" << setprecision(20) << det_covar << endl;
				}*/
			}
			// ついでに混同度alphaを設定する
			mix_deg[cls][com] = (1 - unlearn_mix_deg) * tmp_component.size() / input_data.size();
		}
	}
	log_ofs << get_date_sec() + " calc_params() called." << endl;
}
void Unlearn::k_means(const vector<vector<double>> &input,vector<int> &class_label,const int cluster_num, const int max_loop) {
	// 平均の初期値としてランダムな二つのデータを割り当てている．
	//class_label.resize(input.size());
	//std::random_device rnd;     // 非決定的な乱数生成器でシード生成機を生成
	//std::mt19937 mt(rnd());     //  メルセンヌツイスターの32ビット版、引数は初期シード
	//std::uniform_int_distribution<> rand_index(0, input.size()-1);     // [0, ?] 範囲の一様乱数
	//vector<vector<double>> cls_mean = vector<vector<double>>(cluster_num, vector<double>(input[0].size(), 0));
	//for (int cls = 0; cls < cluster_num; ++cls) {
	//	cls_mean[cls] = input[rand_index(mt)];
	//	// 同じデータが2回現れていないかチェック
	//	for (int exist_cls = 0; exist_cls < cls; ++exist_cls) {
	//		if (cls_mean[cls] == cls_mean[exist_cls]) {
	//			cls--;
	//			break;
	//		}
	//	}
	//}
	class_label.resize(input.size());// CHECK:resize()でちゃんとサイズ変更されるんだっけ？reserve()とかじゃなかったっけ？
	if (cluster_num == 1) {
		for (int in = 0; in < input.size(); ++in) {
			class_label[in] = 0;
		}
		log_ofs << get_date_sec() + " k_means() is called." << endl;
		return;
	}
	vector<vector<double>> cls_mean = vector<vector<double>>(cluster_num, vector<double>(input[0].size(), 0));
	vector<int> cls_mean_idx = make_rand_array_unique(cluster_num, 0, input.size() - 1);
	for (int i = 0; i < cluster_num; ++i) {
		cls_mean[i] = input[cls_mean_idx[i]];
	}
	// クラス分けが変わらなくなったら終了
	bool change = true;// クラス分けが変わったかのフラグ
	int loop = 0;// 繰り返し回数
	vector<int> cls_data_cnt(cluster_num, 0);
	while (change && loop < max_loop) {
		change = false;
		for (int in = 0; in < input.size(); ++in) {
			double min_dist = numeric_limits<double>::max();
			int min_cls = 0;
			// 全てのクラスについて2乗誤差を比較する
			for (int cls = 0; cls < cluster_num; ++cls) {
				double cls_dist = 0;
				for (int d = 0; d < input[0].size(); ++d) {
					cls_dist += pow(input[in][d] - cls_mean[cls][d], 2);
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
			vector<vector<double>> cls_sum(cluster_num, vector<double>(input[0].size(), 0));
			cls_data_cnt = vector<int>(cluster_num,0);// 後で平均を取る用，そのクラスに割り振られたデータの数
			for (int in = 0; in < input.size(); ++in) {
				int cls_label = class_label[in];
				cls_data_cnt[cls_label]++;
				for (int d = 0; d < input[0].size(); ++d) {
					cls_sum[cls_label][d] += input[in][d];
				}
			}
			for (int cls = 0; cls < cluster_num; ++cls) {
				for (int d = 0; d < input[0].size(); ++d) {
					cls_mean[cls][d] = cls_sum[cls][d] / cls_data_cnt[cls];
				}
			}
		}
		else {
			// NOTE:クラス平均値が変化しなくなっても平均が同じになっているときは一つをランダムに変える．
			for (int cls = 0; cls < cluster_num; ++cls) {
				if (cls_data_cnt[cls] == 0) {
					vector<int> tmp = make_rand_array_unique(1, 0, input.size()-1);
					cls_mean[cls] = input[tmp[0]];
					change = true;
				}
			}
			// NOTE:for debug
			/*if (!change) {
				for (int cls = 0; cls < cluster_num; ++cls) {
					cout <<"class:" <<cls << "," << cls_data_cnt[cls] << endl;
				}
			}*/
		}
		loop++;
	}
	log_ofs << get_date_sec() + " k_means() is called." << endl;
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
	int cluster_num = 2;
	int max_times = 1000;
	k_means(input, class_label, cluster_num, max_times);
	cout << accumulate(class_label.begin(), class_label.begin() + 100, 0) << endl;
	cout << accumulate(class_label.begin() + 100, class_label.begin() + 200, 0) << endl;;
	return 0;
}
// ちなみに行列のサイズが大きくなるとdeterminabt()とinverse()が死ぬらしいけど，よくわからん．
double Unlearn::gauss(vector<double>& input, const VectorXd& mean,MatrixXd& covar) {
	int input_size = input.size();
	VectorXd inp = Map<VectorXd>(&input[0],input_size);
	VectorXd dif = inp - mean;
	double det_covar = covar.determinant();
	// print_mat(covar);
	double kappa; // TODO:change the variable name to understandable name.
	if (is_approximate)
		kappa = pow(2 * M_PI, -input_size / 2.0) / approximate_sqrt(det_covar);
	else 
		kappa = pow(2 * M_PI, -input_size / 2.0) * pow(det_covar, -1 / 2.0);

	if (isnan(kappa)) {
		cerr << "kappa value:" + to_string(kappa) + " is invalid!!" << endl;
		cout << "det_covar:" + to_string(det_covar) << endl;
	}

	MatrixXd arg_of_exp = -1 * (dif.transpose() * covar.inverse() * dif) / 2;
	double rtn_val;
	if (is_approximate) 
		rtn_val = kappa * approximate_exp(arg_of_exp(0, 0));
	else 
		rtn_val = kappa * exp(arg_of_exp(0, 0));
	if (isnan(rtn_val)) {
		cout << "covar.inverse()" << endl;
		print_mat(covar.inverse());
		cerr << "arg of exp value:" + to_string(arg_of_exp(0,0)) + " is invalid!!" << endl;
		cerr << "return value "+to_string(rtn_val)+" is invalid!!" << endl;
	}
	return rtn_val;
}
void Unlearn::print_mat(const MatrixXd& mat) {
	for (int r = 0; r < mat.rows(); r++) {
		for (int c = 0; c < mat.cols(); c++) {
			cout << mat(r, c) << " ";
		}
		cout << endl;
	}
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

	double det_covar = covar.determinant();
	if (is_approximate)
		kappa = pow(2 * M_PI, -input_size / 2.0) / approximate_sqrt(det_covar);
	else
		kappa = pow(2 * M_PI, -input_size / 2.0) * pow(det_covar, -1 / 2.0);

	if (isnan(kappa)) {
		cerr << "kappa value:" + to_string(kappa) + " is invalid!!" << endl;
	}
	MatrixXd widened_covar = complementary_covar_coef * covar;
	MatrixXd arg_of_exp_former = -1 * (dif.transpose() * widened_covar.inverse() * dif) / 2;
	MatrixXd arg_of_exp_latter = -1 * (dif.transpose() * covar.inverse() * dif) / 2;
	double rtn_val;
	if(is_approximate)
		rtn_val = kappa / (pow(complementary_covar_coef, static_cast<double>(input_size) / 2) - 1)
			* (approximate_exp(arg_of_exp_former(0, 0)) - approximate_exp(arg_of_exp_latter(0, 0)));
	else
		rtn_val = kappa / (pow(complementary_covar_coef, static_cast<double>(input_size) / 2) - 1) * (exp(arg_of_exp_former(0,0)) - exp(arg_of_exp_latter(0,0))); 
	if (isnan(rtn_val)) {
		print_mat(widened_covar);
		cerr << "return value " + to_string(rtn_val) + " is invalid!!" << endl;
	}
	return rtn_val;
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
void Unlearn::get_mean_covar(vector<vector<double>>& input, VectorXd &mean, MatrixXd & covar) {
	if (input.size() <= 1) {
		cerr << "input size is under zero!!" << endl;
		return;
	}
	const int input_data_size = input[0].size();
	// 平均
	mean = VectorXd::Zero(input_data_size);
	for (int i = 0; i < input.size(); ++i) {
		mean += Map<VectorXd>(&input[i][0],input_data_size);
	}
	mean /= input.size();
	vector<VectorXd> deviation_data = vector<VectorXd>(input.size(),VectorXd::Zero(input_data_size));
	for (int i = 0; i < input.size();++i){
		deviation_data[i] = Map<VectorXd>(&input[i][0], input_data_size) - mean;// 偏差行列
	}
	// 共分散行列
	covar = MatrixXd::Zero(input_data_size, input_data_size);
	for (int d = 0; d < deviation_data.size(); ++d) {
		// NOTE:直接covarに+=したらエラー吐かれた
		MatrixXd tmp_mat = deviation_data[d] * deviation_data[d].transpose();
		covar += tmp_mat;
	}
	covar /= (deviation_data.size() - 1);
	log_ofs << get_date_sec() + " get_mean_covar() is called." << endl;
}
//int Unlearn::get_mean_covar_test() {
//	vector<double> mean;
//	vector<vector<double>> covar;
//	const int data_num = 200;
//	vector<vector<double>> input = vector<vector<double>>(data_num, vector<double>(2, 0));
//	std::random_device seed_gen;     // 非決定的な乱数生成器でシード生成機を生成
//	std::mt19937 mt(seed_gen());     //  メルセンヌツイスターの32ビット版、引数は初期シード
//	std::normal_distribution<> ndist(5.0, 2.0);     // 平均5，標準偏差1.0のガウス分布
//	std::normal_distribution<> ndist1(4.0, 1.0);     // 平均5，標準偏差1.0のガウス分布
//	for (int da = 0; da < data_num; ++da) {
//		input[da][0] = ndist(mt);
//		input[da][1] = ndist1(mt);
//	}
//	get_mean_covar(input, mean, covar);
//	for (auto me : mean) {
//		cout << me << ",";
//	}
//	cout << endl;
//	for (auto cova : covar) {
//		for (auto co : cova) {
//			cout << co << ",";
//		}
//		cout << endl;
//	}
//	return 0;
//}
void Unlearn::learn_beta(vector<vector<double>> &verification_data,vector<vector<int>> &verification_class_data) {
	const unsigned int dont_change_threshold = 10;
	for (int cls = 0; cls < class_num; ++cls) {
		// NOTE: デバック用
		ofstream ofs(time_prefix + "\\class"+to_string(cls)+"beta.csv");
		check_open_file(ofs, time_prefix + "\\class" + to_string(cls) + "beta.csv");
		double energy = 0;
		class_beta[cls] -= delta_beta;// NOTE: 最初に足しちゃうから先にひいてるよ
		unsigned int times = 0;
		unsigned int dont_change_cnt = 0;
		double previous_energy = 0;
		// NOTE:Count datas of each class.

		vector<unsigned int> cls_data_num(class_num, 0);
		for (const vector<int> v_data : verification_class_data) {
			cls_data_num[distance(v_data.begin(), max_element(v_data.begin(), v_data.end()))]++;
		}
		while (energy <= beta_threshold && (times++) < 150) {
			energy = 0;// NOTE: エネルギー関数初期化
			class_beta[cls] += delta_beta;// NOTE:βをちょっと増やす
			cout << "class:"<<cls<<", times:" << times << " beta:" << class_beta[cls];// NOTE:デバック用
			vector<double> prob;
			for (int d = 0; d < verification_data.size(); ++d) {
				calc_probability(verification_data[d], prob);
				// NOTE: 正解がこのクラス
				if (verification_class_data[d][cls] == 1) {
					// NOTE: このクラスの確率が一番高い
					if (distance(prob.begin(), max_element(prob.begin(), prob.end())) == cls + 1) {
						energy += 1;
					}
				}
			}
			energy /= cls_data_num[cls];
			cout << " " << TO_STRING(energy) << ":" << energy << endl;
			if (energy < previous_energy && energy == 0) {
				cerr << "energy is invalid!!" << endl;
			}
			ofs << to_string(times) + "," + to_string(class_beta[cls]) + "," + to_string(energy) << endl;
			// NOTE:dont_change_threshold回recallが同じだったら学習を止める．
			if (energy <= previous_energy) {
				dont_change_cnt += 1;
			}else {
				dont_change_cnt = 0;
			}
			if (dont_change_cnt >= dont_change_threshold) {
				break;
			}
			previous_energy = energy;
		}
	}
	log_ofs << get_date_sec() + " learn_beta() is called." << endl;
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
			// 学習クラスの確率算出
			MatrixXd beta_covar = class_beta[cls] * covar[cls][com];
			class_probability[cls] += mix_deg[cls][com] * gauss(input_data, mean[cls][com], beta_covar);
			for (int cls = 0; cls < class_num; ++cls) {
				if (isnan(class_probability[cls])) {
					cerr << "cls_probabiligy" + to_string(cls)+"_"+ to_string(com) + " is nan" << endl;
				}
			}
			// 未学習クラスの確率算出
			unlearn_probability *= hgauss(input_data, mean[cls][com], covar[cls][com]);
			double output_hga = hgauss(input_data, mean[cls][com], covar[cls][com]);
			//uln_ofs <<  output_hga <<",";
		}
		/*learn_class_ofs << class_probability[cls] << ",";
		if (cls >= cluster_num-1) {
			learn_class_ofs << endl;
		}*/
		sum_all_probability += class_probability[cls];
	}
	
	if (isnan(unlearn_probability)) {
		cerr << "unlearn_probabiligy is nan" << endl;
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
//void Unlearn::test() {
//	//VectorXd mu = { 1 };
//}
void Unlearn::open_class_csv(const string filename) {
	class_ofs.open(filename);	if (class_ofs.fail()) {throw "can't opne " + filename;}
}
// 確率分布を出すのに必要な各種の値をファイルに保存する．
void Unlearn::out_file_mean() const {
	// 平均の保存
	const string filename = time_prefix + "\\mean.csv";
	ofstream ofs(filename); if (ofs.fail()) { throw "Can't open " + filename; }
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			for (int mean_idx = 0; mean_idx < data_size;++mean_idx) {
				ofs << mean[cls][com](mean_idx) << ",";
			}
			ofs << endl;
		}
	}
	ofs.close();
}

void Unlearn::out_file_covar() const {
	// 共分散の保存
	const string filename = time_prefix + "\\covar.csv";
	ofstream ofs(filename); if (ofs.fail()) { throw "Can't open " + filename; }
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			for (int row = 0; row < data_size; ++row) {
				for (int col = 0; col < data_size;++col) {
					ofs << covar[cls][com](row,col) << ",";
				}
				ofs << endl;
			}
		}
	}
	ofs.close();
}
void Unlearn::out_file_params() const {
	// その他の値の保存
	const string filename = time_prefix + "\\params.csv";
	ofstream ofs(filename); if (ofs.fail()) { throw "Can't open " + filename; }
	ofs << "cluster_num,component_num,data_size,unlearn_mixing_degree,normalize_unlearn,beta_threshold,delta_beta,complementary_covar_coef" << endl;
	ofs << class_num << "," << component_num << "," << data_size << "," << unlearn_mix_deg << ","
		<< normalize_unlearn << "," << beta_threshold << "," << delta_beta << ","
		<< complementary_covar_coef << endl;
	ofs << "beta " << endl;
	for (int cls = 0; cls < class_num; ++cls) {
		ofs << class_beta[cls] << ",";
	}
	ofs << endl;
	ofs.close();
}
void Unlearn::out_file_mix_deg() const {
	// 混合度の値．
	const string filename = time_prefix + "\\mix_deg.csv";
	ofstream ofs(filename); if (ofs.fail()) { throw "Can't open " + filename; }
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_num; ++com) {
			ofs << mix_deg[cls][com] << "," << endl;
		}
	}
	ofs.close();
}
void Unlearn::out_file_data() const{
	out_file_mean();
	out_file_covar();
	out_file_params();
	out_file_mix_deg();
}
/*
 * NOTE:calculate accuracy and precision, recall, F-measure
*/
vector<vector<double>>& Unlearn::evaluate(vector<vector<double>>& test_data, const vector<vector<int>>& class_data, 
	const bool output2csv){
	vector<int> class_true_positive(class_num + 1, 0), class_true_negative(class_num + 1, 0),
		        class_false_positive(class_num + 1, 0), class_false_negative(class_num + 1, 0);
	// NOTE:confusion_matrix[true_class][predicted_class]
	vector<vector<int>> confusion_matrix(class_num + 1, vector<int>(class_num + 1, 0));
	for (int d = 0; d < test_data.size(); ++d) {
		vector<double> prob;
		calc_probability(test_data[d], prob);
		int max_class_index = distance(class_data[d].begin(), max_element(class_data[d].begin(), class_data[d].end()));
		int max_prob_index = distance(prob.begin(), max_element(prob.begin(), prob.end()));
		confusion_matrix[max_class_index][max_prob_index]++;
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
	std::cout << "confusion_matrix" << endl;
	std::cout << "         class_0  class_1  class_2  class_3" << endl;
	for (int true_cls = 0; true_cls <= class_num; ++true_cls) {
		std::cout << "class_" << to_string(true_cls);
		for (int pred_cls = 0; pred_cls <= class_num; ++pred_cls) {
			std::cout << "     "<<setw(4) << confusion_matrix[true_cls][pred_cls];
		}
		std::cout << std::endl;
	}
	// NOTE:result[class][indeces] Indeces's order is [accuracy, precision, recall, F-measure]
	vector<vector<double>> result(class_num + 1, vector<double>(4, 0.0));
	for (int cls = 0; cls < class_num + 1; ++cls) {
		result[cls][0] = static_cast<double>(class_true_positive[cls] + class_true_negative[cls])
			/ (class_true_positive[cls] + class_false_positive[cls] + class_false_negative[cls] + class_true_negative[cls]);
		result[cls][1] = static_cast<double>(class_true_positive[cls]) / (class_true_positive[cls] + class_false_positive[cls]);
		result[cls][2] = static_cast<double>(class_true_positive[cls]) / (class_true_positive[cls] + class_false_negative[cls]);
		result[cls][3] = 2 * result[cls][1] * result[cls][2] / (result[cls][1] + result[cls][2]);
		std::cout << "class: " << setw(3) << cls 
			<<", accuracy: "   << fixed << setprecision(8) << result[cls][0] 
			<< ", precision: " << fixed << setprecision(8) << result[cls][1]
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
	std::cout << "macro_average" << endl
		<< "accuracy: " << fixed << setprecision(8) << macro_average[0] << endl
		<< "precision: " << fixed << setprecision(8) << macro_average[1] << endl
		<< "recall: " << fixed << setprecision(8) << macro_average[2] << endl
		<< "F-measure(ave): " << fixed << setprecision(8) << macro_average[3] << endl
		<< "F-measure(calc): " << fixed << setprecision(8)
		<< 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;
	double micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();
	std::cout << "micro_average: " << fixed << setprecision(8) << micro_average << endl;
	if (output2csv) {
		ofstream ofs(time_prefix + "\\evaluate.csv");
		ofs << "confusion matrix" << endl;
		for (int true_cls = 0; true_cls <= class_num; ++true_cls) {
			for (int pred_cls = 0; pred_cls <= class_num; ++pred_cls) {
				ofs << confusion_matrix[true_cls][pred_cls]<<",";
			}
			ofs << std::endl;
		}
		ofs << "class,accuracy,precision,recall,F-measure" << endl;
		for (int cls = 0; cls < class_num + 1; ++cls) {
			ofs << cls << "," << result[cls][0] << "," << result[cls][1] << "," << result[cls][2] << "," << result[cls][3] << endl;
		}
		ofs << "macro_average" << endl
			<< "accuracy," << macro_average[0] << endl
			<< "precision," << macro_average[1] << endl
			<< "recall," << macro_average[2] << endl
			<< "F-measure(ave)," << macro_average[3] << endl
			<< "F-measure(calc),"
			<< 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;
		double micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();
		ofs << "micro_average," << micro_average << endl;
	}
	log_ofs << get_date_sec() + " evaluate() is called." << endl;
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