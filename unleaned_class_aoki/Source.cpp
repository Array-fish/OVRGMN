#define aCREATE_FROM_FILE
#include"Header.h"
#include"utils.h"
#include<string>
#include<fstream>
#include<random>
#include<algorithm>
#include<iostream>
#include<stdexcept>
vector<vector<double>> get_vector_from_file(const string filename);
vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);
int main() {
	// データを取り込み　学習データ，その正解クラスのデータ
	vector<vector<double>> learn_data      = get_vector_from_file("..\\..\\unlearned_files\\created_data\\004_learn_data.csv");
	vector<vector<double>> class_data      = get_vector_from_file("..\\..\\unlearned_files\\created_data\\004_learn_class_data.csv");
	vector<vector<double>> test_data       = get_vector_from_file("..\\..\\unlearned_files\\created_data\\004_test_data.csv");
	vector<vector<double>> test_class_data = get_vector_from_file("..\\..\\unlearned_files\\created_data\\004_test_class_data.csv");
#ifndef CREATE_FROM_FILE
	// モデルの条件
	std::random_device rnd;     // 非決定的な乱数生成器でシード生成機を生成
	std::mt19937 engine(rnd());     //  メルセンヌツイスターの32ビット版、引数は初期シード
	std::uniform_real_distribution<double> beta_0(0.001, 0.01);
	// Unlearn(int class_num, int component_num, double beta, double unlearn_mixing_degree,
	//     double normalize_unlearn, double beta_threshold, double delta_beta, double complementary_covar_coef);
	// beta:乱数, unlearn_mixing_degree:alpha_0 これは適当．今は0.01,  normalize_unlearn:未学習クラスの正規化項．これは論文の値
	// beta_threshold:βを決めるときの閾値．論文の値, beta_delta:βを決めるときの変化量．論文の値, 
	// complementary_covar_coef:h_gaussを算出するときの値．論文の値,
	const int    class_num         = 3;
	const int    component_num     = 2;
	const double unlearn_mix_deg   = 0.01;
	const double normalize_unlearn = 0.5;
	const double beta_threshold    = 0.99;
	const double delta_beta        = 0.01;
	const double complementary_covar_coef = 10.0;
	Unlearn uln(class_num, component_num, beta_0(engine), unlearn_mix_deg,
		normalize_unlearn, beta_threshold, delta_beta, complementary_covar_coef);
#else
	// NOTE:ここは出力部との対応が取れているか確かめること．
	const string file_directory = "..\\..\\unlearned_files\\2019_12_1_18_26_51";
	// prepare variables
	string filename;
	ifstream ifs;
	string line;
	// load parameters and beta, then create new instance.
	filename = file_directory + "\\params.csv";
	ifs.open(filename); if (ifs.fail()) { throw "can't opne " + filename; }
	getline(ifs, line); getline(ifs, line); // NOTE: Pass headers.
	vector<string> load_params = split(line, ',');
	const int    class_num         = stoi(load_params[0]);
	const int    component_num     = stoi(load_params[1]);
	const int    data_size         = stoi(load_params[2]);
	const double unlearn_mix_deg   = stod(load_params[3]);
	const double normalize_unlearn = stod(load_params[4]);
	const double beta_threshold    = stod(load_params[5]);
	const double delta_beta        = stod(load_params[6]);
	const double ccomplementary_covar_coef = stod(load_params[7]);
	getline(ifs, line); getline(ifs, line);
	load_params = split(line, ',');
	vector<double> class_beta(class_num);
	for (int cls = 0; cls < class_num; ++cls) {
		class_beta[cls] = stod(load_params[cls]);
	}
	ifs.close();

	Unlearn uln(class_num, component_num, data_size, class_beta, unlearn_mix_deg, normalize_unlearn,
		beta_threshold, delta_beta, ccomplementary_covar_coef);
	uln.load_file_mean_covar_mixdeg(file_directory);
#endif
	uln.set_approximate(true);
	// 学習データを　beta出すための検証データと普通の最尤法のための学習データに分ける．
	// とりあえず学習データの半分を検証用データに使う
	vector<int> verification_index;
	for (int cls = 0; cls < class_num; ++cls) {
		vector<int> class_index = make_rand_array_unique(learn_data.size() / class_num / 2, 0, learn_data.size() / class_num - 1);
		int class_first_index = learn_data.size() / class_num * cls;
		for (int idx : class_index) {
			verification_index.push_back(class_first_index + idx);
		}
	}
	vector<vector<double>> verification_data;
	vector<vector<double>> verification_class_data;
	
	for (int i : verification_index) {
		verification_data.push_back(learn_data[i]);
		verification_class_data.push_back(class_data[i]);
	}
	// ここ途中でindexError出ないように後ろから消してます．
	for (int i = verification_index.size() - 1; i >= 0; --i) {
		learn_data.erase(learn_data.begin() + verification_index[i]);
		class_data.erase(class_data.begin() + verification_index[i]);
}
#ifndef CREATE_FROM_FILE
	// 学習データをクラスごとにk-means法に放り込んでクラス分け，
	// クラス，コンポーネントごとにパラメータの値を出す．
	// class_data_altはcalc_params()関数を作った時の気分でクラスの入力をindexにしてしまったので変換している
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
	// パラメータを使って検証データとともにbetaを出す．
	uln.learn_beta(verification_data,verification_class_data);
#endif
	// 正解のクラス/全てのデータで識別率をみる
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
	// TODO:各クラスごとのaccuracy, recall, preciseを出せ．
	cout << "accuracy = " << static_cast<double>(positive) / test_data.size() << endl; 
	uln.evaluate(test_data, test_class_data, true);
	// 検証用に算出した平均値と分散をファイルに出力する．
	uln.out_file_mean_covar_params();

	// GIVE UP:テストデータを作ることが出来なかったので，test()をつくるのはあきらめた
	// test()
	//Unlearn uln(3, 2, 0.01, 0.01, 0.5, 0.99, 0.01, 10);
	// uln.k_means_test();
	// uln.gauss_test();
	//uln.hgauss_test();
	//uln.get_mean_covar_test();
	return 0;
}
// csvファイルの内容を2次元vector<double>に変換する
vector<vector<double>> get_vector_from_file(const string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		cerr << "Can't open " + filename << endl;
		exit(-1);
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
	if (max_min_diff < size) throw std::invalid_argument("引数が異常です");

	std::vector<int> tmp;
	std::random_device rnd;     // 非決定的な乱数生成器でシード生成機を生成
	std::mt19937 engine(rnd());     //  メルセンヌツイスターの32ビット版、引数は初期シード
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