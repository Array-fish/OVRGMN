#define	aCREATE_FROM_FILE
#include"Header.h"
#include"utils.h"
#include<string>
#include<fstream>
#include<random>
#include<algorithm>
#include<iostream>
#include<stdexcept>
#include<chrono>
#include<thread>
int main() {
	// データを取り込み　学習データ，その正解クラスのデータ
	const string data_folder = "C:\\Users\\watanabe\\Desktop\\uekusa\\unlearned_files\\EMG_signals_for_forearm_classification\\";
	const string learn_data_path = data_folder + "EMG1_train_data2.csv";
	const string learn_cls_path = data_folder  + "EMG1_train_cls2.csv";
	const string test_data_path = data_folder  + "EMG1_test_data2.csv";
	const string test_cls_path = data_folder   + "EMG1_test_cls2.csv";
	vector<vector<double>> learn_data   = get_vector_from_file<double>(learn_data_path);
	vector<vector<int>> class_data      = get_vector_from_file<int>(learn_cls_path);
	vector<vector<double>> test_data    = get_vector_from_file<double>(test_data_path);
	vector<vector<int>> test_class_data = get_vector_from_file<int>(test_cls_path);
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
	const int    class_num         = 4;
	const int    component_num     = 2;
	const int    data_size         = 4;
	const double unlearn_mix_deg   = 0.01;
	const double normalize_unlearn = 0.5;
	const double beta_threshold    = 0.99;
	const double delta_beta        = 0.01;
	const double complementary_covar_coef = 10.0;

	const int repeat_num = 50;
	for (int re = 0; re < repeat_num; ++re) {
		// データを取り込み　学習データ，その正解クラスのデータ
		vector<vector<double>> learn_data   = get_vector_from_file<double>(learn_data_path);
		vector<vector<int>> class_data      = get_vector_from_file<int>(learn_cls_path);
		vector<vector<double>> test_data    = get_vector_from_file<double>(test_data_path);
		vector<vector<int>> test_class_data = get_vector_from_file<int>(test_cls_path);
		Unlearn uln(class_num, component_num, data_size, beta_0(engine), unlearn_mix_deg,
			normalize_unlearn, beta_threshold, delta_beta, complementary_covar_coef,
			"C:\\Users\\watanabe\\Desktop\\uekusa\\unlearned_files\\EMG_signals_for_forearm_classification\\data2\\bo_app\\");
		vector<int> verification_index;
		vector<int>cls_data_num(class_num);
		for (int d = 0; d < learn_data.size(); d++) {
			cls_data_num[distance(class_data[d].begin(), max_element(class_data[d].begin(), class_data[d].end()))]++;
		}
		int class_first_index = 0;
		for (int cls = 0; cls < class_num; ++cls) {
			vector<int> class_index = make_rand_array_unique(cls_data_num[cls] / 2, 0, cls_data_num[cls] - 1);
			for (const int idx : class_index) {
				verification_index.push_back(class_first_index + idx);
			}
			class_first_index += cls_data_num[cls];
		}
		vector<vector<double>> verification_data;
		vector<vector<int>> verification_class_data;
		for (int i : verification_index) {
			verification_data.push_back(learn_data[i]);
			verification_class_data.push_back(class_data[i]);
		}
		// ここ途中でindexError出ないように後ろから消してます．
		for (int i = verification_index.size() - 1; i >= 0; --i) {
			learn_data.erase(learn_data.begin() + verification_index[i]);
			class_data.erase(class_data.begin() + verification_index[i]);
		}
#else
	// NOTE:ここは出力部との対応が取れているか確かめること．
	const string folder_path = "C:\\Users\\watanabe\\Desktop\\uekusa\\unlearned_files\\EMG_signals_for_forearm_classification\\data2\\no_app";
	vector<string> file_names;
	if (!getFileNames(folder_path, file_names)) {
		cerr << "Cannot find file names." << endl;
	}
	for(int par = 0;par < file_names.size(); ++par){
		this_thread::sleep_for(std::chrono::seconds(1)); // NOTE:同じ秒で実行するとfolderの名前被りでerror吐くから
		const string file_directory = file_names[par];
		// prepare variables
		string filename;
		ifstream ifs;
		string line;
		// load parameters and beta, then create new instance.
		filename = file_directory + "\\params.csv";
		ifs.open(filename); if (ifs.fail()) { cerr << "can't opne " + filename << endl; exit(-1); }
		getline(ifs, line); getline(ifs, line); // NOTE: Pass headers.
		vector<string> load_params = split(line, ',');
		const int    class_num = stoi(load_params[0]);
		const int    component_num = stoi(load_params[1]);
		const int    data_size = stoi(load_params[2]);
		const double unlearn_mix_deg = stod(load_params[3]);
		const double normalize_unlearn = stod(load_params[4]);
		const double beta_threshold = stod(load_params[5]);
		const double delta_beta = stod(load_params[6]);
		const double ccomplementary_covar_coef = stod(load_params[7]);
		getline(ifs, line); getline(ifs, line);
		load_params = split(line, ',');
		vector<double> class_beta(class_num);
		for (int cls = 0; cls < class_num; ++cls) {
			class_beta[cls] = stod(load_params[cls]);
		}
		ifs.close();

		Unlearn uln(class_num, component_num, data_size, class_beta, unlearn_mix_deg, normalize_unlearn,
			beta_threshold, delta_beta, ccomplementary_covar_coef,
			"C:\\Users\\watanabe\\Desktop\\uekusa\\unlearned_files\\EMG_signals_for_forearm_classification\\data2\\pr_app\\");
		uln.load_file_mean_covar_mixdeg(file_directory);
#endif
	uln.set_approximate(true);

	// 学習データを　beta出すための検証データと普通の最尤法のための学習データに分ける．
	// とりあえず学習データの半分を検証用データに使う
	// NOTE:前提として学習データにおいて同じクラスのデータは1塊に連続しているとする．
	vector<int> verification_index;
	vector<int>cls_data_num(class_num);
	for (int d = 0; d < learn_data.size(); d++) {
		cls_data_num[distance(class_data[d].begin(), max_element(class_data[d].begin(), class_data[d].end()))]++;
	}
	int class_first_index = 0;
	for (int cls = 0; cls < class_num; ++cls) {
		vector<int> class_index = make_rand_array_unique(cls_data_num[cls] / 2, 0, cls_data_num[cls] - 1);
		for (const int idx : class_index) {
			verification_index.push_back(class_first_index + idx);
		}
		class_first_index += cls_data_num[cls];
	}
	vector<vector<double>> verification_data;
	vector<vector<int>> verification_class_data;
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
	uln.calc_params(learn_data, class_data);
	uln.out_file_mean();
	uln.out_file_covar();
	uln.out_file_mix_deg();
	// パラメータを使って検証データとともにbetaを出す．
	uln.learn_beta(verification_data, verification_class_data);
	uln.out_file_params();
#endif
	uln.evaluate(test_data, test_class_data, true);
	}

	// GIVE UP:テストデータを作ることが出来なかったので，test()をつくるのはあきらめた
	// test()
	//Unlearn uln(3, 2, 0.01, 0.01, 0.5, 0.99, 0.01, 10);
	// uln.k_means_test();
	// uln.gauss_test();
	//uln.hgauss_test();
	//uln.get_mean_covar_test();
	return 0;
}
