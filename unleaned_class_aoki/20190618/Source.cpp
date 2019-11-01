/*
出力までは書いたから，誤差関数を出すところを書いてほしい．
まだ生の確率しか出せていないからね．
最終的な目標は各パラメータがどんな役割を果たすか見ること
だからグラフを出したいからmean,covarとかの値をどっかのファイルに保存してやりたい
2019/06/04
途中からenergyがすべて0になるからβの学習が出来てない．
原因を探れ，入力と平均は大丈夫だったから分散が怪しいかも
2019/06/11
*/
#include"Header.h"
#include<string>
#include<fstream>
#include<random>
#include<algorithm>
#include<iostream>
vector<vector<double>> get_vector_from_file(const string filename);
vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);
int main() {
	// データを取り込み　学習データ，その正解クラスのデータ
	vector<vector<double>> learn_data = get_vector_from_file("lea_sig.csv");
	vector<vector<double>> class_data = get_vector_from_file("lea_class.csv");
	vector<vector<double>> test_data = get_vector_from_file("dis_sig2.csv");
	vector<vector<double>> test_class_data = get_vector_from_file("dis_class.csv");
	// 学習データを　beta出すための検証データと普通の最尤法のための学習データに分ける．
	// とりあえず学習データの半分を検証用データに使う
	vector<int> verification_index = make_rand_array_unique(learn_data.size() / 2, 0, learn_data.size() - 1);
	vector<vector<double>> verification_data;
	vector<vector<double>> verification_class_data;
	for (int i : verification_index) {
		verification_data.push_back(learn_data[i]);
		verification_class_data.push_back(class_data[i]);
	}
	for (int i = verification_index.size() - 1; i >= 0; --i) {
		learn_data.erase(learn_data.begin() + verification_index[i]);
		class_data.erase(class_data.begin() + verification_index[i]);
	}

	int class_num = 3;
	int component_num = 2;
	std::random_device rnd;     // 非決定的な乱数生成器でシード生成機を生成
	std::mt19937 engine(rnd());     //  メルセンヌツイスターの32ビット版、引数は初期シード
	std::uniform_real_distribution<double> beta_0(0.001, 0.01);
	// Unlearn(int class_num, int component_num, double beta, double zeta, double psi, double beta_threshold, double delta_beta, double epsilon);
	// beta:乱数, zeta:alpha_0 これは適当．今は0.01, psi:未学習クラスの正規化項．これは論文の値
	// beta_threshold:βを決めるときの閾値．論文の値, beta_delta:βを決めるときの変化量．論文の値, epsilon:h_gaussを算出するときの値．論文の値,
	//double beta0 = ;
	//Unlearn uln(class_num, component_num, beta_0(engine), 0.01, 0.5, 0.99, 0.01, 10);
	Unlearn uln(class_num, component_num, 8, 0.01, 0.5, 0.99, 0.1, 10);// ちょっと簡単にするためにbeta_0とdelta_betaを変えてるよ上のが本物
	// 学習データをクラスごとにk-means法に放り込んでクラス分け，
	// クラス，コンポーネントごとにパラメータの値を出す．
	// class_data_altはcalc_params()関数を作った時の気分でクラスの入力をindexにしてしまったので変換している
	// ex) [0 0 1] -> [2]
	vector<double> class_data_alt;
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
	// 学習データを使って識別を試す．
	// 評価関数 -T^n*log(probability);正解クラスの確率の和の平均
	double j_x = 0;
	for (int d = 0; d < test_class_data.size(); ++d) {
		vector<double> prob;
		uln.calc_probability(test_data[d], prob);
		for (int cls = 0; cls < test_class_data[0].size(); ++cls) {
			j_x += test_class_data[d][cls] * prob[cls];
		}
	}
	cout << "J(x)= " << j_x/test_data.size() << endl;
	// 検証用に算出した平均値と分散をファイルに出力する．
	uln.out_file_mean_covar_params("test_004");
	return 0;
}
// csvファイルの内容を2次元vector<double>に変換する
vector<vector<double>> get_vector_from_file(const string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		throw "Can't open " + filename;
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
	if (max_min_diff < size) throw std::runtime_error("引数が異常です");

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