#include <vector>
#include<iostream>
#include<algorithm>
#include<numeric>
#include<fstream>
#include<string>
#include<sstream>
#include<Eigen/Core>
using namespace std;
using namespace Eigen;
void get_mean_covar(vector<vector<double>>& input, vector<double>& mean, vector<vector<double>>& covar);
vector<vector<double>> get_vector_from_file(const string filename);
int main() {
	VectorXd a=VectorXd::Zero(3);
	VectorXd b = VectorXd::Zero(3);
	for (int i = 0; i < 3; ++i) {
		a(i) = i + 1;
		b(i) = i + 4;
	}
	MatrixXd c = a * b.transpose();
	cout <<  c << endl;
	return 0;
}
void get_mean_covar(vector<vector<double>>& input, vector<double>& mean, vector<vector<double>>& covar) {
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
	covar = vector<vector<double>>(input_data_size, vector<double>(input_data_size, 0));
	for (int row = 0; row < deviation_data[0].size(); ++row) {
		for (int cols = row; cols < deviation_data[0].size(); ++cols) {
			double var = 0;
			if (deviation_data.size() > 1) {
				var = accumulate(deviation_data.begin(), deviation_data.end(), 0.0, [&row, &cols](double acc, vector<double>& vec) {return acc + vec[row] * vec[cols]; }) / (deviation_data.size() - 1);
			}
			else {
				cerr << "deviation_data size is 1" << endl;
			}
			covar[row][cols] = var;
			covar[cols][row] = var;
		}
	}
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
	while (std::getline(ifs, str)) {
		stringstream ss{ str };
		vector<double> tmp;
		while (std::getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}