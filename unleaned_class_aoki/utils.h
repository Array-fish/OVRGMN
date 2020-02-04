#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<fstream>
#include<string>
#include<vector>
#include<iostream>
std::string get_date_sec();
int check_open_file(const std::ofstream& ofs, const std::string filename);
std::vector<std::string> split(const std::string& instr, const char delimiter);
bool getFileNames(std::string folderPath, std::vector<std::string>& file_names);
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);
// csvファイルの内容を2次元vector<T>に変換する
template<class T>
std::vector<std::vector<T>> get_vector_from_file(const std::string filename) {
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		std::cerr << "Can't open " + filename << std::endl;
		exit(-1);
	}
	std::string str, str1;
	std::vector<std::vector<T>> data;
	while (std::getline(ifs, str)) {
		std::stringstream ss{ str };
		std::vector<T> tmp;
		while (std::getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}
#endif UTILS_H

