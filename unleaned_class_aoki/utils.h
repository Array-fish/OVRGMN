#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<fstream>
#include<string>
#include<vector>
std::string get_date_sec();
int check_open_file(const std::ofstream& ofs, const std::string filename);
std::vector<std::string> split(const std::string& instr, const char delimiter);
bool getFileNames(std::string folderPath, std::vector<std::string>& file_names);
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);
#endif UTILS_H

