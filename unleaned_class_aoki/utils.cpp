#include"utils.h"
#include<iostream>
#include<iomanip>
#include<sstream>
#include<string>
std::string get_date_sec() {
	time_t t = time(nullptr);
	struct tm lt;
	const errno_t error = localtime_s(&lt, &t);
	// put to stringstream
	std::stringstream ss;
	ss << lt.tm_year + 1900;
	ss << "_";
	ss << lt.tm_mon + 1;
	ss << "_";
	ss << lt.tm_mday;
	ss << "_";
	ss << lt.tm_hour;
	ss << "_";
	ss << lt.tm_min;
	ss << "_";
	ss << lt.tm_sec;
	return ss.str();
}

int check_open_file(const std::ofstream& ofs, const std::string filename) {
	try{
		if (ofs.fail()){
			throw "can't opne " + filename;
		}
	}
	catch (std::string errstr) {
		std::cerr << errstr << std::endl;
	}
}
std::vector<std::string> split(const std::string& instr, const char delimiter) {
	std::istringstream stream(instr);
	std::string field;
	std::vector<std::string> result;
	while (getline(stream, field, delimiter)) {
		result.push_back(field);
	}
	return result;
}