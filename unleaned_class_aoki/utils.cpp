#include"utils.h"
#include<iostream>
#include<iomanip>
#include<sstream>
#include<string>
#include<filesystem>
#include<random>
#include<algorithm>
#include<vector>
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
	while (std::getline(stream, field, delimiter)) {
		result.push_back(field);
	}
	return result;
}
/**
* @brief �t�H���_�ȉ��̃t�@�C���ꗗ���擾����֐�
* @param[in]    folderPath  �t�H���_�p�X
* @param[out]   file_names  �t�@�C�����ꗗ
* return        true:����, false:���s
* NOTE: C++17
*/
bool getFileNames(std::string folderPath, std::vector<std::string>& file_names)
{
	using namespace std::filesystem;
	directory_iterator iter(folderPath), end;
	std::error_code err;

	for (; iter != end && !err; iter.increment(err)) {
		const directory_entry entry = *iter;

		file_names.push_back(entry.path().string());
		printf("%s\n", file_names.back().c_str());
	}

	/* �G���[���� */
	if (err) {
		std::cout << err.value() << std::endl;
		std::cout << err.message() << std::endl;
		return false;
	}
	return true;
}
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max) {
	if (rand_min > rand_max) std::swap(rand_min, rand_max);
	const size_t max_min_diff = static_cast<size_t>(rand_max - rand_min + 1);
	if (max_min_diff < size) throw std::invalid_argument("�������ُ�ł�");

	std::vector<int> tmp;
	std::random_device rnd;     // �񌈒�I�ȗ���������ŃV�[�h�����@�𐶐�
	std::mt19937 engine(rnd());     //  �����Z���k�c�C�X�^�[��32�r�b�g�ŁA�����͏����V�[�h
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