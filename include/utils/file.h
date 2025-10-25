#pragma once

#include "common.h"

bool isFileExist(const std::string &file_path);

bool isDirExist(const std::string &dir_path);

bool createDir(const std::string &dir_path);

bool removeFile(const std::string &file_path);

bool removeDir(const std::string &dir_path);

bool checkSuffix(const std::string &file_path, const std::string &suffix);

bool loadJson(const std::string &file_path, nlohmann::json &json);

bool ParseArg(int argc, char* argv[],nlohmann::json &json);

