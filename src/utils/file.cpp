#include "utils/file.h"

bool isFileExist(const std::string &file_path) {
    std::ifstream f(file_path.c_str());
    return f.good();
}

bool isDirExist(const std::string &dir_path) {
    if (access(dir_path.c_str(), 0) == 0) {
        return true;
    } else {
        return false;
    }
}

bool createDir(const std::string &dir_path) {
    if (isDirExist(dir_path)) {
        return true;
    }
    if (mkdir(dir_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        return true;
    } else {
        return false;
    }
}

bool removeFile(const std::string &file_path) {
    if (isFileExist(file_path)) {
        if (remove(file_path.c_str()) == 0) {
            return true;
        } else {
            return false;
        }
    } else {
        return true;
    }
}

bool removeDir(const std::string &dir_path) {
    if (isDirExist(dir_path)) {
        if (rmdir(dir_path.c_str()) == 0) {
            return true;
        } else {
            return false;
        }
    } else {
        return true;
    }
}

bool checkSuffix(const std::string &file_path, const std::string &suffix) {
    if (file_path.length() >= suffix.length()) {
        return (0 == file_path.compare(file_path.length() - suffix.length(), suffix.length(), suffix));
    } else {
        return false;
    }
}

bool loadJson(const std::string &file_path, nlohmann::json &json) {
    if (!isFileExist(file_path)) {
        std::cout<<fmt::format("[Error] file: {}{}{} not exists",ansi_colors["red"],file_path,ansi_colors["reset"])<<std::endl;
        return false;
    }
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        return false;
    }
    try {
        ifs >> json;
    } catch (const std::exception &e) {
        ifs.close();
        return false;
    }
    ifs.close();
    return true;
}

bool ParseArg(int argc, char* argv[],nlohmann::json &json){
    std::string file_path;
#ifdef CONFIGFILE
    file_path = CONFIGFILE;
#endif
    if(argc >= 2){
        file_path = argv[1];
    }
    return loadJson(file_path,json);
}