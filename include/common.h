#pragma once

#include <iostream>
#include <stdint.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <cerrno>

#include <vector>
#include <map>
#include <tuple>
#include <utility>
#include <set>
#include <variant>
#include <unordered_set>
#include <string>

#include <algorithm>

#include <iomanip>
#include <fstream>

#include <time.h>
#include <chrono>

#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/stat.h>

#include <memory>
#include <atomic>
#include <thread>
#include <pthread.h>
#include <mutex>
#include <future>

#include <functional>

#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "fmt/core.h"

#include "nlohmann/json.hpp"

#include "ThreadPool/ThreadPool.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

inline std::map<std::string, std::string> ansi_colors = {
    {"black", "\033[30m"},
    {"red", "\033[31m"},
    {"green", "\033[32m"},
    {"yellow", "\033[33m"},
    {"blue", "\033[34m"},
    {"magenta", "\033[35m"},
    {"cyan", "\033[36m"},
    {"white", "\033[37m"},

    {"reset", "\033[0m"}
};

template <typename Func, typename... Args>
inline int RecordTimeCost(Func &&func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<Func>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return static_cast<int>(duration.count());
}