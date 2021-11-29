#pragma once
#include <chrono>

class Stopwatch {
public:
    Stopwatch() : last_time_(std::chrono::high_resolution_clock::now()) {}

    double elapsed() {
        std::chrono::time_point<std::chrono::high_resolution_clock> this_time = std::chrono::high_resolution_clock::now();
        double delta_time = std::chrono::duration<double>(this_time - last_time_).count();
        last_time_ = this_time;
        return delta_time;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time_;
};

