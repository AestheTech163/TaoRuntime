#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include "common.hpp"


template <typename T>
bool vec_equal(const T* x, const T* y, uint64_t n) {
    bool equal = true;
    for (int i=0; i<n; i++) {
        if (x[i] != y[i]) {
            equal = false;
        }
    }
    return equal;
}


class Timer {
private:
    std::chrono::_V2::system_clock::time_point start_, stop_;
    std::string title_;
public:
    Timer(const std::string& title) {
        title_ = title;
        std::cout << "\n========== Timer for [" << title_ << "], BEGIN ==========" << std::endl;
        start_ = std::chrono::_V2::high_resolution_clock::now();
    }
    ~Timer() {
        stop_ = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_ - start_);
        std::cout << "========== Time cost: " << duration.count()*1.0/1000/1000 << "ms." << std::endl;
        std::cout << "========== Timer for [" << title_ << "], END ==========\n" << std::endl;
    }
};

template <typename T>
void random_init(T* x, uint64_t len, T min=0, T max=1) {
    taoAssert(min < max, "Invalid range for random_init");
    T scale = max - min;
    for (int i=0; i<len; i++) {
        T r = rand() * (T)1.0 / RAND_MAX;
        x[i] = r * scale + min;
    }
}

template <>
void random_init(int* x, uint64_t len, int min, int max) {
    taoAssert(min < max, "Invalid range for random_init");
    for (int i=0; i<len; i++) {
        int r = rand() % (max-min) + min;
        x[i] = r;
    }
}
template <>
void random_init(long* x, uint64_t len, long min, long max) {
    taoAssert(min < max, "Invalid range for random_init");
    for (int i=0; i<len; i++) {
        long r = rand() % (max-min) + min;
        x[i] = r;
    }
}
template <>
void random_init(int16_t* x, uint64_t len, int16_t min, int16_t max) {
    taoAssert(min < max, "Invalid range for random_init");
    for (int i=0; i<len; i++) {
        int16_t r = rand() % (max-min) + min;
        x[i] = r;
    }
}

template <typename T>
void print_vec(const T* vec, int len) {
    std::cout << "[";
    for (int i=0; i<len; i++)
        std::cout << vec[i] << ", ";
    std::cout << "]" << std::endl;
}

template <>
void print_vec(const int8_t* vec, int len) {
    std::cout << "[";
    for (int i=0; i<len; i++)
        std::cout << static_cast<short>(vec[i]) << ", ";
    std::cout << "]" << std::endl;
}