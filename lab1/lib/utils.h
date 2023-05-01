#pragma once
#include <string>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;


template<class T>
T MyRand(const T & min, const T & max) {
    static thread_local std::mt19937 generator;
    std::uniform_real_distribution<T> distribution(min,max);
    return distribution(generator);
}

template<class T>
void GenerateRandomMatrix(T* input, size_t input_length) {
  for(int i = 0; i < input_length; ++i) {
    input[i] = MyRand<T>(-1.0f, 1.0f);
  }
}

#define CUDA_CHECK(err) cuda_check((err), __FILE__, __LINE__);
inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        string(file) + ":" + std::to_string(line) + ": " + string(cudaGetErrorString(err)));
  }
}
