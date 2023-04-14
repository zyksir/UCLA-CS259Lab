#pragma once
#include <string>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::endl;
using std::string;

bool IsDiffSingle(float a, float b);
bool IsDiffMatrix(float* a, float* b, size_t msize);

template<class T>
T MyRand(const T & min, const T & max) {
    static thread_local std::mt19937 generator;
    std::uniform_real_distribution<T> distribution(min,max);
    return distribution(generator);
}
void GenerateRandomMatrix(float* input, size_t input_size);

#define CUDA_CHECK(err) cuda_check((err), __FILE__, __LINE__);
inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        string(file) + ":" + std::to_string(line) + ": " + string(cudaGetErrorString(err)));
  }
}

void print_performance_result(const uint64_t float_calculation_num, 
  const duration<double>& duration, const string& name);