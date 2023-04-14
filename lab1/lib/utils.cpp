#include <iostream>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>
#include "utils.h"

using std::clog;
using std::endl;

bool IsDiffSingle(float a, float b) {
  return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

bool IsDiffMatrix(float* a, float* b, size_t msize) {
  for(int i = 0; i < msize; ++i) {
    if (IsDiffSingle(a[i], b[i])) {
      clog << "Find different is pos " << i << ": Expected: " << b[i] << ", Get: " << a[i] << endl;
      return true;
    }
  }
  return false;
}

void GenerateRandomMatrix(float* input, size_t input_size) {
  for(int i = 0; i < input_size; ++i) {
    input[i] = MyRand<float>(-1.0f, 1.0f);
  }
}

void print_performance_result(const uint64_t float_calculation_num, 
  const duration<double>& duration, const string& name) {
  uint64_t run_time_us = duration_cast<microseconds>(duration).count();
  float gflops = float_calculation_num /(run_time_us * 1e3);
  clog << "[" << name << "]\tTimeCost:" << duration.count() << "ns" << std::endl;
  clog << "[" << name << "]\tGFLOPS:" << gflops << "gflops" << std::endl;
}