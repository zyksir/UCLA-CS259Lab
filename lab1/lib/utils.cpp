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
