#include <cmath>

#include <chrono>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cnn.h"
#include "macros.cuh"

using std::clog;
using std::endl;
// using std::isfinite;
using std::max;
using std::string;

// Sequential CNN implementation
void CnnSequential(const float *input,
    const float *weight, const float* bias,
    float *output) {

  // Allocate memory on heap to avoid stack overflow.
  static float C[kNum][kImSize][kImSize];

  // Bias
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        C[i][h][w] = bias[i];
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) {
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q)
              C[i][h][w] += weight(i, j, p, q) * input(j, h + p, w + q);
          }
        }
      }
    }
  }

  // ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[i][h][w] = max(0.f, C[i][h][w]);
      }
    }
  }

  // Max pooling
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output(i, h, w) = max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
      }
    }
  }
}

void LoadData(const string& data_dir, float* input,
    float* weight, float* bias) {
  const char kInputFile[] = "lib/testdata/input.bin";
  const char kWeightFile[] = "lib/testdata/weight.bin";
  const char kBiasFile[] = "lib/testdata/bias.bin";

  // sizes are known ahead of time for this particular example
  auto input_size = kNum * kInImSize * kInImSize * sizeof(float);
  auto weight_size = kNum * kNum * kKernel * kKernel * sizeof(float);
  auto bias_size = kNum * sizeof(float);

  int input_fd = open((data_dir + kInputFile).c_str(), O_RDONLY);
  int weight_fd = open((data_dir + kWeightFile).c_str(), O_RDONLY);
  int bias_fd = open((data_dir + kBiasFile).c_str(), O_RDONLY);

  if (input_fd == -1) {
    clog << "Cannot find " << kInputFile << endl;
    exit(EXIT_FAILURE);
  }
  if (weight_fd == -1) {
    clog << "Cannot find " << kWeightFile << endl;
    exit(EXIT_FAILURE);
  }
  if (bias_fd == -1) {
    clog << "Cannot find " << kBiasFile << endl;
    exit(EXIT_FAILURE);
  }

  auto input_in = reinterpret_cast<float*>(mmap(
        nullptr, input_size, PROT_READ, MAP_SHARED, input_fd, 0));
  if (input_in == MAP_FAILED) {
    clog << "Incomplete " << kInputFile << endl;
    close(input_fd);
    exit(EXIT_FAILURE);
  }

  auto weight_in = reinterpret_cast<float*>(mmap(
        nullptr, weight_size, PROT_READ, MAP_SHARED, weight_fd, 0));
  if (weight_in == MAP_FAILED) {
    clog << "Incomplete " << kWeightFile << endl;
    close(weight_fd);
    exit(EXIT_FAILURE);
  }

  float* bias_in = reinterpret_cast<float*>(mmap(
        nullptr, bias_size, PROT_READ, MAP_SHARED, bias_fd, 0));
  if (bias_in == MAP_FAILED) {
    clog << "Incomplete " << kBiasFile << endl;
    close(bias_fd);
    exit(EXIT_FAILURE);
  }

  memcpy(input, input_in, input_size);
  memcpy(weight, weight_in, weight_size);
  memcpy(bias, bias_in, bias_size);
  munmap(input_in, input_size);
  munmap(weight_in, weight_size);
  munmap(bias_in, bias_size);
  close(input_fd);
  close(weight_fd);
  close(bias_fd);
}

float IsError(float a, float b) {
  return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

int Verify(const string& data_dir,
    const float* output, int output_size) {
  int error = 0;
  const char kOutputFile[] = "lib/testdata/output.bin";
  int fd = open((data_dir + kOutputFile).c_str(), O_RDONLY);
  if (fd == -1) {
    clog << "Cannot find " << kOutputFile << endl;
    return EXIT_FAILURE;
  }
  auto ground_truth = reinterpret_cast<float(*)[kOutImSize][kOutImSize]>(mmap(
        nullptr, output_size, PROT_READ, MAP_SHARED, fd, 0));
  if (ground_truth == MAP_FAILED) {
    clog << "Incomplete " << kOutputFile << endl;
    close(fd);
    return EXIT_FAILURE;
  }
  bool first = true;
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        if (IsError(output(i, h, w), ground_truth[i][h][w])) {
          if (first) {
            clog << "First error: got " << output(i, h, w) << ", while expecting "
              << ground_truth[i][h][w] << " @ i = " << i << ", h = " << h
              << ", w = " << w << endl;
            first = false;
          }
          ++error;
        }
      }
    }
  }
  munmap(ground_truth, output_size);
  close(fd);
  return error;
}
