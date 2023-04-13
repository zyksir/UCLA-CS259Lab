#include <cmath>

#include <chrono>
#include <iostream>
#include <string>
#include <random>

// #include <fcntl.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>

#include "conv_seq.h"
#include "macros.cuh"

#include "conv_gpu.h"

using std::clog;
using std::endl;
using std::max;

// Sequential CNN implementation
// input: Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: Nn * NySCL * NxSCL
void ConvSequential(const float *input,
    const float *weight,
    float *output) {

  for(int nn = 0; nn < Nn; ++nn) {
    for(int ny = 0; ny < Ny; ny += Sy) {
      for(int nx = 0; nx < Nx; nx += Sx) {
        int xout = nx / Sx;
        int yout = ny / Sy;
        float sum = 0.0f;

        for(int ni = 0; ni < Ni; ++ni) {
          for(int ky = 0; ky < Ky; ++ky) {
            for(int kx = 0; kx < Kx; ++kx) {
              sum += weight(nn, ni, ky, kx) * input(ni, ny+ky, nx+kx);
            }
          }
        }

        // Perform Relu
        output(nn, yout, xout) = max(0.0f, sum);
      }
    }
  }
}

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

template<class T>
T MyRand(const T & min, const T & max) {
    static thread_local std::mt19937 generator;
    std::uniform_real_distribution<T> distribution(min,max);
    return distribution(generator);
}

void GenerateRandomMatrix(float* input, size_t input_size) {
  for(int i = 0; i < input_size; ++i) {
    input[i] = MyRand<float>(-1.0f, 1.0f);
  }
}

int main() {
  auto input_size = Ni * NyPAD * NxPAD;
  auto output_size = Nn * NySCL * NxSCL;
  auto weight_size = Nn * Ni * Ky * Kx;
  float* input = static_cast<float*>(malloc(input_size * sizeof(float)));
  float* output = static_cast<float*>(malloc(output_size * sizeof(float)));
  float* weight = static_cast<float*>(malloc(weight_size * sizeof(float)));
  auto sta = std::chrono::steady_clock::now();
  GenerateRandomMatrix(input, input_size);
  GenerateRandomMatrix(weight, weight_size);
  std::chrono::duration<double> rand_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Generate Random Matrix]\tTimeCost:" << rand_duration.count() << "ns" << std::endl;

  sta = std::chrono::steady_clock::now();
  ConvSequential(input, weight, output);
  std::chrono::duration<double> conv_seq_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Conv Sequence]\tTimeCost:" << conv_seq_duration.count() << "ns" << std::endl;

  float* cuda_output = static_cast<float*>(malloc(output_size * sizeof(float)));
  float* g_input, *g_weight, *g_output;
  cudaMalloc((float**)&g_input, input_size * sizeof(float));
  cudaMalloc((float**)&g_weight, weight_size * sizeof(float));
  cudaMalloc((float**)&g_output, output_size * sizeof(float));
  cudaMemcpy(g_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(g_weight, weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);

  auto block = dim3(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
  auto grid = dim3(GRIDDIMX, GRIDDIMY, GRIDDIMZ);
  std::clog << "Using thread block dims: " << block.x << ' ' << block.y << ' ' << block.z << '\n';
  std::clog << "Using grid dims: " << grid.x << ' ' << grid.y << ' ' << grid.z << '\n';
  cudaSetDevice(0);

  sta = std::chrono::steady_clock::now();
  conv_gpu<<<grid, block>>>(g_input, g_weight, g_output);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::duration<double> conv_gpu_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Conv CUDA]\tTimeCost:" << conv_gpu_duration.count() << "ns" << std::endl;

  cudaMemcpy(cuda_output, g_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
  if (IsDiffMatrix(cuda_output, output, output_size)) {
    clog << "FAIL" << endl;
  } else {
    clog << "PASS" << endl;
  }
}

// int Verify(const string& data_dir,
//     const float* output, int output_size) {
//   int error = 0;
//   const char kOutputFile[] = "lib/testdata/output.bin";
//   int fd = open((data_dir + kOutputFile).c_str(), O_RDONLY);
//   if (fd == -1) {
//     clog << "Cannot find " << kOutputFile << endl;
//     return EXIT_FAILURE;
//   }
//   auto ground_truth = reinterpret_cast<float(*)[kOutImSize][kOutImSize]>(mmap(
//         nullptr, output_size, PROT_READ, MAP_SHARED, fd, 0));
//   if (ground_truth == MAP_FAILED) {
//     clog << "Incomplete " << kOutputFile << endl;
//     close(fd);
//     return EXIT_FAILURE;
//   }
//   bool first = true;
//   for (int i = 0; i < kNum; ++i) {
//     for (int h = 0; h < kOutImSize; ++h) {
//       for (int w = 0; w < kOutImSize; ++w) {
//         if (IsError(output(i, h, w), ground_truth[i][h][w])) {
//           if (first) {
//             clog << "First error: got " << output(i, h, w) << ", while expecting "
//               << ground_truth[i][h][w] << " @ i = " << i << ", h = " << h
//               << ", w = " << w << endl;
//             first = false;
//           }
//           ++error;
//         }
//       }
//     }
//   }
//   munmap(ground_truth, output_size);
//   close(fd);
//   return error;
// }
