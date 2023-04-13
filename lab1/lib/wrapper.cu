/******************************************************************************
 *                                                                            *
 *  In this file the OpenCL C++ APIs are used instead of the C APIs taught    *
 *  in class. Please refer to                                                 *
 *  https://github.khronos.org/OpenCL-CLHPP/namespacecl.html                  *
 *  or                                                                        *
 *  https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf    *
 *  for more information about how the C++ APIs wrap the C APIs.              *
 *                                                                            *
 ******************************************************************************/

#include <cstdlib>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iterator>

#include "conv_seq.h"
#include "macros.cuh"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

void cnn_wrapper(float* input, float* weight, float* bias, float* output) {
  auto block = dim3(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
  auto grid = dim3(GRIDDIMX, GRIDDIMY, GRIDDIMZ);
  std::clog << "Using thread block dims: " << block.x << ' ' << block.y << ' ' << block.z << '\n';
  std::clog << "Using grid dims: " << grid.x << ' ' << grid.y << ' ' << grid.z << '\n';

  cudaSetDevice(0); // only 1 GPU

  // Execute cnn.
  const auto begin = steady_clock::now();
  cnn_gpu<<<grid, block>>>(input, weight, bias, output);
  CUDA_CHECK(cudaDeviceSynchronize()); // wait until cnn is completely finished
  const auto end = steady_clock::now();

  uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  float gflops = float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2
    / (run_time_us * 1e3);
  std::clog << "Time: " << run_time_us * 1e-6 << " s\n";
  std::clog << "Perf: " << gflops << " GFlops\n";
}
