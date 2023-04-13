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

#include "cnn.h"
#include "../kernel.h"
#include "macros.cuh"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

dim3 get_param(std::string setting) {
  if (auto var = getenv(setting.c_str())) {
    std::istringstream iss{std::string(var)};
    std::vector<std::string> nums(std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>());
    switch (nums.size()) {
      case 3:
        return dim3(std::stoul(nums[0]), std::stoul(nums[1]), std::stoul(nums[2]));
      default:
        throw std::runtime_error("invalid params.sh settings: " + std::string(var));
    }
  }
  throw std::runtime_error("invalid setting: " + std::string(setting));
}

void kernel_wrapper(float* input, float* weight, float* bias, float* output) {
  // auto block = get_param("BLOCK");
  // auto grid = get_param("GRID");
  auto block = dim3(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
  auto grid = dim3(GRIDDIMX, GRIDDIMY, GRIDDIMZ);
  std::clog << "Using thread block dims: " << block.x << ' ' << block.y << ' ' << block.z << '\n';
  std::clog << "Using grid dims: " << grid.x << ' ' << grid.y << ' ' << grid.z << '\n';

  cudaSetDevice(0); // only 1 GPU

  // Execute kernel.
  const auto begin = steady_clock::now();
  cnn_gpu<<<grid, block>>>(input, weight, bias, output);
  CUDA_CHECK(cudaDeviceSynchronize()); // wait until kernel is completely finished
  const auto end = steady_clock::now();

  uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  float gflops = float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2
    / (run_time_us * 1e3);
  std::clog << "Time: " << run_time_us * 1e-6 << " s\n";
  std::clog << "Perf: " << gflops << " GFlops\n";
}
