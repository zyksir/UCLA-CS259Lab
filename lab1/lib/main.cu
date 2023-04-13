#include <chrono>
#include <iostream>
#include <string>
#include "cnn.h"

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::clog;
using std::string;

int main(int argc, char** argv) {

  // sizes are known ahead of time for this particular example
  auto input_size = kNum * kInImSize * kInImSize * sizeof(float);
  auto weight_size = kNum * kNum * kKernel * kKernel * sizeof(float);
  auto bias_size = kNum * sizeof(float);
  auto output_size = kNum * kOutImSize * kOutImSize * sizeof(float);

  // Allocate memory on heap to avoid stack overflow
  float* input = static_cast<float*>(malloc(input_size));
  float* weight = static_cast<float*>(malloc(weight_size));
  float* bias = static_cast<float*>(malloc(bias_size));
  float* output = static_cast<float*>(malloc(output_size));

  if (argc > 2) {
    clog << "Usage: " << argv[0] << " [data dir]\n";
    return EXIT_FAILURE;
  }

  const string data_dir = argc == 2 ? string(argv[1]) + "/" : "";
  LoadData(data_dir, input, weight, bias);

  clog << "Create device memory\n";
  float* g_input, *g_weight, *g_bias, *g_output;
  cudaMalloc((float**)&g_input, input_size);
  cudaMalloc((float**)&g_weight, weight_size);
  cudaMalloc((float**)&g_bias, bias_size);
  cudaMalloc((float**)&g_output, output_size);

  clog << "Transfer to global memory\n";
  cudaMemcpy(g_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_weight, weight, weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_bias, bias, bias_size, cudaMemcpyHostToDevice);

  clog << "Invoke CNN computation kernel\n";
  if (getenv("SEQUENTIAL")) {
    const auto begin = steady_clock::now();
    CnnSequential(input, weight, bias, output);
    const auto end = steady_clock::now();

    uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
    float gflops = float(kNum) * kNum * kImSize * kImSize * kKernel * kKernel * 2
      / (run_time_us * 1e3);
    clog << "Time: " << run_time_us * 1e-6 << " s\n";
    clog << "Perf: " << gflops << " GFlops\n";
  } else {
    kernel_wrapper(g_input, g_weight, g_bias, g_output);
    cudaMemcpy(output, g_output, output_size, cudaMemcpyDeviceToHost);
  }

  int error = Verify(data_dir, output, output_size);
  if (error != 0) {
    clog << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
    clog << "FAIL\n";
    return EXIT_FAILURE;
  } else {
    clog << "PASS\n";
    return EXIT_SUCCESS;
  }
}
