#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>

#include "lib/test.hpp"
#include "lib/macros.cuh"
#include "kernels/gemm_seq.hpp"
#include "kernels/gemm_kernels.hpp"

using std::max;

int main() {
  CudaDeviceInfo();
  const int64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nn*Ni;
  auto input_length = BatchSize*Ni;
  auto output_length = BatchSize*Nn;
  auto weight_length = Ni * Nn;
  auto gemm_test = Test<float, decltype(gemm_naive<BatchSize, Ni, Nn, BatchSize, 16>)>(input_length, output_length, weight_length, float_calculation_num, "GEMM ", 10);
  gemm_test.run_seq(gemm_seq<BatchSize, Ni, Nn>);
  gemm_test.test_cuda(gemm_naive<BatchSize, Ni, Nn, BatchSize, 16>, "CUDA NAIVE");
  gemm_test.test_cuda(gemm_coalescing<BatchSize, Ni, Nn, BatchSize, 16>, "CUDA coalescing");
  gemm_test.test_cuda(gemm_naive_shared<BatchSize, Ni, Nn, BatchSize, 64, 512>, "CUDA NAIVE SHARED");
  gemm_test.test_cuda(gemm_shared<BatchSize, Ni, Nn, BatchSize, 64, 64>, "CUDA SHARED");
  gemm_test.test_cuda(gemm_block_tiling<BatchSize, Ni, Nn, BatchSize, 64, 64, 2, 2>, "CUDA TILING");
  gemm_test.test_cuda(gemm_vectorize<BatchSize, Ni, Nn, BatchSize, 64, 64, 2, 2>, "CUDA VECTERIZE");
}
