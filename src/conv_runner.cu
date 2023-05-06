#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>

#include "lib/test.hpp"
#include "lib/macros.cuh"
#include "kernels/conv_seq.hpp"
#include "kernels/conv_kernels.hpp"

int main() {
  const uint num_repeats = 1;
  CudaDeviceInfo();
  const uint64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nnn*Nxx*Nyy*Nii*Kxx*Kyy;
  constexpr int NyPAD = Nyy + Kyy - 1;
  constexpr int NxPAD = Nxx + Kxx - 1;
  constexpr int NySCL = Nyy/Syy;
  constexpr int NxSCL = Nxx/Sxx;
  auto input_length = BatchSize * Nii * NyPAD * NxPAD;
  auto output_length = BatchSize * Nnn * NySCL * NxSCL;
  auto weight_length = Nnn * Nii * Kyy * Kxx;
  auto conv_test = Test<float, decltype(conv_naive<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy, 16, 16>)>
    (input_length, output_length, weight_length, float_calculation_num, "CONV ", num_repeats);
  conv_test.run_seq(conv_seq<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy>);
  constexpr uint BX = 32 > Nxx ? Nxx : 32;
  conv_test.test_cuda(conv_naive<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy, BX, BX>, "CUDA NAIVE ");
  auto reformat_input = [](const float* input) {
    float* cinput = static_cast<float*>(malloc(BatchSize * Nii * NyPAD * NxPAD * sizeof(float)));
    for (int b = 0; b < BatchSize; b ++)
      for (int ni = 0; ni < Nii; ni ++)
        for (int y = 0; y < NyPAD; y ++)
          for (int x = 0; x < NxPAD; x ++)
            Val4D(cinput, b, y, x, ni, NyPAD, NxPAD, Nii) = Val4D(input, b, ni, y, x, Nii, NyPAD, NxPAD);
            // channel_input(cinput, b, ni, h, w) = input(b, ni, h, w);
    return cinput;
  };
  auto reformat_weight = [](const float* weight) {
    float* cweight = static_cast<float*>(malloc(Nnn * Nii * Kyy * Kxx * sizeof(float)));
    for (int nn = 0; nn < Nnn; nn ++)
      for (int ni = 0; ni < Nii; ni ++)
        for (int p = 0; p < Kyy; p ++)
          for (int q = 0; q < Kxx; q ++)
            Val4D(cweight, p, q, ni, nn, Kxx, Nii, Nnn) = Val4D(weight, nn, ni, p, q, Nii, Kyy, Kxx);
            // channel_weight(cweight, nn, ni, p, q) = weight(nn, ni, p, q);
    return cweight;
  };
  auto reformat_output = [](float* coutput) {
    float* output = static_cast<float*>(malloc(BatchSize * Nnn * NySCL * NxSCL * sizeof(float)));
    for (int b = 0; b < BatchSize; b ++)
      for (int nn = 0; nn < Nnn; nn ++)
        for (int y = 0; y < NySCL; y ++)
          for (int x = 0; x < NxSCL; x ++)
            Val4D(output, b, nn, y, x, Nnn, NySCL, NxSCL) = Val4D(coutput, b, y, x, nn, NySCL, NxSCL, Nnn);
            // output(b, nn, h, w) = channel_output(coutput, b, nn, h, w);
    return output;
  };
  conv_test.test_cuda(conv_coalescing<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy, 64>, "CUDA COALESCING ",
    reformat_input, reformat_weight, reformat_output);
  conv_test.test_cuda(conv_block_tiling<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy, 64, 14, 14>, "CUDA BLOCK TILING ",
    reformat_input, reformat_weight, reformat_output);
  conv_test.test_cuda(conv_shared<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy, 64, 14, 14, 16>, "CUDA SHARED ",
    reformat_input, reformat_weight, reformat_output);
  conv_test.test_cuda(conv_vectorize<BatchSize, Nii, Nnn, Nxx, Nyy, Kxx, Kyy, 64, 14, 14, 16>, "CUDA VECTORIZE ",
    reformat_input, reformat_weight, reformat_output);
}
