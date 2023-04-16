#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>

#include "lib/utils.h"
#include "lib/macros.cuh"

using std::clog;
using std::endl;
using std::max;

#define input(b, ni, h, w) input[(b)*Ni*NyPAD*NxPAD + (ni)*NyPAD*NxPAD + (h)*NxPAD + (w)]
#define output(b, nn, h, w) output[(b)*Nn*NySCL*NxSCL + (nn)*NySCL*NxSCL + (h)*NxSCL + (w)]
#define weight(nn, ni, p, q) weight[(nn)*Ni*Ky*Kx + (ni)*Ky*Kx + (p)*Kx + (q)]

// Sequential CNN implementation
// input: BatchSize * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
void ConvSequential(const float *input,
    const float *weight,
    float *output) {

  for(int b = 0; b < BatchSize; ++b) {
    for(int nn = 0; nn < Nn; ++nn) {
      for(int ny = 0; ny < Ny; ny += Sy) {
        for(int nx = 0; nx < Nx; nx += Sx) {
          int xout = nx / Sx;
          int yout = ny / Sy;
          float sum = 0.0f;

          for(int ni = 0; ni < Ni; ++ni) {
            for(int ky = 0; ky < Ky; ++ky) {
              for(int kx = 0; kx < Kx; ++kx) {
                sum += weight(nn, ni, ky, kx) * input(b, ni, ny+ky, nx+kx);
              }
            }
          }

          // Perform Relu
          output(b, nn, yout, xout) = max(0.0f, sum);
        }
      }
    }
  }
}

// kernal number that each thread need to deal with
constexpr int KERNAL_COUNT = (Nn / GRIDDIMZ / BLOCKSIZEZ);
constexpr int BLOCK_IN_X = (BLOCKSIZEX + Kx - 1);
constexpr int BLOCK_IN_Y = (BLOCKSIZEY + Ky - 1);
constexpr int KERNEL_SQUARE = Kx * Ky;

// CUDA CONV implementation
// input: BatchSize * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
__global__ void conv_gpu(const float* input, const float* weight, float* output) {
    __shared__ float weight_blocked[BLOCK_CHANNEL][Kx][Ky];
    __shared__ float input_blocked[BLOCK_CHANNEL][BLOCK_IN_X][BLOCK_IN_X];
    float output_thread[KERNAL_COUNT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int kernalOffset = blockIdx.z * KERNAL_COUNT;

    int row = blockIdx.y * BLOCKSIZEY;
    int col = blockIdx.x * BLOCKSIZEX;

    for(int b = 0; b < BatchSize; ++b) {
      // set bias
      #pragma unroll
      for (int k = kernalOffset; k < kernalOffset + KERNAL_COUNT; k++) {
          output_thread[k - kernalOffset] = 0.0f;
      }

      for(int ni = 0; ni < Nn; ni += BLOCK_CHANNEL) {

          /* Step1. load input to shared memory */
          int x, y;
          #pragma unroll
          for(int i = 0; i < BLOCK_CHANNEL; ++i) {
              #pragma unroll
              for (int offset = ty*BLOCKSIZEX + tx; offset < BLOCK_IN_X * BLOCK_IN_Y; offset += BLOCKSIZEX*BLOCKSIZEY) {
                  x = offset % BLOCK_IN_X;
                  y = offset / BLOCK_IN_X;
                  input_blocked[i][y][x] = input(b, ni + i, row + y, col + x);
              }
          }

          for (int k = kernalOffset; k < kernalOffset + KERNAL_COUNT; k++) {
              /* Step2. load weight to shared memory */
              #pragma unroll
              for(int offset = ty*BLOCKSIZEX + tx; offset < KERNEL_SQUARE * BLOCK_CHANNEL; offset += BLOCKSIZEX*BLOCKSIZEY) {
                  int cz = offset / KERNEL_SQUARE;
                  int cz_off = offset % KERNEL_SQUARE;
                  int cx = cz_off % Kx;
                  int cy = cz_off / Kx;
                  weight_blocked[cz][cy][cx] = weight(k, ni + cz, cy, cx);
              }
              __syncthreads();

              /* Step3. Computation */
              for (int kk = 0; kk < BLOCK_CHANNEL; kk++) {
                  for (int ky = 0; ky < Ky; ky++) {
                      for (int kx = 0; kx < Kx; kx++) {
                          output_thread[k - kernalOffset] += input_blocked[kk][ty+ky][tx+kx] * weight_blocked[kk][ky][kx];
                      }
                  }
              }
              __syncthreads();
          }
      }

      // Relu
      for (int k = 0; k < KERNAL_COUNT; k++) {
          output(b, kernalOffset + k, row + ty, col + tx) = max(0.0f, output_thread[k]);
      }
    }
}

// CUDA CONV implementation
// input: BatchSize * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
__global__ void conv_naive(const float* input, const float* weight, float* output) {
    int tx = threadIdx.x + blockIdx.x * BLOCKSIZEX;
    int ty = threadIdx.y + blockIdx.y * BLOCKSIZEY;
    int kernalOffset = blockIdx.z * KERNAL_COUNT;

    for(int b = 0; b < BatchSize; ++b) {
        for (int nn = kernalOffset; nn < kernalOffset + KERNAL_COUNT; nn++) {
          float sum = 0.0f;
          for(int ni = 0; ni < Nn; ni++) {
            for (int ky = 0; ky < Ky; ky++) {
                for (int kx = 0; kx < Kx; kx++) {
                  sum += input(b, ni, ty+ky, tx+kx) * weight(nn, ni, ky, kx);
                }
            }
          }
          output(b, nn, ty, tx) = max(0.0f, sum);
      }
    }
}

int main() {
  const uint64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nn*Nx*Ny*Ni*Kx*Ky;
  auto input_length = BatchSize * Ni * NyPAD * NxPAD; auto input_size = input_length * sizeof(float);
  auto output_length = BatchSize * Nn * NySCL * NxSCL; auto output_size = output_length * sizeof(float);
  auto weight_length = Nn * Ni * Ky * Kx; auto weight_size = weight_length * sizeof(float);
  float* input = static_cast<float*>(malloc(input_size));
  float* output = static_cast<float*>(malloc(output_size));
  float* weight = static_cast<float*>(malloc(weight_size));
  auto sta = std::chrono::steady_clock::now();
  GenerateRandomMatrix(input, input_length);
  GenerateRandomMatrix(weight, weight_length);
  std::chrono::duration<double> rand_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Generate Random Matrix]\tTimeCost:" << rand_duration.count() << "ns" << std::endl;

  sta = std::chrono::steady_clock::now();
  ConvSequential(input, weight, output);
  std::chrono::duration<double> conv_seq_duration = std::chrono::steady_clock::now() - sta;
  print_performance_result(float_calculation_num, conv_seq_duration, "CONV SEQ");

  float* cuda_output = static_cast<float*>(malloc(output_size));
  float* g_input, *g_weight, *g_output;
  cudaMalloc((float**)&g_input, input_size);
  cudaMalloc((float**)&g_weight, weight_size);
  cudaMalloc((float**)&g_output, output_size);
  cudaMemcpy(g_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_weight, weight, weight_size, cudaMemcpyHostToDevice);

  constexpr int GRIDDIMX = (Nx / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Ny / BLOCKSIZEY);
  auto block = dim3(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
  auto grid = dim3(GRIDDIMX, GRIDDIMY, GRIDDIMZ);
  std::clog << "Using thread block dims: " << block.x << ' ' << block.y << ' ' << block.z << '\n';
  std::clog << "Using grid dims: " << grid.x << ' ' << grid.y << ' ' << grid.z << '\n';
  cudaSetDevice(0);

  sta = std::chrono::steady_clock::now();
  conv_gpu<<<grid, block>>>(g_input, g_weight, g_output);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::duration<double> conv_gpu_duration = std::chrono::steady_clock::now() - sta;
  print_performance_result(float_calculation_num, conv_gpu_duration, "CONV CUDA");

  cudaMemcpy(cuda_output, g_output, output_size, cudaMemcpyDeviceToHost);
  if (IsDiffMatrix(cuda_output, output, output_length)) {
    clog << "FAIL" << endl;
  } else {
    clog << "PASS" << endl;
  }

  /******************/
  sta = std::chrono::steady_clock::now();
  conv_naive<<<grid, block>>>(g_input, g_weight, g_output);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::duration<double> conv_naive_duration = std::chrono::steady_clock::now() - sta;
  print_performance_result(float_calculation_num, conv_naive_duration, "CONV NAIVE");

  cudaMemcpy(cuda_output, g_output, output_size, cudaMemcpyDeviceToHost);
  if (IsDiffMatrix(cuda_output, output, output_length)) {
    clog << "FAIL" << endl;
  } else {
    clog << "PASS" << endl;
  }
}
