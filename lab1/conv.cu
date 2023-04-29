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
void conv_seq(const float *input,
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
constexpr int BLOCK_IN_X = (BLOCKSIZEX + Kx - 1);
constexpr int BLOCK_IN_Y = (BLOCKSIZEY + Ky - 1);
constexpr int BLOCK_IN_SQUARE = BLOCK_IN_X * BLOCK_IN_Y;

// CUDA CONV implementation
// input: BatchSize * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
__global__ void conv_naive(const float* input, const float* weight, float* output) {
    int tx = threadIdx.x + blockIdx.x * BLOCKSIZEX;
    int ty = threadIdx.y + blockIdx.y * BLOCKSIZEY;
    int nn = blockIdx.z;

    for(int b = 0; b < BatchSize; ++b) {
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

// CUDA CONV implementation
// input: BatchSize * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
constexpr int KERNEL_SQUARE = Kx * Ky;
__global__ void conv_naive_shared(const float* input, const float* weight, float* output) {
    __shared__ float weight_blocked[BLOCK_CHANNEL][Kx][Ky];
    __shared__ float input_blocked[BLOCK_CHANNEL][BLOCK_IN_X][BLOCK_IN_X];
    int block_size = blockDim.x*blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x;
    int row = blockIdx.y * blockDim.y;
    int nn = blockIdx.z;

    for(int b = 0; b < BatchSize; ++b) {
      float sum = 0.0f;

      for(int ni = 0; ni < Ni; ni += BLOCK_CHANNEL) {
        /* Step1. load input to shared memory */
        for(int offset=ty*blockDim.x+tx; offset < BLOCK_CHANNEL*BLOCK_IN_SQUARE; offset += block_size) {
          int cz = offset / BLOCK_IN_SQUARE;
          int cz_off = offset % BLOCK_IN_SQUARE;
          int cx = cz_off % BLOCK_IN_X;
          int cy = cz_off / BLOCK_IN_X;
          input_blocked[cz][cy][cx] = input(b, ni + cz, row + cy, col + cx);
        }

        /* Step2. load weight to shared memory */
        for(int offset=ty*blockDim.x+tx; offset < BLOCK_CHANNEL*KERNEL_SQUARE; offset += block_size) {
            int cz = offset / KERNEL_SQUARE;
            int cz_off = offset % KERNEL_SQUARE;
            int cx = cz_off % Kx;
            int cy = cz_off / Kx;
            weight_blocked[cz][cy][cx] = weight(nn, ni + cz, cy, cx);
        }
        __syncthreads();

        /* Step3. Computation */
        for (int kk = 0; kk < BLOCK_CHANNEL; kk++) {
            for (int ky = 0; ky < Ky; ky++) {
                for (int kx = 0; kx < Kx; kx++) {
                    sum += input_blocked[kk][ty+ky][tx+kx] * weight_blocked[kk][ky][kx];
                    // sum += input_blocked[kk][ty+ky][tx+kx] * weight(nn, ni + kk, ky, kx);
                }
            }
        }
        __syncthreads();
      }

      // Relu
      output(b, nn, row + ty, col + tx) = max(0.0f, sum);
    }
}

void conv_naive_gridblock(dim3 &grid, dim3 &block) {
  assert(Nx % BLOCKSIZEX == 0 && Ny % BLOCKSIZEY == 0);
  int GRIDDIMX = Nx / BLOCKSIZEX;
  int GRIDDIMY = Ny / BLOCKSIZEY;

  block = dim3(BLOCKSIZEX, BLOCKSIZEY, 1);
  grid = dim3(GRIDDIMX, GRIDDIMY, Nn);
}

#define channel_input(cinput, b, ni, h, w)    cinput[(((b)*NyPAD + (h))*NxPAD + (w))*Ni + (ni)]
#define channel_output(coutput, b, nn, h, w)  coutput[(((b)*NySCL + (h))*NxSCL + (w))*Nn + (nn)]
#define channel_weight(cweight, nn, ni, p, q) cweight[(((p)*Kx + (q))*Ni + (ni))*Nn + (nn)]

// CUDA CONV implementation
// input: BatchSize * NyPAD * NxPAD * Ni
// weight: Ky * Kx * Ni * Nn
// output: BatchSize * NySCL * NxSCL * Nn
__global__ void conv_channel(const float* cinput, const float* cweight, float* coutput) {
    // assume blockDim.x == 1 and blockDim.y == 1
    int tx = blockIdx.z;
    int ty = blockIdx.y;
    int nn = blockIdx.x * blockDim.x + threadIdx.x;

    for(int b = 0; b < BatchSize; ++b) {
      float sum = 0.0f;
      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          for(int ni = 0; ni < Ni; ni++) {
            sum += channel_input(cinput, b, ni, ty+ky, tx+kx) * channel_weight(cweight, nn, ni, ky, kx);
          }
        }
      }
      channel_output(coutput, b, nn, ty, tx) = max(0.0f, sum);
    }
}

__global__ void conv_channel_shared(const float* cinput, const float* cweight, float* coutput) {
    // assume blockDim.z == 1 and blockDim.y == 1
    int tx = blockIdx.z;
    int ty = blockIdx.y;
    int nn = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocksize = blockDim.x;

    __shared__ float tmp_input[Ky][Kx][Ni];

    for(int b = 0; b < BatchSize; ++b) {
      for(int offset=threadIdx.x; offset<Ky*Kx*Ni; offset+=blocksize) {
        int ky = offset / (Kx*Ni);
        int ky_off = offset % (Kx*Ni);
        int kx = ky_off / Ni;
        int ni = ky_off % Ni;
        tmp_input[ky][kx][ni] = channel_input(cinput, b, ni, ty+ky, tx+kx);
      }
      __syncthreads();

      float sum = 0.0f;
      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          for(int ni = 0; ni < Ni; ni++) {
            // sum += tmp_input[ky][kx][ni] * tmp_weight[ky][kx][ni];
            sum += tmp_input[ky][kx][ni] * channel_weight(cweight, nn, ni, ky, kx);
          }
        }
      }
      channel_output(coutput, b, nn, ty, tx) = max(0.0f, sum);
      __syncthreads();
    }
}

void conv_channel_gridblock(dim3 &grid, dim3 &block) {
	assert(Nn % BLOCKSIZEZ == 0);
	int GRIDDIMZ = Nn / BLOCKSIZEZ;

	block = dim3(BLOCKSIZEZ, 1, 1);
	grid = dim3(GRIDDIMZ, Ny, Nx);
}

int main() {
  const uint64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nn*Nx*Ny*Ni*Kx*Ky;
  auto input_length = BatchSize * Ni * NyPAD * NxPAD;
  auto output_length = BatchSize * Nn * NySCL * NxSCL;
  auto weight_length = Nn * Ni * Ky * Kx;
  auto conv_test = Test<float, decltype(conv_seq)>(input_length, output_length, weight_length, float_calculation_num, "CONV ");
  conv_test.run_seq(conv_seq);
  // conv_test.test_cuda(conv_block, conv_block_gridblock, "CUDA SHARED");
  conv_test.test_cuda(conv_naive, conv_naive_gridblock, "CUDA NAIVE");
  conv_test.test_cuda(conv_naive_shared, conv_naive_gridblock, "CUDA SHARED");

  auto reformat_input = [](const float* input) {
    float* cinput = static_cast<float*>(malloc(BatchSize * Ni * NyPAD * NxPAD * sizeof(float)));
    for (int b = 0; b < BatchSize; b ++)
      for (int ni = 0; ni < Ni; ni ++)
        for (int h = 0; h < NyPAD; h ++)
          for (int w = 0; w < NxPAD; w ++)
            channel_input(cinput, b, ni, h, w) = input(b, ni, h, w);
    return cinput;
  };
  auto reformat_weight = [](const float* weight) {
    float* cweight = static_cast<float*>(malloc(Nn * Ni * Ky * Kx * sizeof(float)));
    for (int nn = 0; nn < Nn; nn ++)
      for (int ni = 0; ni < Ni; ni ++)
        for (int p = 0; p < Ky; p ++)
          for (int q = 0; q < Kx; q ++)
            channel_weight(cweight, nn, ni, p, q) = weight(nn, ni, p, q);
    return cweight;
  };
  auto reformat_output = [](float* coutput) {
    float* output = static_cast<float*>(malloc(BatchSize * Nn * NySCL * NxSCL * sizeof(float)));
    for (int b = 0; b < BatchSize; b ++)
      for (int nn = 0; nn < Nn; nn ++)
        for (int h = 0; h < NySCL; h ++)
          for (int w = 0; w < NxSCL; w ++)
            output(b, nn, h, w) = channel_output(coutput, b, nn, h, w);
    return output;
  };
  conv_test.test_cuda(conv_channel, conv_channel_gridblock, "CUDA CHANNEL", 
    reformat_input, reformat_weight, reformat_output);
  conv_test.test_cuda(conv_channel_shared, conv_channel_gridblock, "CUDA CHANNEL SHARED", 
    reformat_input, reformat_weight, reformat_output);
}
