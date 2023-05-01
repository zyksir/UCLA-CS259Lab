#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>

#include "lib/test.hpp"
#include "lib/macros.cuh"

using std::max;

#define input(b, ni) input[(b)*Ni + (ni)]
#define weight(ni, nn) weight[(ni)*Nn + (nn)]
#define output(b, nn) output[(b)*Nn + (nn)]

static_assert(BatchSize >= BLOCKSIZEX, "BatchSize should be greater than BLOCKSIZEX");

// Sequential GEMM implementation
// input: BatchSize * Ni
// weight: Ni * Nn
// output: BatchSize * Nn
void gemm_seq(const float *input, const float *weight, float *output) {
  for(int b = 0; b < BatchSize; ++b) {
    for(int ni = 0; ni < Ni; ++ni) {
        float input_val = input(b, ni);
        for(int nn = 0; nn < Nn; ++nn) {
            output(b, nn) += input_val * weight(ni, nn);
        }
    }
  }
}

// input: BatchSize * Ni
// weight: Ni * Nn
// output: BatchSize * Nn
__global__ void gemm_naive(const float *input, const float *weight, float *output) {
 
  int b = threadIdx.x + blockIdx.x * BLOCKSIZEX;
  int nn = threadIdx.y + blockIdx.y * BLOCKSIZEY;
	float sum = 0;
	for(int ni = 0; ni < Ni; ++ni) {
		sum += input(b, ni) * weight(ni, nn);
	}
  output(b, nn) = sum;
}

// input: BatchSize * Ni  -> X * Z
// weight: Ni * Nn        -> Z * Y
// output: BatchSize * Nn -> X * Y
__global__ void gemm_naive_shared(const float *input, const float *weight, float *output) {
  __shared__ float input_blocked[BLOCKSIZEX*BLOCKSIZEZ];
	int x = blockIdx.x * BLOCKSIZEX + threadIdx.x;
	int y = blockIdx.y * BLOCKSIZEY + threadIdx.y;

  float sum = 0;
  for(int ni = 0; ni < Ni; ni += BLOCKSIZEZ) {
    for(int z = threadIdx.y; z < BLOCKSIZEZ; z+=blockDim.y)
      input_blocked[threadIdx.x + z*BLOCKSIZEX] = input(x, z+ni);
    __syncthreads();

    for(int z = 0; z < BLOCKSIZEZ; ++z)
      sum += input_blocked[threadIdx.x + z*BLOCKSIZEX] * weight(z+ni, y);
    __syncthreads();
  }
  output(x, y) = sum;
}

void gemm_naive_gridblock(dim3 &grid, dim3 &block) {
  assert(BatchSize % BLOCKSIZEX == 0 && Nn % BLOCKSIZEY == 0);
  constexpr int GRIDDIMX = (BatchSize / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Nn / BLOCKSIZEY);
  block = dim3(BLOCKSIZEX, BLOCKSIZEY, 1);
  grid = dim3(GRIDDIMX, GRIDDIMY, 1);
}

// 由于 threadIdx.x 相邻的 thread 会放在一个 wrap 里执行
// 我们尽量让他们去访问相邻的数据。
// 优化后，这里一个 wrap 的 thread 会访问一个 input，并访问相邻的 weight
__global__ void gemm_coalescing(const float *input, const float *weight, float *output) {
  int x = threadIdx.x / BLOCKSIZEY + blockIdx.x * BLOCKSIZEX;
  int y = threadIdx.x % BLOCKSIZEY + blockIdx.y * BLOCKSIZEY;
	float sum = 0;
	for(int ni = 0; ni < Ni; ++ni) {
		sum += input(x, ni) * weight(ni, y);
	}
  output(x, y) = sum;
}

void gemm_coalescing_gridblock(dim3 &grid, dim3 &block) {
  assert(BatchSize % BLOCKSIZEX == 0 && Nn % BLOCKSIZEY == 0);
  constexpr int GRIDDIMX = (BatchSize / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Nn / BLOCKSIZEY);
  block = dim3(BLOCKSIZEX*BLOCKSIZEY, 1);
  grid = dim3(GRIDDIMX, GRIDDIMY);
}

// 一个 block 能用的 shared_memory 的最大是 48KB = 1024 * 12 Float
// 16*64 + 64*64
// TODO: see sample of instructions
// TODO: see profiler’s sampling of warp states
constexpr int blocksizez = 64;
__global__ void gemm_shared(const float *input, const float *weight, float *output) {
  __shared__ float input_blocked[BLOCKSIZEX*blocksizez];
  __shared__ float weight_blocked[blocksizez*BLOCKSIZEY];
  int tx = threadIdx.x / BLOCKSIZEY;
  int ty = threadIdx.x % BLOCKSIZEY;
	int x = blockIdx.x * BLOCKSIZEX + tx;
	int y = blockIdx.y * BLOCKSIZEY + ty;

  float sum = 0;
  for(int ni = 0; ni < Ni; ni += blocksizez) {
    for(int z = ty; z < blocksizez; z+=BLOCKSIZEY)
      input_blocked[tx*blocksizez + z] = input(x, z+ni);
    for(int z = tx; z < blocksizez; z+=BLOCKSIZEX)
      weight_blocked[z*BLOCKSIZEY + ty] = weight(z+ni, y);
    __syncthreads();

    for(int z = 0; z < blocksizez; ++z) {
      sum += input_blocked[tx*blocksizez + z] * weight_blocked[z*BLOCKSIZEY + ty];
    }
    __syncthreads();
  }
  output(x, y) = sum;
}

__global__ void gemm_blocktiling(const float *input, const float *weight, float *output) {
  __shared__ float input_blocked[BLOCKSIZEX][blocksizez];
  __shared__ float weight_blocked[blocksizez][BLOCKSIZEY];
  int tx = threadIdx.x / (BLOCKSIZEY/TILEY);
  int ty = threadIdx.x % (BLOCKSIZEY/TILEY);
  const int thread_per_x = BLOCKSIZEX/TILEX;
  const int thread_per_y = BLOCKSIZEY/TILEY;
	const int bx_offset = blockIdx.x * BLOCKSIZEX;
	const int by_offset = blockIdx.y * BLOCKSIZEY;
  const int tx_offset = tx*TILEX;
	const int ty_offset = ty*TILEY;
  float tmp_z[TILEX][TILEY] = {0.0f};
  for(int bz = 0; bz < Ni; bz += blocksizez) {
    for(int z = ty; z < blocksizez; z+=thread_per_y) {
      for(int tilex = 0; tilex < TILEX; tilex++) {
        input_blocked[tx_offset+tilex][z] = input(bx_offset+tx_offset+tilex, bz+z);
      }
    }

    for(int z = tx; z < blocksizez; z+=thread_per_x) {
      for(int tiley = 0; tiley < TILEY; tiley++) {
        weight_blocked[z][ty_offset+tiley] = weight(bz+z, by_offset+ty_offset+tiley);
      }
    }
    __syncthreads();
    for(int z = 0; z < blocksizez; ++z) {
      for(int tilex = 0; tilex < TILEX; ++tilex) {
        for(int tiley = 0; tiley < TILEY; ++tiley) {
          tmp_z[tilex][tiley] += input_blocked[tx_offset+tilex][z] * weight_blocked[z][ty_offset+tiley];
        }
      }
    }
    __syncthreads();
  }

  for(int tilex = 0; tilex < TILEX; ++tilex) {
    for(int tiley = 0; tiley < TILEY; ++tiley) {
      output(bx_offset+tx_offset+tilex, by_offset+ty_offset+tiley) = tmp_z[tilex][tiley];
    }
  }
}

void gemm_gemm_blocktiling_gridblock(dim3 &grid, dim3 &block) {
  assert(BatchSize % BLOCKSIZEX == 0 && Nn % BLOCKSIZEY == 0);
  constexpr int GRIDDIMX = (BatchSize / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Nn / BLOCKSIZEY);
  block = dim3(BLOCKSIZEX*BLOCKSIZEY/(TILEX*TILEY), 1);
  grid = dim3(GRIDDIMX, GRIDDIMY);
}

__global__ void gemm_vectorize(const float *input, const float *weight, float *output) {
  __shared__ float input_blocked[BLOCKSIZEX][blocksizez];
  __shared__ float weight_blocked[blocksizez][BLOCKSIZEY];
  int tx = threadIdx.x / (BLOCKSIZEY/TILEY);
  int ty = threadIdx.x % (BLOCKSIZEY/TILEY);
  const int thread_per_x = BLOCKSIZEX/TILEX;
  const int thread_per_y = BLOCKSIZEY/TILEY;
	const int bx_offset = blockIdx.x * BLOCKSIZEX;
	const int by_offset = blockIdx.y * BLOCKSIZEY;
  const int tx_offset = tx*TILEX;
	const int ty_offset = ty*TILEY;
  float tmp_z[TILEX][TILEY] = {0.0f};
  for(int bz = 0; bz < Ni; bz += blocksizez) {
    for(int z = ty*4; z < blocksizez; z+=thread_per_y*4) {
      for(int tilex = 0; tilex < TILEX; tilex++) {
        // input_blocked[tx_offset+tilex][z] = input(bx_offset+tx_offset+tilex, bz+z);
        reinterpret_cast<float4 *>(&(input_blocked[tx_offset+tilex][z]))[0] =
        reinterpret_cast<float4 *>(const_cast<float*>(&(input(bx_offset+tx_offset+tilex, bz+z))))[0];
      }
    }

    for(int z = tx; z < blocksizez; z+=thread_per_x) {
      for(int tiley = 0; tiley < TILEY; tiley+=2) {
        // weight_blocked[z][ty_offset+tiley] = weight(bz+z, by_offset+ty_offset+tiley);
        reinterpret_cast<float2 *>(&(weight_blocked[z][ty_offset+tiley]))[0] =
        reinterpret_cast<float2 *>(const_cast<float*>(&(weight(bz+z, by_offset+ty_offset+tiley))))[0];
      }
    }
    __syncthreads();
    for(int z = 0; z < blocksizez; ++z) {
      for(int tilex = 0; tilex < TILEX; ++tilex) {
        for(int tiley = 0; tiley < TILEY; ++tiley) {
          tmp_z[tilex][tiley] += input_blocked[tx_offset+tilex][z] * weight_blocked[z][ty_offset+tiley];
        }
      }
    }
    __syncthreads();
  }

  for(int tilex = 0; tilex < TILEX; ++tilex) {
    for(int tiley = 0; tiley < TILEY; ++tiley) {
      output(bx_offset+tx_offset+tilex, by_offset+ty_offset+tiley) = tmp_z[tilex][tiley];
    }
  }
}


int main() {
  const int64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nn*Ni;
  auto input_length = BatchSize*Ni;
  auto output_length = BatchSize*Nn;
  auto weight_length = Ni * Nn;
  auto gemm_test = Test<float, decltype(gemm_seq)>(input_length, output_length, weight_length, float_calculation_num, "GEMM ");
  gemm_test.run_seq(gemm_seq);
  gemm_test.test_cuda(gemm_naive, gemm_naive_gridblock, "CUDA NAIVE");
  gemm_test.test_cuda(gemm_naive_shared, gemm_naive_gridblock, "CUDA NAIVE SHARED");
  gemm_test.test_cuda(gemm_coalescing, gemm_coalescing_gridblock, "CUDA coalescing");
  gemm_test.test_cuda(gemm_shared, gemm_coalescing_gridblock, "CUDA SHARED");
  gemm_test.test_cuda(gemm_blocktiling, gemm_gemm_blocktiling_gridblock, "CUDA TILING");
  gemm_test.test_cuda(gemm_vectorize, gemm_gemm_blocktiling_gridblock, "CUDA vectorize");
}
