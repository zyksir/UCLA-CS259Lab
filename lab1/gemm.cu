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

// input: BatchSize * Ni  -> X * Z
// weight: Nn * Ni        -> Y * Z
// output: BatchSize * Nn -> X * Y
#define permute_weight(pweight, ni, nn) pweight[(nn)*Ni + (ni)]
__global__ void gemm_naive_shared_permute(const float *input, const float *pweight, float *output) {
  __shared__ float input_blocked[BLOCKSIZEX*BLOCKSIZEZ];
	int b = blockIdx.x * blockDim.x + threadIdx.x;
	int nn = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0;
  for(int ni = 0; ni < Ni; ni += BLOCKSIZEZ) {
    for(int z = threadIdx.y; z < BLOCKSIZEZ; z+=blockDim.y)
      input_blocked[threadIdx.x + z*BLOCKSIZEX] = input(b, z+ni);
    __syncthreads();

    for(int z = 0; z < BLOCKSIZEZ; ++z)
      sum += input_blocked[threadIdx.x + z*BLOCKSIZEX] * permute_weight(pweight, z+ni, nn);
    __syncthreads();
  }
  output(b, nn) = sum;
}

void gemm_naive_gridblock(dim3 &grid, dim3 &block) {
  assert(BatchSize % BLOCKSIZEX == 0 && Nn % BLOCKSIZEY == 0);
  constexpr int GRIDDIMX = (BatchSize / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Nn / BLOCKSIZEY);
  block = dim3(BLOCKSIZEX, BLOCKSIZEY, 1);
  grid = dim3(GRIDDIMX, GRIDDIMY, 1);
}

// input: BatchSize * Ni  -> X * Z
// weight: Ni * Nn        -> Z * Y
// output: BatchSize * Nn -> X * Y
constexpr int blocksizez = 1024/BLOCKSIZEX/BLOCKSIZEY;
constexpr int num_per_thread = Ni/blocksizez; 
__global__ void gemm_aggr(const float *input, const float *weight, float *output) {
  __shared__ float tmp_input[BLOCKSIZEX][blocksizez];
  __shared__ float tmp_weight[blocksizez][BLOCKSIZEY];
  __shared__ float sum;
  float tmp_output;
	int x = blockIdx.x * BLOCKSIZEX + threadIdx.x;
	int y = blockIdx.y * BLOCKSIZEY + threadIdx.y;
  int tz = threadIdx.z*num_per_thread;

  tmp_output = 0.0f;
  for(int z = tz; z < tz+num_per_thread; ++z) {
    tmp_output += input(x, z) * weight(z, y);
  }
  atomicAdd(&sum, tmp_output);
  __syncthreads();

  if (tz == 0) {
    output(x, y) = sum;
  }

  // float sum = 0;
  // for(int ni = 0; ni < Ni; ni += BLOCKSIZEZ) {
  //   for(int z = threadIdx.y; z < BLOCKSIZEZ; z+=blockDim.y)
  //     tmp_input[threadIdx.x + z*BLOCKSIZEX] = input(x, z+ni);
  //   __syncthreads();

  //   for(int z = 0; z < BLOCKSIZEZ; ++z)
  //     sum += tmp_input[threadIdx.x + z*BLOCKSIZEX] * weight(z+ni, y);
  //   __syncthreads();
  // }

}

void gemm_aggr_gridblock(dim3 &grid, dim3 &block) {
  assert(Ni % BLOCK_CHANNEL == 0);
  block = dim3(BLOCKSIZEX, BLOCKSIZEY, blocksizez);
  grid = dim3(BatchSize, Nn, 1);
}


int main() {
  const int64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nn*Ni;
  auto input_length = BatchSize*Ni;
  auto output_length = BatchSize*Nn;
  auto weight_length = Ni * Nn;
  auto gemm_test = Test<float, decltype(gemm_seq)>(input_length, output_length, weight_length, float_calculation_num, "GEMM ");
  gemm_test.run_seq(gemm_seq);
  auto reformat_weight = [](const float* weight) {
    float* pweight = static_cast<float*>(malloc(Nn * Ni * sizeof(float)));
    for (int nn = 0; nn < Nn; nn ++)
      for (int ni = 0; ni < Ni; ni ++)
        permute_weight(pweight, ni, nn) = weight(ni, nn);
    return pweight;
  };
  gemm_test.test_cuda(gemm_naive, gemm_naive_gridblock, "CUDA NAIVE");
  gemm_test.test_cuda(gemm_naive_shared, gemm_naive_gridblock, "CUDA SHARED");
  gemm_test.test_cuda(gemm_naive_shared_permute, gemm_naive_gridblock, "CUDA SHARED PERMUTE",
    nullptr, reformat_weight, nullptr
  );
  gemm_test.test_cuda(gemm_aggr, gemm_aggr_gridblock, "CUDA AGGR");
}
