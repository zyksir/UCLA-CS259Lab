#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

#include "lib/utils.h"
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
__global__ void gemm_cuda_naive(const float *input, const float *weight, float *output) {
 
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
__global__ void gemm_cuda(const float *input, const float *weight, float *output) {

  __shared__ float input_blocked[BLOCKSIZEX][BLOCKSIZEZ];
	__shared__ float weight_blocked[BLOCKSIZEZ][BLOCKSIZEY];

	int tx = threadIdx.x;
  int ty = threadIdx.y;
	int x_offset = blockIdx.x * BLOCKSIZEX;
	int y_offset = blockIdx.y * BLOCKSIZEY;

  float sum = 0;
  for(int ni = 0; ni < Ni; ni += BLOCKSIZEZ) {
    #pragma unroll
    for(int z = tx; z < BLOCKSIZEZ; z+=BLOCKSIZEX) {
      weight_blocked[z][ty] = weight(z+ni, ty+y_offset);
    }
    #pragma unroll
    for(int z = ty; z < BLOCKSIZEZ; z+=BLOCKSIZEY) {
      input_blocked[tx][z] = input(tx+x_offset, z+ni);
    }
    __syncthreads();

    #pragma unroll
    for(int z = 0; z < BLOCKSIZEZ; ++z) {
      sum += input_blocked[tx][z] * weight_blocked[z][ty];
    }
    __syncthreads();

  }
  output(tx+x_offset, ty+y_offset) = sum;
}

int main() {
  const int64_t float_calculation_num = 2*static_cast<uint64_t>(BatchSize)*Nn*Ni;
  auto input_length = BatchSize*Ni; auto input_size = input_length * sizeof(float);
  auto output_length = BatchSize*Nn; auto output_size = output_length * sizeof(float);
  auto weight_length = Ni * Nn; auto weight_size = weight_length * sizeof(float);
  float* input = static_cast<float*>(malloc(input_size));
  float* output = static_cast<float*>(malloc(output_size));
  float* weight = static_cast<float*>(malloc(weight_size));
  auto sta = std::chrono::steady_clock::now();
  GenerateRandomMatrix(input, input_length);
  GenerateRandomMatrix(weight, weight_length);
  memset(output, 0, output_size);
  std::chrono::duration<double> rand_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Generate Random Matrix]\tTimeCost:" << rand_duration.count() << "ns" << std::endl;

  sta = std::chrono::steady_clock::now();
  gemm_seq(input, weight, output);
  std::chrono::duration<double> gemm_seq_duration = std::chrono::steady_clock::now() - sta;
  print_performance_result(float_calculation_num, gemm_seq_duration, "GEMM SEQ");

  float* cuda_output = static_cast<float*>(malloc(output_size));
  float* g_input, *g_weight, *g_output;
  cudaMalloc((float**)&g_input, input_size);
  cudaMalloc((float**)&g_weight, weight_size);
  cudaMalloc((float**)&g_output, output_size);
  cudaMemcpy(g_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_weight, weight, weight_size, cudaMemcpyHostToDevice);
  cudaMemset(g_output, 0, output_size);

  constexpr int GRIDDIMX = (BatchSize / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Nn / BLOCKSIZEY);
  auto block = dim3(BLOCKSIZEX, BLOCKSIZEY);
  auto grid = dim3(GRIDDIMX, GRIDDIMY);
  clog << "Using thread block dims: " << block.x << ' ' << block.y << '\n';
  clog << "Using grid dims: " << grid.x << ' ' << grid.y << '\n';
  cudaSetDevice(0);

  sta = std::chrono::steady_clock::now();
  gemm_cuda<<<grid, block>>>(g_input, g_weight, g_output);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::duration<double> gemm_cuda_duration = std::chrono::steady_clock::now() - sta;
  print_performance_result(float_calculation_num, gemm_cuda_duration, "GEMM CUDA");

  cudaMemcpy(cuda_output, g_output, output_size, cudaMemcpyDeviceToHost);
  if (IsDiffMatrix(cuda_output, output, output_length)) {
    clog << "FAIL" << endl;
  } else {
    clog << "PASS" << endl;
  }

  /*****/
  cudaMemset(g_output, 0, output_size);
  sta = std::chrono::steady_clock::now();
  gemm_cuda_naive<<<grid, block>>>(g_input, g_weight, g_output);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::duration<double> gemm_cuda_naive_duration = std::chrono::steady_clock::now() - sta;
  print_performance_result(float_calculation_num, gemm_cuda_naive_duration, "GEMM CUDA NAIVE");

  cudaMemcpy(cuda_output, g_output, output_size, cudaMemcpyDeviceToHost);
  if (IsDiffMatrix(cuda_output, output, output_length)) {
    clog << "FAIL" << endl;
  } else {
    clog << "PASS" << endl;
  }
}
