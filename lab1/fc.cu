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

#define weight(nn, ni) weight[(nn)*Ni + (ni)]

// Sequential FC implementation
// input: Ni
// weight: Nn * Ni
// output: Nn
void fc_seq(const float *input, const float *weight, float *output) {
    for(int nn = 0; nn < Nn; ++nn) {
        float sum_sc = 0.0f;
        for(int ni = 0; ni < Ni; ++ni) {
            sum_sc += input[ni] * weight(nn, ni);
        }
        output[nn] = sum_sc;
    }
}

// CUDA FC implementation
// input: Ni
// weight: Nn * Ni
// output: Nn
__global__ void fc_cuda(const float *input, const float *weight, float *output) {

  // int ty = threadIdx.y + blockIdx.y * BLOCKSIZEY;
	// float sum = 0;
	// for(int x = 0; x < Ni; ++x) {
	// 	sum += input[x] * weight(ty, x);
	// }
  // output[ty] += sum;

	__shared__ float weight_blocked[BLOCKSIZEY][BLOCKSIZEX];
  __shared__ float input_blocked[BLOCKSIZEX];

	int ty = threadIdx.y;
	int x_offset = blockIdx.x * BLOCKSIZEX;
	int y_offset = blockIdx.y * BLOCKSIZEY;

  for(int tx = ty; tx < BLOCKSIZEX; tx += BLOCKSIZEY) {
    input_blocked[tx] = input[x_offset + tx];
  }

	int y, x;
	for(int offset = ty; offset < BLOCKSIZEY*BLOCKSIZEX; offset += BLOCKSIZEY) {
		y = offset / BLOCKSIZEX;
		x = offset % BLOCKSIZEX;
		weight_blocked[y][x] = weight(y_offset + y, x_offset + x);
	}

	__syncthreads();
	float sum = 0;
	for(int x = 0; x < BLOCKSIZEX; ++x) {
		sum += input_blocked[x] * weight_blocked[ty][x];
	}

	atomicAdd(output + ty + y_offset, sum);
}

int main() {
  auto input_length = Ni; auto input_size = input_length * sizeof(float);
  auto output_length = Nn; auto output_size = output_length * sizeof(float);
  auto weight_length = Nn * Ni; auto weight_size = weight_length * sizeof(float);
  float* input = static_cast<float*>(malloc(input_size));
  float* output = static_cast<float*>(malloc(output_size));
  float* weight = static_cast<float*>(malloc(weight_size));
  auto sta = std::chrono::steady_clock::now();
  GenerateRandomMatrix(input, input_length);
  GenerateRandomMatrix(weight, weight_length);
  std::chrono::duration<double> rand_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Generate Random Matrix]\tTimeCost:" << rand_duration.count() << "ns" << std::endl;

  sta = std::chrono::steady_clock::now();
  fc_seq(input, weight, output);
  std::chrono::duration<double> conv_seq_duration = std::chrono::steady_clock::now() - sta;
  clog << "[Conv Sequence]\tTimeCost:" << conv_seq_duration.count() << "ns" << std::endl;

  float* cuda_output = static_cast<float*>(malloc(output_size));
  float* g_input, *g_weight, *g_output;
  cudaMalloc((float**)&g_input, input_size);
  cudaMalloc((float**)&g_weight, weight_size);
  cudaMalloc((float**)&g_output, output_size);
  cudaMemcpy(g_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(g_weight, weight, weight_size, cudaMemcpyHostToDevice);
  cudaMemset(g_output, 0, output_size);

  constexpr int GRIDDIMX = (Ni / BLOCKSIZEX);
  constexpr int GRIDDIMY = (Nn / BLOCKSIZEY);
  auto block = dim3(1, BLOCKSIZEY);
  auto grid = dim3(GRIDDIMX, GRIDDIMY);
  clog << "Using thread block dims: " << block.x << ' ' << block.y << '\n';
  clog << "Using grid dims: " << grid.x << ' ' << grid.y << '\n';
  cudaSetDevice(0);

  sta = std::chrono::steady_clock::now();
  fc_cuda<<<grid, block>>>(g_input, g_weight, g_output);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::duration<double> fc_cuda_duration = std::chrono::steady_clock::now() - sta;
  clog << "[FC CUDA]\tTimeCost:" << fc_cuda_duration.count() << "ns" << std::endl;

  cudaMemcpy(cuda_output, g_output, output_size, cudaMemcpyDeviceToHost);
  if (IsDiffMatrix(cuda_output, output, output_length)) {
    clog << "FAIL" << endl;
  } else {
    clog << "PASS" << endl;
  }
}
