#pragma once

// kernal number that each thread need to deal with
constexpr int BLOCK_IN_X = (BLOCKSIZEX + Kx - 1);
constexpr int BLOCK_IN_Y = (BLOCKSIZEY + Ky - 1);
constexpr int BLOCK_IN_SQUARE = BLOCK_IN_X * BLOCK_IN_Y;


#define channel_input(cinput, b, ni, h, w)    cinput[(((b)*NyPAD + (h))*NxPAD + (w))*Ni + (ni)]
#define channel_output(coutput, b, nn, h, w)  coutput[(((b)*NySCL + (h))*NxSCL + (w))*Nn + (nn)]
#define channel_weight(cweight, nn, ni, p, q) cweight[(((p)*Kx + (q))*Ni + (ni))*Nn + (nn)]

__global__ void conv_batch(const float* cinput, const float* cweight, float* coutput) {
  int thread_nn = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_b = blockIdx.y;
  int grid_ny = blockIdx.z;

  for(int nx = 0; nx < Nx; nx += Sx) {
    float sum = 0.0f;
    for(int ky = 0; ky < Ky; ++ky) {
      for(int kx = 0; kx < Kx; ++kx) {
        for(int ni = 0; ni < Ni; ++ni) {
          sum += channel_weight(cweight, thread_nn, ni, ky, kx) * channel_input(cinput, grid_b, ni, grid_ny+ky, nx+kx);
        }
      }
    }
    // Perform Relu
    channel_output(coutput, grid_b, thread_nn, grid_ny, nx) = max(0.0f, sum);
  }
}

// CUDA CONV implementation
// input: BatchSize * Ni * NyPAD * NxPAD    -> X * Z
// weight: Nn * Ni * Ky * Kx                -> Y * Z   
// output: BatchSize * Nn * NySCL * NxSCL   -> X * Y
// parallelize(as wraps) along output channel dimension; parallelize along batch and y dimension
// used shared memory to share loaded inputs. 
__global__ void conv_batch_shared(const float* cinput, const float* cweight, float* coutput) {
  int thread_nn = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_b = blockIdx.y;
  int grid_ny = blockIdx.z;

  __shared__ float tmp_input[Ky][Kx][Ni];
  for(int ky = 0; ky < Ky; ++ky) {
    for(int kx = 0; kx < Kx; ++kx) {
      for(int ni = threadIdx.x; ni < Ni; ni += BLOCKSIZEZ) {
        tmp_input[ky][kx][ni] = channel_input(cinput, grid_b, ni, grid_ny+ky, kx);
      }
    }
  }
  __syncthreads();

  for(int nx_k = 0; nx_k < Nx; nx_k += Kx) {
    for (int nx_roll = 0; nx_roll < Kx && nx_k + nx_roll < Nx; nx_roll ++) {
      int nx = nx_k + nx_roll;
      float sum = 0.0f;

      int swapped_x = nx_roll == 0 ? Kx - 1: nx_roll - 1;
      for(int ky = 0; ky < Ky; ++ky) {
        for(int ni = threadIdx.x; ni < Ni; ni += BLOCKSIZEZ) {
          tmp_input[ky][swapped_x][ni] = channel_input(cinput, grid_b, ni, grid_ny+ky, nx+Kx-1);
        }
      }
      __syncthreads();

      for(int ky = 0; ky < Ky; ++ky) {
        for(int kx = 0; kx < Kx; ++kx) {
          int idx = nx_roll + kx;
          if (idx >= Kx) idx -= Kx;
          #pragma unroll 16
          for(int ni = 0; ni < Ni; ++ni) {
            sum += channel_weight(cweight, thread_nn, ni, ky, kx) * tmp_input[ky][idx][ni];
          }
        }
      }
      // Perform Relu
      channel_output(coutput, grid_b, thread_nn, grid_ny, nx) = max(0.0f, sum);
      // channel_parallel_output(grid_b, thread_nn, grid_ny, nx) = max(0.0f, sum);
      __syncthreads();
    }
  }
}

void conv_batch_gridblock(dim3 &grid, dim3 &block) {
  assert(Nn % BLOCKSIZEZ == 0);
  int GRIDDIM = Nn / BLOCKSIZEZ;

  block = dim3(BLOCKSIZEZ, 1, 1);
  grid = dim3(GRIDDIM, BatchSize, Ny);
}

// CUDA CONV implementation
// input: BatchSize * NyPAD * NxPAD * Ni
// weight: Ky * Kx * Ni * Nn
// output: BatchSize * NySCL * NxSCL * Nn
// each cuda thread is responsible for one sub block of CBLOCKSIZE*CBLOCKSIZE in one channel and one batch
constexpr int CBLOCKSIZE = 14;
constexpr int CBLOCKNUM = Nx/CBLOCKSIZE;
constexpr int CBLOCK_INSIZE = (14 + Kx - 1);
constexpr int Ni_BLOCK = 4;
__global__ void conv_channel_subblock_shared(const float* cinput, const float* cweight, float* coutput) {
    // assume blockDim.z == 1 and blockDim.y == 1
    int nn = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y; // blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.z/CBLOCKNUM*CBLOCKSIZE;
    int tx = blockIdx.z%CBLOCKNUM*CBLOCKSIZE;

    __shared__ float tmp_input[CBLOCK_INSIZE][CBLOCK_INSIZE][Ni_BLOCK];
    float tmp_output[CBLOCKSIZE][CBLOCKSIZE];
    for(int tyy = 0; tyy < CBLOCKSIZE; tyy++) {
      for(int txx = 0; txx < CBLOCKSIZE; txx++) {
        tmp_output[tyy][txx] = 0;
      }
    }
    for(int nii = 0; nii < Ni; nii+=Ni_BLOCK) {
      for(int offset=threadIdx.x; offset<CBLOCK_INSIZE*CBLOCK_INSIZE*Ni_BLOCK; offset+=blockDim.x) {
        int ky = offset / (CBLOCK_INSIZE*Ni_BLOCK);
        int ky_off = offset % (CBLOCK_INSIZE*Ni_BLOCK);
        int kx = ky_off / Ni_BLOCK;
        int ni = ky_off % Ni_BLOCK;
        tmp_input[ky][kx][ni] = channel_input(cinput, b, nii+ni, ty+ky, tx+kx);
      }
      __syncthreads();

      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          for(int ni = nii; ni < nii+Ni_BLOCK; ni++) {
            float tmp_weight = channel_weight(cweight, nn, ni, ky, kx);
            for(int tyy=0; tyy < CBLOCKSIZE; ++tyy) {
              for(int txx=0; txx < CBLOCKSIZE; ++txx) {
                tmp_output[tyy][txx] += tmp_input[tyy+ky][txx+kx][ni-nii] * tmp_weight;
              }
            }
          }
        }
      }
      __syncthreads();
    }
    for(int tyy = 0; tyy < CBLOCKSIZE; tyy++) {
      for(int txx = 0; txx < CBLOCKSIZE; txx++) {
        channel_output(coutput, b, nn, tyy+ty, txx+tx) = max(tmp_output[tyy][txx], 0.0f);
      }
    }
}

void conv_channel_subblock_gridblock(dim3 &grid, dim3 &block) {
	assert(Nn % BLOCKSIZEZ == 0);
  if (BLOCKSIZEZ >= 512) {
    block = dim3(256, 1, 1);
	  grid = dim3(Nn/256, BatchSize, CBLOCKNUM*CBLOCKNUM);
    return;
  }
	int GRIDDIMZ = Nn / BLOCKSIZEZ;

	block = dim3(BLOCKSIZEZ, 1, 1);
	grid = dim3(GRIDDIMZ, BatchSize, CBLOCKNUM*CBLOCKNUM);
}
