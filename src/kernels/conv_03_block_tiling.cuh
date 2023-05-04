#pragma once

// input: BatchSize * NyPAD * NxPAD * Ni
// weight: Ky * Kx * Ni * Nn
// output: BatchSize * NySCL * NxSCL * Nn
template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint TX, const uint TY,
  const uint Sx=1, const uint Sy=1, 
  const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
  const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx,
  const uint BLOCKNUMX=Nx/TX, const uint BLOCKNUMY=Ny/TY, 
  const uint BIX=TX+Kx-1, const uint BIY=TY+Ky-1
>
__global__ void conv_block_tiling_kernel(const float* input, const float* weight, float* output) {
    // assume blockDim.z == 1 and blockDim.y == 1
    int nn = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y; // blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.z/BLOCKNUMY*TY;
    int tx = blockIdx.z%BLOCKNUMY*TX;

    // __shared__ float tmp_input[BIY][BIX][TZ];
    float tmp_output[TY][TX] = {0.0f};
    for(int ni = 0; ni < Ni; ni++) {
      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          float tmp_weight = Val4D(weight, ky, kx, ni, nn, Kx, Ni, Nn); ;
          for(int y=0; y < TY; ++y) {
            for(int x=0; x < TX; ++x) {
              tmp_output[y][x] += Val4D(input, b, ty+y+ky, tx+x+kx, ni, NyPAD, NxPAD, Ni) * tmp_weight;
              // tmp_output[y][x] += tmp_input[y+ky][x+kx][ni-bni] * tmp_weight;
            }
          }
        }
      }
    }
    for(int y = 0; y < TY; y++) {
      for(int x = 0; x < TX; x++) {
        Val4D(output, b, y+ty, x+tx, nn, NySCL, NxSCL, Nn) = max(tmp_output[y][x], 0.0f);
      }
    }
}

template<const uint B, const uint Ni, const uint Nn, 
  const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint BZ,
  const uint TX, const uint TY,
  const uint BLOCKNUMX=Nx/TX, const uint BLOCKNUMY=Ny/TY>
void conv_block_tiling(const float* input, const float* weight, float* output, dim3 &grid, dim3 &block) {
	block = dim3(BZ, 1, 1);
	grid = dim3(Nn/BZ, B, BLOCKNUMX*BLOCKNUMY);
  conv_block_tiling_kernel<B, Ni, Nn, Nx, Ny, Kx, Ky, TX, TY>
      <<<grid, block>>>(input, weight, output);
}

