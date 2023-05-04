#pragma once

// input: BatchSize * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint Sx=1, const uint Sy=1, 
  const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
  const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx
>
__global__ void conv_naive_kernel(const float* input, const float* weight, float* output) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int nn = blockIdx.z;

    for(int b = 0; b < BatchSize; ++b) {
      float sum = 0.0f;
      for(int ni = 0; ni < Ni; ni++) {
        for (int ky = 0; ky < Ky; ky++) {
            for (int kx = 0; kx < Kx; kx++) {
                sum += Val4D(weight, nn, ni, ky, kx, Ni, Ky, Kx) * Val4D(input, b, ni, ty+ky, tx+kx, Ni, NyPAD, NxPAD);
            //   sum += input(b, ni, ty+ky, tx+kx) * weight(nn, ni, ky, kx);
            }
        }
      }
      Val4D(output, b, nn, ty, tx, Nn, NySCL, NxSCL) = max(0.0f, sum);
    //   output(b, nn, ty, tx) = max(0.0f, sum);
    }
}

template<const uint B, const uint Ni, const uint Nn, 
const uint Nx, const uint Ny,
const uint Kx, const uint Ky, 
const uint BX, const uint BY>
void conv_naive(const float* input, const float* weight, float* output, dim3 &grid, dim3 &block) {
  block = dim3(BX, BY, 1);
  grid = dim3(CEIL_DIV(Nx, BX), CEIL_DIV(Ny, BY), Nn);
  if (BX > Nx && BY > Ny) {
    block = dim3(Nx, Ny, 1);
    grid = dim3(1, 1, Nn);
  }
  conv_naive_kernel<B, Ni, Nn, Nx, Ny, Kx, Ky>
    <<<grid, block>>>(input, weight ,output);
}