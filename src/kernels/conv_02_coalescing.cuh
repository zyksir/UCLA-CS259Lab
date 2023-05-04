#pragma once

// input: BatchSize * NyPAD * NxPAD * Ni
// weight: Ky * Kx * Ni * Nn
// output: BatchSize * NySCL * NxSCL * Nn
template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint Sx=1, const uint Sy=1, 
  const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
  const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx
>
__global__ void conv_coalescing_kernel(const float* input, const float* weight, float* output) {
    // assume blockDim.x == 1 and blockDim.y == 1
    int ty = blockIdx.z / Ny;
    int tx = blockIdx.z % Ny;
    int b = blockIdx.y;
    // int tx = blockIdx.z;
    // int ty = blockIdx.y;
    int nn = blockIdx.x * blockDim.x + threadIdx.x;

    // for(int b = 0; b < B; ++b) {
      float sum = 0.0f;
      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          for(int ni = 0; ni < Ni; ni++) {
            // sum += Val4D(weight, nn, ni, ky, kx, Ni, Ky, Kx) * Val4D(input, b, ni, ty+ky, tx+kx, Ni, NyPAD, NxPAD);
            sum += Val4D(input, b, ty+ky, tx+kx, ni, NyPAD, NxPAD, Ni) * Val4D(weight, ky, kx, ni, nn, Kx, Ni, Nn);
          }
        }
      }
      Val4D(output, b, ty, tx, nn, NySCL, NxSCL, Nn) = max(0.0f, sum);
    // }
}

template<const uint B, const uint Ni, const uint Nn, 
  const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint BZ>
void conv_coalescing(const float* input, const float* weight, float* output, dim3 &grid, dim3 &block) {
	static_assert(Nn % BZ == 0, "BatchZ not valid");
	// block = dim3(BZ, 1, 1);
	// grid = dim3(Nn/BZ, Ny, Nx);
  block = dim3(BZ, 1, 1);
	grid = dim3(Nn/BZ, B, Nx*Ny);
  conv_coalescing_kernel<B, Ni, Nn, Nx, Ny, Kx, Ky>
    <<<grid, block>>>(input, weight, output);
}