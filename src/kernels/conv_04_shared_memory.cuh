#pragma once

// input: BatchSize * NyPAD * NxPAD * Ni
// weight: Ky * Kx * Ni * Nn
// output: BatchSize * NySCL * NxSCL * Nn
template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint TX, const uint TY, const uint TZ,
  const uint Sx=1, const uint Sy=1, 
  const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
  const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx,
  const uint BLOCKNUMX=Nx/TX, const uint BLOCKNUMY=Ny/TY, 
  const uint BIX=TX+Kx-1, const uint BIY=TY+Ky-1
>
__global__ void conv_shared_kernel(const float* input, const float* weight, float* output) {
    // assume blockDim.z == 1 and blockDim.y == 1
    int nn = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y; // blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.z/BLOCKNUMY*TY;
    int tx = blockIdx.z%BLOCKNUMY*TX;

    __shared__ float tmp_input[BIY][BIX][TZ];
    float tmp_output[TY][TX] = {0.0f};
    for(int bni = 0; bni < Ni; bni+=TZ) {
      for(int offset=threadIdx.x; offset<BIX*BIY*TZ; offset+=blockDim.x) {
        int y = offset / (BIX*TZ);
        int y_off = offset % (BIX*TZ);
        int x = y_off / TZ;
        int ni = y_off % TZ;
        tmp_input[y][x][ni] = Val4D(input, b, ty+y, tx+x, bni+ni, NyPAD, NxPAD, Ni);//channel_input(cinput, b, nii+ni, ty+ky, tx+kx);
      }
      __syncthreads();

      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          for(int ni = bni; ni < bni+TZ; ni++) {
            float tmp_weight = Val4D(weight, ky, kx, ni, nn, Kx, Ni, Nn); ;//channel_weight(cweight, nn, ni, ky, kx);
            for(int y=0; y < TY; ++y) {
              for(int x=0; x < TX; ++x) {
                tmp_output[y][x] += tmp_input[y+ky][x+kx][ni-bni] * tmp_weight;
              }
            }
          }
        }
      }
      __syncthreads();
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
  const uint TX, const uint TY, const uint TZ=4,
  const uint BLOCKNUMX=Nx/TX, const uint BLOCKNUMY=Ny/TY>
void conv_shared(const float* input, const float* weight, float* output, dim3 &grid, dim3 &block) {
	block = dim3(BZ, 1, 1);
	grid = dim3(Nn/BZ, B, BLOCKNUMX*BLOCKNUMY);
  conv_shared_kernel<B, Ni, Nn, Nx, Ny, Kx, Ky, TX, TY, TZ>
      <<<grid, block>>>(input, weight, output);
}


// // input: BatchSize * NyPAD * NxPAD * Ni
// // weight: Ky * Kx * Ni * Nn
// // output: BatchSize * NySCL * NxSCL * Nn
// template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
//   const uint Kx, const uint Ky, 
//   const uint Sx=1, const uint Sy=1, 
//   const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
//   const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx
// >
// __global__ void conv_shared_kernel(const float* input, const float* weight, float* output) {
//     // assume blockDim.z == 1 and blockDim.y == 1
//     int y = blockIdx.z / Ny;
//     int x = blockIdx.z % Ny;
//     int b = blockIdx.y;
//     int nn = blockIdx.x * blockDim.x + threadIdx.x;
//     const int blocksize = blockDim.x;

//     __shared__ float tmp_input[Ky][Kx][Ni];

//     for(int offset=threadIdx.x; offset<Ky*Kx*Ni; offset+=blocksize) {
//       int ky = offset / (Kx*Ni);
//       int ky_off = offset % (Kx*Ni);
//       int kx = ky_off / Ni;
//       int ni = ky_off % Ni;
//       tmp_input[ky][kx][ni] = Val4D(input, b, y+ky, x+kx, ni, NyPAD, NxPAD, Nii); // channel_input(cinput, b, ni, ty+ky, tx+kx);
//     }
//     __syncthreads();

//     float sum = 0.0f;
//     for (int ky = 0; ky < Ky; ky++) {
//       for (int kx = 0; kx < Kx; kx++) {
//         for(int ni = 0; ni < Ni; ni++) {
//           sum += tmp_input[ky][kx][ni] * Val4D(weight, ky, kx, ni, nn, Kxx, Nii, Nnn); // channel_weight(cweight, nn, ni, ky, kx);
//         }
//       }
//     }
//     Val4D(output, b, y, x, nn, NySCL, NxSCL, Nnn) = max(0.0f, sum);
// }

// template<const uint B, const uint Ni, const uint Nn, 
//   const uint Nx, const uint Ny,
//   const uint Kx, const uint Ky, 
//   const uint BZ>
// void conv_shared(const float* input, const float* weight, float* output, dim3 &grid, dim3 &block) {
// 	static_assert(Nn % BZ == 0, "BatchZ not valid");
// 	block = dim3(BZ, 1, 1);
// 	grid = dim3(Nn/BZ, B, Nx*Ny);
//   conv_shared_kernel<B, Ni, Nn, Nx, Ny, Kx, Ky>
//     <<<grid, block>>>(input, weight, output);
// }



// // input: BatchSize * Ni * NyPAD * NxPAD
// // weight: Nn * Ni * Ky * Kx
// // output: BatchSize * Nn * NySCL * NxSCL
// template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
//   const uint Kx, const uint Ky, 
//   const uint BX, const uint BY, const uint BC,
//   const uint Sx=1, const uint Sy=1, 
//   const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
//   const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx, const uint KSQUARE=Kx*Ky,
//   const uint BIX=BX+Kx-1, const uint BIY=BY+Ky-1, const uint BSQUARE=BIX*BIY
// >
// __global__ void conv_naive_shared_kernel(const float* input, const float* weight, float* output) {
//     // static_assert(BC*BIY*BIX<=1024, "shared memory too large");
//     __shared__ float weight_blocked[BC][Ky][Kx];
//     __shared__ float input_blocked[BC][BIY][BIX];
//     int block_size = blockDim.x*blockDim.y;

//     int tidx = threadIdx.x;
//     int tidy = threadIdx.y;

//     int bx = blockIdx.x * blockDim.x;
//     int by = blockIdx.y * blockDim.y;
//     int nn = blockIdx.z;

//     for(int b = 0; b < B; ++b) {
//       float sum = 0.0f;

//       for(int bni = 0; bni < Ni; bni += BC) {
//         /* Step1. load input to shared memory */
//         for(int ni=0; ni<BC; ni++) {
//           for(int offset=tidy*blockDim.x+tidx; offset < BSQUARE; offset += block_size) {
//             int x = offset % BIX;
//             int y = offset / BIX;
//             input_blocked[ni][y][x] = Val4D(input, b, bni+ni, y+by, x+bx, Ni, NyPAD, NxPAD);
//           }
//         }

//         /* Step2. load weight to shared memory */
//         for(int offset=tidy*blockDim.x+tidx; offset < BC*KSQUARE; offset += block_size) {
//             int ni = offset / KSQUARE;
//             int ni_off = offset % KSQUARE;
//             int kx = ni_off % Kx;
//             int ky = ni_off / Kx;
//             weight_blocked[ni][ky][kx] = Val4D(weight, nn, bni+ni, ky, kx, Ni, Ky, Kx);
//             // weight(nn, ni + cz, cy, cx);
//         }
//         __syncthreads();

//         /* Step3. Computation */
//         for (int ni = 0; ni < BC; ni++) {
//             for (int ky = 0; ky < Ky; ky++) {
//                 for (int kx = 0; kx < Kx; kx++) {
//                     sum += input_blocked[ni][tidy+ky][tidx+kx] * weight_blocked[ni][ky][kx];
//                 }
//             }
//         }
//         __syncthreads();
//       }

//       // Relu
//       Val4D(output, b, nn, by+tidy, bx+tidx, Nn, NySCL, NxSCL) = max(0.0f, sum);
//     }
// }

// template<const uint B, const uint Ni, const uint Nn, 
// const uint Nx, const uint Ny,
// const uint Kx, const uint Ky, 
// const uint BX, const uint BY, const uint BC=8>
// void conv_naive_shared(const float* input, const float* weight, float* output, dim3 &grid, dim3 &block) {
//   block = dim3(BX, BY, 1);
//   grid = dim3(CEIL_DIV(Nx, BX), CEIL_DIV(Ny, BY), Nn);
//   conv_naive_shared_kernel<B, Ni, Nn, Nx, Ny, Kx, Ky, BX, BY, BC>
//     <<<grid, block>>>(input, weight ,output);
// }

