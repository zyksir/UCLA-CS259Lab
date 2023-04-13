#include <stdio.h>
#include "macros.cuh"
#include "conv_gpu.h"

// kernal number that each thread need to deal with
constexpr int KERNAL_COUNT = (Nn / GRIDDIMZ / BLOCKSIZEZ);
constexpr int BLOCK_IN_X = (BLOCKSIZEX + Kx - 1);
constexpr int BLOCK_IN_Y = (BLOCKSIZEY + Ky - 1);
constexpr int KERNEL_SQUARE = Kx * Ky;

// Sequential CNN implementation
// input: Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: Nn * NySCL * NxSCL
__global__ void conv_gpu(float* input, float* weight, float* output) {
    __shared__ float weight_blocked[BLOCK_CHANNEL][Kx][Ky];
    __shared__ float input_blocked[BLOCK_CHANNEL][BLOCK_IN_X][BLOCK_IN_X];
    float output_thread[KERNAL_COUNT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int kernalOffset = blockIdx.z * KERNAL_COUNT;

    int row = blockIdx.y * BLOCKSIZEY;
    int col = blockIdx.x * BLOCKSIZEX;

    // set bias
    for (int k = kernalOffset; k < kernalOffset + KERNAL_COUNT; k++) {
        output_thread[k - kernalOffset] = 0.0f;
    }

    for(int ni = 0; ni < Nn; ni += BLOCK_CHANNEL) {

        /* Step1. load input to shared memory */
        int x, y;
        for(int i = 0; i < BLOCK_CHANNEL; ++i) {
            for (int offset = ty*BLOCKSIZEX + tx; offset < BLOCK_IN_X * BLOCK_IN_Y; offset += BLOCKSIZEX*BLOCKSIZEY) {
                x = offset % BLOCK_IN_X;
                y = offset / BLOCK_IN_X;
                input_blocked[i][y][x] = input(ni + i, row + y, col + x);
            }
        }

        for (int k = kernalOffset; k < kernalOffset + KERNAL_COUNT; k++) {
            /* Step2. load weight to shared memory */
            for(int offset = ty*BLOCKSIZEX + tx; offset < KERNEL_SQUARE * BLOCK_CHANNEL; offset += BLOCKSIZEX*BLOCKSIZEY) {
                int cz = offset / KERNEL_SQUARE;
                int cz_off = offset % KERNEL_SQUARE;
                int cx = cz_off % Kx;
                int cy = cz_off / Kx;
                weight_blocked[cz][cy][cx] = weight(k, ni + cz, cy, cx);
            }
            __syncthreads();

            /* Step3. Computation */
            for (int kk = 0; kk < BLOCK_CHANNEL; kk++) {
                for (int ky = 0; ky < Ky; ky++) {
                    for (int kx = 0; kx < Kx; kx++) {
                        output_thread[k - kernalOffset] += input_blocked[kk][ty+ky][tx+kx] * weight_blocked[kk][ky][kx];
                    }
                }
            }
            __syncthreads();
        }
    }

    // Relu
    for (int k = 0; k < KERNAL_COUNT; k++) {
        output(kernalOffset + k, row + ty, col + tx) = max(0.0f, output_thread[k]);
    }
}