#include <stdio.h>
#include "lib/macros.cuh"
#include "cnn_gpu.h"

// kernal number that each thread need to deal with
constexpr int KERNAL_COUNT = (kNum / GRIDDIMZ / BLOCKSIZEZ);
constexpr int BLOCK_OUT_SIZE = (BLOCKSIZEX << 1) ; //(kImSize / GRIDDIMX);
constexpr int THREAD_SIZE = BLOCKSIZEX;
constexpr int BLOCK_IN_SIZE = (BLOCK_OUT_SIZE + kKernel - 1);
constexpr int KERNEL_SQUARE = kKernel * kKernel;

__global__ void cnn_gpu(float* input, float* weight, float* bias, float* output) {
    // each block: 4 * 5 * 5 + 4 * 36 * 36 = 5284
    __shared__ float weight_blocked[BLOCK_CHANNEL][kKernel][kKernel];
    __shared__ float input_blocked[BLOCK_CHANNEL][BLOCK_IN_SIZE][BLOCK_IN_SIZE];

    float output_thread[KERNAL_COUNT][4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int kernalOffset = blockIdx.z * KERNAL_COUNT;

    int row = blockIdx.y * BLOCK_OUT_SIZE;
    int col = blockIdx.x * BLOCK_OUT_SIZE;

    // set bias
    for (int k = kernalOffset; k < kernalOffset + KERNAL_COUNT; k++) {
        output_thread[k - kernalOffset][0] = bias[k];
        output_thread[k - kernalOffset][1] = bias[k];
        output_thread[k - kernalOffset][2] = bias[k];
        output_thread[k - kernalOffset][3] = bias[k];
    }

    // for each kernal, seperate to BLOCK_CHANNEL * kKernel * kKernel
    for (int c = 0; c < kNum; c += BLOCK_CHANNEL) {

        // load input to shared memory
        int x, y;
        #pragma unroll
        for (int i = 0; i < BLOCK_CHANNEL; i++) {
            #pragma unroll
            for (int offset = ty * THREAD_SIZE + tx; offset < BLOCK_IN_SIZE * BLOCK_IN_SIZE; offset += THREAD_SIZE * THREAD_SIZE) {
                x = offset % BLOCK_IN_SIZE;
                y = offset / BLOCK_IN_SIZE;
                input_blocked[i][y][x] = input(c + i, row + y, col + x);
            }
        }
        // __syncthreads();

        // calculate for different kernal
        for (int k = kernalOffset; k < kernalOffset + KERNAL_COUNT; k++) {
            #pragma unroll
            for(int offset = ty * THREAD_SIZE + tx; offset < kKernel * kKernel * BLOCK_CHANNEL; offset += THREAD_SIZE * THREAD_SIZE) {
                int cz = offset / KERNEL_SQUARE;
                int cz_off = offset % KERNEL_SQUARE;
                int cx = cz_off % kKernel;
                int cy = cz_off / kKernel;
                weight_blocked[cz][cy][cx] = weight(k, c + cz, cy, cx);
            }
            __syncthreads();

            // calculate
            for (int kk = 0; kk < BLOCK_CHANNEL; kk++) {
                for (int yy = 0; yy < kKernel; yy++) {
                    for (int xx = 0; xx < kKernel; xx++) {
                        output_thread[k - kernalOffset][0] += input_blocked[kk][2*ty+yy][2*tx+xx] * weight_blocked[kk][yy][xx];
                        output_thread[k - kernalOffset][1] += input_blocked[kk][2*ty+1+yy][2*tx+xx] * weight_blocked[kk][yy][xx];
                        output_thread[k - kernalOffset][2] += input_blocked[kk][2*ty+yy][2*tx+1+xx] * weight_blocked[kk][yy][xx];
                        output_thread[k - kernalOffset][3] += input_blocked[kk][2*ty+1+yy][2*tx+1+xx] * weight_blocked[kk][yy][xx];
                    }
                }
            }
            __syncthreads();
        }
    }

    // Relu && Pooling
    for (int k = 0; k < KERNAL_COUNT; k++) {
        output(kernalOffset + k, row/2 + ty, col/2 + tx) = max(0.0f, max(
            max(output_thread[k][0], output_thread[k][1]),
            max(output_thread[k][2], output_thread[k][3])
        ));
    }
}