#pragma once 

// 由于 threadIdx.x 相邻的 thread 会放在一个 wrap 里执行
// 我们尽量让他们去访问相邻的数据。
// 优化后，这里一个 wrap 的 thread 会访问一个 input，并访问相邻的 weight

// Matrix Size: NxK * KxM = NxM
template<const uint BX, const uint BY>
__global__ void gemm_coalescing_kernel(const float *A, const float *B, float *C, const int N, const int K, const int M) {
    int n = threadIdx.x / BY + blockIdx.x * BX;
    int m = threadIdx.x % BY + blockIdx.y * BY;
    if (n < N && m < M) {
        float sum = 0;
        for(int k = 0; k < K; ++k) {
            sum += Val(A, n, k, K) * Val(B, k, m, M);
        }
        Val(C, n, m, M) = sum;
    }
}

template<const uint N, const uint K, const uint M, const uint BX, const uint BY>
void gemm_coalescing(const float *A, const float *B, float *C, dim3 &grid, dim3 &block) {
    block = dim3(BX*BY, 1);
    grid = dim3(CEIL_DIV(N, BX), CEIL_DIV(M, BY));
    gemm_coalescing_kernel<BX, BY>
        <<<grid, block>>>(A, B, C, N, K, M);
}

// for gemm2
// GFLOPS: 306 -> 1394
// dram_read_throughput: 40GB/s -> 163GB/s