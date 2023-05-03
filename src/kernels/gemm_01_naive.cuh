#pragma once 

// Naive GEMM implementation in CUDA
// Matrix size: NxK * KxM = NxM
__global__ void gemm_naive_kernel(const float *A, const float *B, float *C, const int N, const int K, const int M) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int m = threadIdx.y + blockIdx.y * blockDim.y;
    if (n < N && m < M) {
        float sum = 0;
        for(int k = 0; k < K; ++k) {
            sum += Val(A, n, k, K) * Val(B, k, m, M);
        }
        Val(C, n, m, M) = sum;
    }
}

template<const uint N, const uint K, const uint M, const uint BX, const uint BY>
void gemm_naive(const float *A, const float *B, float *C, dim3 &grid, dim3 &block) {
    block = dim3(BX, BY);
    grid = dim3(CEIL_DIV(N, BX), CEIL_DIV(M, BY));
    gemm_naive_kernel<<<grid, block>>>(A, B, C, N, K, M);
}