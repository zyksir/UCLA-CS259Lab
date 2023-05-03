#pragma once 

// Matrix Size: NxK * KxM = NxM
template<const uint BX, const uint BY, const uint BZ>
__global__ void gemm_naive_shared_kernel(const float *A, const float *B, float *C, const int N, const int K, const int M) {
  	__shared__ float bA[BX*BZ];
	int n = blockIdx.x * BX + threadIdx.x;
	int m = blockIdx.y * BY + threadIdx.y;

	float sum = 0;
	for(int k = 0; k < K; k += BZ) {
		for(int kk = threadIdx.y; kk < BZ; kk+=blockDim.y)
			bA[threadIdx.x + kk*BX] = Val(A, n, k+kk, K);
		__syncthreads();

		for(int kk = 0; kk < BZ; ++kk)
			sum += bA[threadIdx.x + kk*BX] * Val(B, k+kk, m, M);
		__syncthreads();
	}
	Val(C, n, m, M) = sum;
}

template<const uint N, const uint K, const uint M, const uint BX, const uint BY, const uint BZ>
void gemm_naive_shared(const float *A, const float *B, float *C, dim3 &grid, dim3 &block) {
	static_assert(K % BZ == 0 && N % BX == 0 && M % BY == 0, "");
    block = dim3(BX, BY);
    grid = dim3(CEIL_DIV(N, BX), CEIL_DIV(M, BY));
    gemm_naive_shared_kernel<BX, BY, BZ>
        <<<grid, block>>>(A, B, C, N, K, M);
}

template<const uint BX, const uint BY, const uint BZ>
__global__ void gemm_shared_kernel(const float *A, const float *B, float *C, const int N, const int K, const int M) {
	__shared__ float bA[BX*BZ];
	__shared__ float bB[BZ*BY];
	int tx = threadIdx.x / BY;
	int ty = threadIdx.x % BY;
	int n = blockIdx.x * BX + tx;
	int m = blockIdx.y * BY + ty;

  float sum = 0;
  for(int k = 0; k < K; k += BZ) {
    for(int kk = ty; kk < BZ; kk+=BY)
		Val(bA, tx, kk, BZ) = Val(A, n, k+kk, K);
     	// bA[tx*BZ + kk] =  Val(A, n, k+kk, K);
    for(int kk = tx; kk < BZ; kk+=BX)
		Val(bB, kk, ty, BY) = Val(B, k+kk, m, M);
    //   bB[kk*BY + ty] = Val(B, k+kk, m, M);
    __syncthreads();

    for(int kk = 0; kk < BZ; ++kk) {
      sum += Val(bA, tx, kk, BZ) * Val(bB, kk, ty, BY);
    }
    __syncthreads();
  }
  Val(C, n, m, M) = sum;
}

template<const uint N, const uint K, const uint M, const uint BX, const uint BY, const uint BZ>
void gemm_shared(const float *A, const float *B, float *C, dim3 &grid, dim3 &block) {
	static_assert(K % BZ == 0 && N % BX == 0 && M % BY == 0, "");
    block = dim3(BX*BY);
    grid = dim3(CEIL_DIV(N, BX), CEIL_DIV(M, BY));
    gemm_shared_kernel<BX, BY, BZ>
        <<<grid, block>>>(A, B, C, N, K, M);
}