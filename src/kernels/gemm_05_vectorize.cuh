#pragma once 

// Matrix Size: NxK * KxM = NxM
template<const uint BX, const uint BY, const uint BZ, 
    const uint TX, const uint TY, 
	const uint TPBX=BX/TX, const uint TPBY=BY/TY // thread num per block in each dim
    >
__global__ void gemm_vectorize_kernel(const float *A, const float *B, float *C, const uint N, const uint K, const uint M) {
	__shared__ float bA[BX*BZ];
	__shared__ float bB[BZ*BY];
	int tidx = threadIdx.x / TPBY;
	int tidy = threadIdx.x % TPBY;
	const int bx = blockIdx.x * BX;
	const int by = blockIdx.y * BY;
	const int tx = tidx*TX;
	const int ty = tidy*TY;
	float tmp_z[TX*TY] = {0.0f};
	for(int bk = 0; bk < K; bk += BZ) {
		for(int k = tidy*4; k < BZ; k+=TPBY*4) {
		for(int x = 0; x < TX; x++) {
			reinterpret_cast<float4 *>(&(Val(bA, tx+x, k, BZ)))[0] =
        	reinterpret_cast<float4 *>(const_cast<float*>(&(Val(A, bx+tx+x, bk+k, K))))[0];
			// Val(bA, tx+x, k, BZ) = Val(A, bx+tx+x, bk+k, K);
		}
		}

		for(int k = tidx; k < BZ; k+=TPBX) {
		for(int y = 0; y < TY; y+=2) {
			reinterpret_cast<float2 *>(&(Val(bB, k, ty+y, BY)))[0] =
        	reinterpret_cast<float2 *>(const_cast<float*>(&(Val(B, bk+k, by+ty+y, M))))[0];
			// Val(bB, k, ty+y, BY) = Val(B, bk+k, by+ty+y, M);
		}
    	}
    	__syncthreads();
		for(int k = 0; k < BZ; ++k) {
		for(int x = 0; x < TX; ++x) {
		for(int y = 0; y < TY; ++y) {
			Val(tmp_z, x, y, TY) += Val(bA, tx+x, k, BZ) * Val(bB, k, ty+y, BY);
		}
		}
		}
		__syncthreads();
	}

	for(int x = 0; x < TX; ++x) {
	for(int y = 0; y < TY; ++y) {
		Val(C, bx+tx+x, by+ty+y, M) = Val(tmp_z, x, y, TY);
	}
	}
}

template<const uint N, const uint K, const uint M, 
  const uint BX, const uint BY, const uint BZ,
  const uint TX, const uint TY>
void gemm_vectorize(const float *A, const float *B, float *C, dim3 &grid, dim3 &block) {
	static_assert(K % BZ == 0 && N % BX == 0 && M % BY == 0, "");
    block = dim3(BX*BY/(TX*TY));
    grid = dim3(CEIL_DIV(N, BX), CEIL_DIV(M, BY));
    gemm_vectorize_kernel<BX, BY, BZ, TX, TY>
        <<<grid, block>>>(A, B, C, N, K, M);
}