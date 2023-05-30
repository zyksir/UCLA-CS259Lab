//
//  mat_mul_opt1.metal
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/09/04.
//

#include <metal_stdlib>
#include "ShaderParams.h"
using namespace metal;


/**
 * Matrix multiplication example: X = A * B
 *
 * Implementaiton notes:
 * This uses thread-independent tiling with 4x4 tile matrix size.
 * The grid is set up so that the total number of threads along the row and column
 * dimension is 1/4 the size of the corresponding dimension in the result matrix X.
 * Each thread position in the grid then corresponds to a corner of a 4x4 submatrix in X.
 * Each thread will then be responsible for computing the result values in X for its 4x4
 * submatrix. This is done by tiling A and B into 4x4 submatrices and accumulating
 * the partial products. Note that we do this without using shared threadgroup memory
 * or synchronization barriers.
 *
 * Requirements:
 * - The supplied matrices must be in row-major order.
 * - Each matrix dimension must be an integer multiple of 4.
 *
 */
 kernel void inspector(
                  device uint* store [[buffer(0)]],
                  uint2 threadIdx [[ thread_position_in_threadgroup ]],
                  uint2 blockIdx [[ threadgroup_position_in_grid ]],
				  uint2 blockDim [[ threads_per_threadgroup ]],
				  uint2 id [[ thread_position_in_grid ]] )
{
    if (id.x == 27 && id.y == 27){
        store[0] = threadIdx.x;
        store[1] = threadIdx.y;
        store[2] = blockIdx.x;
        store[3] = blockIdx.y;
        store[4] = blockDim.x;
        store[5] = blockDim.y;
        store[6] = id.x;
        store[7] = id.y;
    }

}

 kernel void mat_mul_coalescing(device const float* A,
                            device const float* B,
                            device float* X,
                            constant MatMulParams& params,
                            uint2 threadIdx [[ thread_position_in_threadgroup ]],
                            uint2 blockIdx [[ threadgroup_position_in_grid ]],
							uint2 blockDim [[ threads_per_threadgroup ]],
							uint2 id [[ thread_position_in_grid ]])
{
    const uint N = params.row_dim_x;
    const uint M = params.col_dim_x;
    const uint K = params.inner_dim;
    const uint BX = 32, BY = 32;
    // Note: matrices are in row-major order in the supplied backing arrays.
    int n = threadIdx.x / BY + blockIdx.x * BX;
    int m = threadIdx.x % BY + blockIdx.y * BY;
    if (n < N && m < M) {
        float sum = 0;
        for(int k = 0; k < K; ++k) {
            sum += Val(A, n, k, K) * Val(B, k, m, M);
        }
        Val(X, n, m, M) = sum;
    }
}


kernel void mat_mul_opt3(device const float* A,
                            device const float* B,
                            device float* C,
                            constant MatMulParams& params,
                            uint2 threadIdx [[ thread_position_in_threadgroup ]],
                            uint2 blockIdx [[ threadgroup_position_in_grid ]])
{
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint N = params.row_dim_x;
    const uint M = params.col_dim_x;
    const uint K = params.inner_dim;
    const uint BX = 32, BY = 32, BZ = 32;
    const uint TX = 4, TY = 4;
    const uint TPBX = BX/TX, TPBY = BY/TY;

    threadgroup float bA[BX*BZ];
	threadgroup float bB[BZ*BY];
	int tidx = threadIdx.x / TPBY;
	int tidy = threadIdx.x % TPBY;
	const int bx = blockIdx.x * BX;
	const int by = blockIdx.y * BY;
	const int tx = tidx*TX;
	const int ty = tidy*TY;
	float tmp_z[TX*TY] = {0.0f};
	for(int bk = 0; bk < K; bk += BZ) {
		for(int k = tidy; k < BZ; k+=TPBY) {
		for(int x = 0; x < TX; x++) {
			Val(bA, tx+x, k, BZ) = Val(A, bx+tx+x, bk+k, K);
		}
		}

		for(int k = tidx; k < BZ; k+=TPBX) {
		for(int y = 0; y < TY; y++) {
			Val(bB, k, ty+y, BY) = Val(B, bk+k, by+ty+y, M);
		}
    	}
    	threadgroup_barrier(mem_flags::mem_none);
		for(int k = 0; k < BZ; ++k) {
		for(int x = 0; x < TX; ++x) {
		for(int y = 0; y < TY; ++y) {
			Val(tmp_z, x, y, TY) += Val(bA, tx+x, k, BZ) * Val(bB, k, ty+y, BY);
		}
		}
		}
		threadgroup_barrier(mem_flags::mem_none);
	}

	for(int x = 0; x < TX; ++x) {
	for(int y = 0; y < TY; ++y) {
		Val(C, bx+tx+x, by+ty+y, M) = Val(tmp_z, x, y, TY);
	}
	}

}

kernel void mat_mul_opt4(device const float* A,
                            device const float* B,
                            device float* C,
                            constant MatMulParams& params,
                            uint2 threadIdx [[ thread_position_in_threadgroup ]],
                            uint2 blockIdx [[ threadgroup_position_in_grid ]])
{
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint N = params.row_dim_x;
    const uint M = params.col_dim_x;
    const uint K = params.inner_dim;
    const uint BX = 32, BY = 32, BZ = 64;
    const uint TX = 4, TY = 4;
    const uint TPBX = BX/TX, TPBY = BY/TY;

    threadgroup float4x4 bA[BX/4*BZ/4];
	threadgroup float4x4 bB[BZ/4*BY/4];
	int tidx = threadIdx.x / TPBY;
	int tidy = threadIdx.x % TPBY;
	const int bx = blockIdx.y * BX;
	const int by = blockIdx.x * BY;
	const int tx = tidx*TX;
	const int ty = tidy*TY;
	float4x4 tmp_z[TX/4*TY/4] = {{0.0f}};
	for(int bk = 0; bk < K; bk += BZ) {
		for(int k = tidy*4; k < BZ; k+=TPBY*4) {
		for(int x = 0; x < TX; x+=4) {
            // load submatrix start from [x+tx, k]
            for(int kk = 0; kk < 4; ++kk) {
            for(int xx = 0; xx < 4; ++xx) {
                Val(bA, (tx+x)/4, k/4, BZ/4)[kk][xx] = Val(A, bx+tx+x+xx, bk+k+kk, K);
            }
            }
			// Val(bA, (tx+x)/4, k/4, BZ/4)[k%4][(tx+x)%4] = Val(A, bx+tx+x, bk+k, K);
		}
		}

		for(int k = tidx*4; k < BZ; k+=TPBX*4) {
		for(int y = 0; y < TY; y+=4) {
            // load submatrix start from [x+tx, k]
            for(int yy = 0; yy < 4; ++yy) {
            for(int kk = 0; kk < 4; ++kk) {
                Val(bB, k/4, (ty+y)/4, BY/4)[yy][kk] = Val(B, bk+k+kk, by+ty+y+yy, M);
            }
            }
			// Val(bB, k/4, (ty+y)/4, BY/4)[(ty+y)%4][k%4] = Val(B, bk+k, by+ty+y, M);
		}
    	}
    	threadgroup_barrier(mem_flags::mem_none);
		for(int k = 0; k < BZ; k+=4) {
		for(int x = 0; x < TX; x+=4) {
		for(int y = 0; y < TY; y+=4) {
			Val(tmp_z, x/4, y/4, TY/4) += Val(bA, (tx+x)/4, k/4, BZ/4) * Val(bB, k/4, (ty+y)/4, BY/4);
		}
		}
		}
		threadgroup_barrier(mem_flags::mem_none);
	}

	for(int x = 0; x < TX/4; x+=4) {
	for(int y = 0; y < TY/4; y+=4) {
        for(int xx = 0; xx < 4; ++xx) {
        for(int yy = 0; yy < 4; ++yy) {
            Val(C, bx+tx+x+xx, by+ty+y+yy, M) = Val(tmp_z, x/4, y/4, TY/4)[yy][xx];
        }
        }
		// Val(C, bx+tx+x, by+ty+y, M) = Val(tmp_z, x/4, y/4, TY/4)[y%4][x%4];
	}
	}

}
