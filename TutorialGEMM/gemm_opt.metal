//
//  mat_mul_opt2.metal
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/09/04.
//

#include <metal_stdlib>
#include "gemm_params.h"
using namespace metal;


/**
 * Matrix multiplication example: X = A * B
 *
 * Implementaiton notes:
 * Each thread computes the result for an 8x4 sub-matrix of X using two 4x4 sub-matrices
 * which are vertically stacked in X and in A and a single corresponding 4x4 sub-matrix
 * in B.
 * This uses thread-independent tiling with 4x4 tile matrix size, but
 * The grid is set up so that there will be `row-dim_x/8` threads along the row dimension of
 * X and `col_dim_x/4` threads along the column dimension.
 *
 * Each thread position in the grid then corresponds to a corner of an 8x4 submatrix in X.
 * Each thread will be responsible for computing the result values in X for its two stacked 4x4
 * submatrices. This is done by tiling A and B into 4x4 submatrices and accumulating
 * the partial products. Note that we do this without using shared threadgroup memory
 * or synchronization barriers.
 *
 * Requirements:
 * - The supplied matrices must be in row-major order.
 * - The row dimension of X and A must an integer multiple of 8. The other dimensions of X, A, and
 *  B must be a integer mutiple of 4.
 *
 */
kernel void gemm_opt(device const float* A,
                            device const float* B,
                            device float* X,
                            constant GEMMParams& params,
                            uint2 id [[ thread_position_in_grid ]])
{
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint row_dim_x = params.x_rows;
    const uint col_dim_x = params.x_cols;
    const uint inner_dim = params.x_inner;
    const uint idx = id.x*4; // column index of the corner in X.
    const uint idy = id.y*8; // row index of the corner in X.
    // Note: float4x4 uses column major: Asub[m][n] is row n of column m.
    float4x4 Asub(0.0f);
    float4x4 Bsub(0.0f);
    float4x4 Xsub(0.0f);
    float4x4 Asub2(0.0f);
    float4x4 Xsub2(0.0f);
    // bounds check can potentially be removed but does not seem to affect performance
    if ((idx >= col_dim_x) || (idy >= row_dim_x)) return;

    if ((idx+4 < col_dim_x) && (idy+4 < row_dim_x)) {
        uint k = 0;
        while (k+4 < inner_dim) {
            // Read the values into the 4x4 submatrices.
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to A[idy + i, k + j]
                    Asub[j][i] = A[(idy + i)*inner_dim + k + j];
                }
            }
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to B[k + i, idx + j]
                    Bsub[j][i] = B[(k + i)*col_dim_x + idx + j];
                }
            }
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to A[idy + i + 4, k + j]
                    Asub2[j][i] = A[(idy + i + 4)*inner_dim + k + j];
                }
            }
            // Multiply the 4x4 submatrices and accumulate the result.
            Xsub += Asub * Bsub;
            Xsub2 += Asub2 * Bsub;
            k += 4;
        }
        if (k < inner_dim) {
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < inner_dim-k; ++j) { // column offset into X
                    // corresponds to A[idy + i, k + j]
                    Asub[j][i] = A[(idy + i)*inner_dim + k + j];
                }
            }
            for (uint i = 0; i < inner_dim-k; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to B[k + i, idx + j]
                    Bsub[j][i] = B[(k + i)*col_dim_x + idx + j];
                }
            }
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < inner_dim-k; ++j) { // column offset into X
                    // corresponds to A[idy + i + 4, k + j]
                    Asub2[j][i] = A[(idy + i + 4)*inner_dim + k + j];
                }
            }
            // Multiply the 4x4 submatrices and accumulate the result.
            Xsub += Asub * Bsub;
            Xsub2 += Asub2 * Bsub;
        }
        // Write out the results.
        for (uint i = 0; i < 4; ++i) { // row offset into X
            for (uint j = 0; j < 4; ++j) { // column offset into X
                if (idy+i<row_dim_x && idx + j < col_dim_x) {
                    X[(idy + i)*col_dim_x + idx + j] = Xsub[j][i];
                }
                if (idy+i+4<row_dim_x && idx + j < col_dim_x) {
                    X[(idy + i + 4)*col_dim_x + idx + j] = Xsub2[j][i];
                }
            }
        }
    }
}

kernel void gemm_opt2(device const float* A,
                            device const float* B,
                            device float* X,
                            constant GEMMParams& params,
                            uint2 id [[ thread_position_in_grid ]])
{
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint row_dim_x = params.x_rows;
    const uint col_dim_x = params.x_cols;
    const uint inner_dim = params.x_inner;
    const uint idx = id.x*4; // column index of the corner in X.
    const uint idy = id.y*8; // row index of the corner in X.
    // Note: float4x4 uses column major: Asub[m][n] is row n of column m.
    float4x4 Asub(0.0f);
    float4x4 Bsub(0.0f);
    float4x4 Xsub(0.0f);
    float4x4 Asub2(0.0f);
    float4x4 Xsub2(0.0f);
    // bounds check can potentially be removed but does not seem to affect performance
    if ((idx >= col_dim_x) || (idy >= row_dim_x)) return;

    if ((idx+4 < col_dim_x) && (idy+4 < row_dim_x)) {
        uint k = 0;
        while (k+4 < inner_dim) {
            // Read the values into the 4x4 submatrices.
            for (uint i = 0; i < 4; ++i) { // row offset into X
                // corresponds to A[idy + i, k + j], j = 0 to 4
                Asub[i] = *(device const float4*)&A[(idy + i)*inner_dim + k];
                // corresponds to B[k + i, idx + j], j = 0 to 4
                Bsub[i] = *(device const float4*)&B[(k + i)*col_dim_x + idx];
                // corresponds to A[idy + i + 4, k + j], j = 0 to 4
                Asub2[i] = *(device const float4*)&A[(idy + i + 4)*inner_dim + k];
            }
            // Multiply the 4x4 submatrices and accumulate the result.
            // because they are row-major, multiply is reversed.
            Xsub += Bsub * Asub;
            Xsub2 += Bsub * Asub2;
            k += 4;
        }
        if (k < inner_dim) {
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < inner_dim-k; ++j) { // column offset into X
                    // corresponds to A[idy + i, k + j]
                    Asub[i][j] = A[(idy + i)*inner_dim + k + j];
                }
            }
            for (uint i = 0; i < inner_dim-k; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to B[k + i, idx + j]
                    Bsub[i][j] = B[(k + i)*col_dim_x + idx + j];
                }
            }
            for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < inner_dim-k; ++j) { // column offset into X
                    // corresponds to A[idy + i + 4, k + j]
                    Asub2[i][j] = A[(idy + i + 4)*inner_dim + k + j];
                }
            }
            // Multiply the 4x4 submatrices and accumulate the result.
            // because they are row-major, multiply is reversed.
            Xsub += Bsub * Asub;
            Xsub2 += Bsub * Asub2;
        }
        
        // given: (idx+4 < col_dim_x) && (idy+4 < row_dim_x)
        // Write out the results.
        for (uint i = 0; i < 4; ++i) { // row offset into X
            *(device float4*)&X[(idy + i)*col_dim_x + idx] = Xsub[i];
            *(device float4*)&X[(idy + i + 4)*col_dim_x + idx] = Xsub2[i];
        }
    }
}

kernel void gemm_tiling(device const float* A,
                            device const float* B,
                            device float* X,
                            constant GEMMParams& params,
                            uint2 id [[ thread_position_in_grid ]])
{
    constexpr int TX = 1;
    constexpr int TY = 2;
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint row_dim_x = params.x_rows;
    const uint col_dim_x = params.x_cols;
    const uint inner_dim = params.x_inner;
    const uint idx = id.x*params.TX; // column index of the corner in X.
    const uint idy = id.y*params.TY; // row index of the corner in X.
    // Note: float4x4 uses column major: Asub[m][n] is row n of column m.
    float4x4 Asub[TY] = {{0.0f}};
    float4x4 Bsub[TX] = {{0.0f}};
    float4x4 Xsub[TY][TX] = {{{0.0f}}};
    // bounds check can potentially be removed but does not seem to affect performance
    if ((idx < col_dim_x) && (idy < row_dim_x)) {
        uint k = 0;
        while (k < inner_dim) {
            // Read the values into the 4x4 submatrices.
            for(int ty = 0; ty < TY; ++ty) {
                for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to A[idy + i, k + j]
                    Asub[ty][j][i] = A[(idy + i + 4*ty)*inner_dim + k + j];
                }
                }
            }
            for(int tx = 0; tx < TX; ++tx) {
                for (uint i = 0; i < 4; ++i) { // row offset into X
                for (uint j = 0; j < 4; ++j) { // column offset into X
                    // corresponds to B[k + i, idx + j]
                    Bsub[tx][j][i] = B[(k + i)*col_dim_x + idx + j + 4*tx];
                }
                }
            }

            for(int tx = 0; tx < TX; ++tx) {
            for(int ty = 0; ty < TY; ++ty) {
                Xsub[ty][tx] += Asub[ty] * Bsub[tx];
            }
            }
            k += 4;
        }
        // Write out the results.
        for (uint i = 0; i < 4; ++i) { // row offset into X
        for (uint j = 0; j < 4; ++j) { // column offset into X
            for(int tx = 0; tx < TX; ++tx) {
            for(int ty = 0; ty < TY; ++ty) {
                X[(idy + i + 4*ty)*col_dim_x + idx + j + 4*tx] = Xsub[ty][tx][j][i];
            }
            }
        }
        }
    }
}