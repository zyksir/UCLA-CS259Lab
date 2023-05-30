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
kernel void mat_mul_opt1(device const float* A,
                            device const float* B,
                            device float* X,
                            constant MatMulParams& params,
                            uint2 id [[ thread_position_in_grid ]])
{
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint row_dim_x = params.row_dim_x;
    const uint col_dim_x = params.col_dim_x;
    const uint inner_dim = params.inner_dim;
    const uint idx = id.x*4; // column index of the corner in X.
    const uint idy = id.y*4; // row index of the corner in X.
    // Note: float4x4 uses column major: Asub[m][n] is row n of column m.
    float4x4 Asub(0.0f);
    float4x4 Bsub(0.0f);
    float4x4 Xsub(0.0f);
    // bounds check can potentially be removed but does not seem to affect performance
    if ((idx < col_dim_x) && (idy < row_dim_x)) {
        uint k = 0;
        while (k < inner_dim) {
            // Read the values into 4x4 submatrices Asub and Bsub.
            for (uint j = 0; j < 4; ++j) { // column offset into X
                for (uint i = 0; i < 4; ++i) { // row offset into X
                    // corresponds to A[idy + i, k + j]
                    Asub[j][i] = A[(idy + i)*inner_dim + k + j];
                    // corresponds to B[k + i, idx + j]
                    Bsub[j][i] = B[(k + i)*col_dim_x + idx + j];
                }
            }
            // Multiply the two 4x4 submatrices and accumulate the result.
            Xsub += Asub * Bsub;
            k += 4;
        }
        // Write out the results.
        for (uint j = 0; j < 4; ++j) { // column offset into X
            for (uint i = 0; i < 4; ++i) { // row offset into X
                X[(idy + i)*col_dim_x + idx + j] = Xsub[j][i];
            }
        }
    }
}

