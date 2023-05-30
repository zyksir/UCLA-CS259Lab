//
//  mat_mul_simple1.metal
//  metal_performance_testing
//
//  Created by Brian Vogel on 2022/08/27.
//

#include <metal_stdlib>
#include "gemm_params.h"
using namespace metal;
 
kernel void gemm_naive(device const float* A,
                        device const float* B,
                        device float* X,
                        constant GEMMParams& params,
                        uint2 id [[ thread_position_in_grid ]])
{
    // Note: matrices are in row-major order in the supplied backing arrays.
    const uint row_dim_x = params.x_rows;
    const uint col_dim_x = params.x_cols;
    const uint inner_dim = params.x_inner;
    
    // Check if the thread is in-bounds.
    if ((id.x < col_dim_x) && (id.y < row_dim_x)) {
        // id.x is the column index of the result matrix.
        // id.y is the row index of the result matrix.
        const uint index = id.y*col_dim_x + id.x;
        float sum = 0;
        for (uint k = 0; k < inner_dim; ++k) {
            // index_A corresponds to A[id.y, k]
            const uint index_A = id.y*inner_dim + k;
            
            // index_B corresponds to B[k, id.x]
            const uint index_B = k*col_dim_x + id.x;

            sum += A[index_A] * B[index_B];
        }
        X[index] = sum;
    }
}
