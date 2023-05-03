#pragma once

// Sequential GEMM implementation
// input: N * K
// weight: K * M
// output: N * M
template<const uint N, const uint K, const uint M>
void gemm_seq(const float *A, const float *B, float *C) {
  for(int n = 0; n < N; ++n) {
    for(int m = 0; m < M; ++m) {
        float sum = 0.0f;
        for(int k = 0; k < K; ++k) {
            sum +=  Val(A, n, k, K) * Val(B, k, m, M);
        }
        Val(C, n, m, M) = sum;
    }
  }
}