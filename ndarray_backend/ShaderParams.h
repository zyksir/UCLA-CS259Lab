#pragma once

class MatMulParams
{
public:
    unsigned int N; // Number of rows in X
    unsigned int P; // Number of columns in X
    unsigned int M; // Number of columsn in A = number of rows in B
    unsigned int tN;
    unsigned int tP;
    unsigned int blockSize = 8; // threads_per_group
    MatMulParams(unsigned int row_dim_x, unsigned int col_dim_x, unsigned int inner_dim): 
        N(row_dim_x), P(col_dim_x), M(inner_dim), tN(1), tP(1) { }
    MatMulParams(unsigned int row_dim_x, unsigned int col_dim_x, unsigned int inner_dim, unsigned int TN, unsigned TP): 
        N(row_dim_x), P(col_dim_x), M(inner_dim), tN(TN), tP(TP) { }
};