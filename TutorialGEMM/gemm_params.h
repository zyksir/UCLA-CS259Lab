#pragma once

using uint = unsigned int;
class GEMMParams {
public:
    uint N; // Number of rows in X
    uint M; // Number of columns in X
    uint P; // Number of columsn in A = number of rows in B
    uint BX; // threads_per_group
    uint BY; // threads_per_group
    uint TX;
    uint TY;
    GEMMParams(uint row_dim_x, uint col_dim_x, uint inner_dim): 
        N(row_dim_x), P(col_dim_x), M(inner_dim), BX(8), BY(8), TX(1), TY(1) { }
    GEMMParams(uint row_dim_x, uint col_dim_x, uint inner_dim, uint BX, uint BY): 
        N(row_dim_x), P(col_dim_x), M(inner_dim), BX(BX), BY(BY), TX(1), TY(1) { }
    GEMMParams(uint row_dim_x, uint col_dim_x, uint inner_dim, uint BX, uint BY, uint TX, uint TY): 
        N(row_dim_x), P(col_dim_x), M(inner_dim), BX(BX), BY(BY), TX(TX), TY(TY) { }
};