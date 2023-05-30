#pragma once

using uint = unsigned int;
class GEMMParams {
public:
    uint x_rows; // Number of rows in X
    uint x_cols; // Number of columns in X
    uint x_inner; // Number of columsn in A = number of rows in B
    uint BX; // threads_per_group
    uint BY; // threads_per_group
    uint TX;
    uint TY;
    GEMMParams(uint row_dim_x, uint col_dim_x, uint inner_dim): 
        x_rows(row_dim_x), x_cols(col_dim_x), x_inner(inner_dim), BX(8), BY(8), TX(1), TY(1) { }
    GEMMParams(uint row_dim_x, uint col_dim_x, uint inner_dim, uint BX, uint BY): 
        x_rows(row_dim_x), x_cols(col_dim_x), x_inner(inner_dim), BX(BX), BY(BY), TX(1), TY(1) { }
    GEMMParams(uint row_dim_x, uint col_dim_x, uint inner_dim, uint BX, uint BY, uint TX, uint TY): 
        x_rows(row_dim_x), x_cols(col_dim_x), x_inner(inner_dim), BX(BX), BY(BY), TX(TX), TY(TY) { }
};