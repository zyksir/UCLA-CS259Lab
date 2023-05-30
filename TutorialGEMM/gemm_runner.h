#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include "gemm_params.h"
#include "utils.h"


class GEMMRunner {
public:
    GEMMRunner(MTL::Device *device);
    ~GEMMRunner();
    void change_shader(std::string shader_name);
    void initialize_data(int rows_X, int cols_X, int inner_dim);
    
    void run_gemm_on_gpu(const GEMMParams& p);
    void run_gemm_on_cpu(decltype(mat_multiply_naive) gemm_func);
    void check_results();
    void print_performance_result(const float& microsecs, const std::string& name);

    std::string shader_name;
    
private:
    MTL::CommandQueue *m_CommandQueue;
    MTL::Device *m_device_ptr;
    MTL::ComputePipelineState *m_MatMultiplyFunctionPSO;

    MTL::Buffer *m_device_buffer_A_ptr;
    MTL::Buffer *m_device_buffer_B_ptr;
    MTL::Buffer *m_device_buffer_X_ptr;
    float* CPU_X;
    MTL::Buffer* m_device_buffer_params_ptr;    

    int m_rows_X;
    int m_cols_X;
    int m_cols_A;
};
