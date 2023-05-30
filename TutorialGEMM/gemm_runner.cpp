#include "macros.h"
#include "Matrix.h"
#include "gemm_runner.h"
#include "utils.h"
#include <iostream>
#include <stdio.h>

using namespace std;

GEMMRunner::GEMMRunner(MTL::Device *device) {
    m_device_ptr = device;
    string shader_name = "gemm_naive";
    change_shader("gemm_naive");
}

GEMMRunner::~GEMMRunner() {
    if (m_device_buffer_A_ptr != nullptr) m_device_buffer_A_ptr->release();
    if (m_device_buffer_B_ptr != nullptr) m_device_buffer_B_ptr->release();
    if (m_device_buffer_X_ptr != nullptr) m_device_buffer_X_ptr->release();
    if (m_device_buffer_params_ptr != nullptr) m_device_buffer_params_ptr->release();
    if (CPU_X != nullptr) free(CPU_X);
}

void GEMMRunner::change_shader(string name) {
    if (name == shader_name) return;
    shader_name = name;
    MTL::Library *defaultLibrary = m_device_ptr->newDefaultLibrary();
    if (defaultLibrary == nullptr) {
        cout << "Failed to find the default library." << endl;
        return;
    }

    auto str = NS::String::string(shader_name.c_str(), NS::ASCIIStringEncoding);
    MTL::Function *gemmFunction = defaultLibrary->newFunction(str);
    if (gemmFunction == nullptr) {
        cout << "Failed to find the matrix multiplication shader." << endl;
        return;
    }

    NS::Error *error;
    m_MatMultiplyFunctionPSO = m_device_ptr->newComputePipelineState(gemmFunction, &error);
    if (m_MatMultiplyFunctionPSO == nullptr) {
        cout << "Failed to create the PSO: " << error << endl;
        return;
    }
    
    NS::UInteger thread_execution_width = m_MatMultiplyFunctionPSO->threadExecutionWidth();
    NS::UInteger max_total_threads_per_threadgroup = m_MatMultiplyFunctionPSO->maxTotalThreadsPerThreadgroup();
    cout << "change shader to " << shader_name << endl;
    cout << "thread execution wdith: " << thread_execution_width << endl;
    cout << "maximum threads per threadgoup: " << max_total_threads_per_threadgroup << endl;

    m_CommandQueue = m_device_ptr->newCommandQueue();
    if (m_CommandQueue == nullptr) {
        cout << "Failed to get the command queue." << endl;
        return;
    }
}

void GEMMRunner::print_performance_result(const float& microsecs, const string& name) {
    float gflops = 2e-3  * static_cast<float>(m_rows_X) * static_cast<float>(m_cols_X) * static_cast<float>(m_cols_A) / microsecs;
    cout << "[" <<  name << "]\tGFLOPS:" << gflops << "gflops\tTimeCost:" << microsecs << "ms" << std::endl;
}

void GEMMRunner::initialize_data(int rows_X, int cols_X, int inner_dim) {
    m_rows_X = rows_X; m_cols_X = cols_X; m_cols_A = inner_dim;
    m_device_buffer_A_ptr = m_device_ptr->newBuffer(m_rows_X * m_cols_A * sizeof(float), MTL::ResourceStorageModeShared);
    m_device_buffer_B_ptr = m_device_ptr->newBuffer(m_cols_A * m_cols_X * sizeof(float), MTL::ResourceStorageModeShared);
    m_device_buffer_X_ptr = m_device_ptr->newBuffer(m_rows_X * m_cols_X * sizeof(float), MTL::ResourceStorageModeShared);
    m_device_buffer_params_ptr = m_device_ptr->newBuffer(sizeof(GEMMParams), MTL::ResourceStorageModeShared);
    CPU_X = static_cast<float*>(malloc(m_rows_X * m_cols_X * sizeof(float)));

    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});

    randomize_uniform(A, -1.0f, 1.0f);
    randomize_uniform(B, -1.0f, 1.0f);
}

void GEMMRunner::run_gemm_on_gpu(const GEMMParams& p) {
    GEMMParams *params = (GEMMParams *)m_device_buffer_params_ptr->contents();
    *params = p;
    MTL::CommandBuffer *commandBuffer = m_CommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    computeEncoder->setComputePipelineState(m_MatMultiplyFunctionPSO);
    computeEncoder->setBuffer(m_device_buffer_A_ptr, 0, 0);
    computeEncoder->setBuffer(m_device_buffer_B_ptr, 0, 1);
    computeEncoder->setBuffer(m_device_buffer_X_ptr, 0, 2);
    computeEncoder->setBuffer(m_device_buffer_params_ptr, 0, 3);

    // Note: The kernel thread's 'x' position in the grid corresponds to the column index in the result matrix
    // and the 'y' position corresponds to the row index. 
    // Note that the matrix is in row-major so that the column index is the "fast" index.
    MTL::Size grid_size = MTL::Size::Make(CEIL_DIV(p.x_cols, p.BX*p.TX), CEIL_DIV(p.x_rows, p.BY*p.TY), 1); // should be the size of the grid = (x_threads, y_threads)
    MTL::Size thread_group_size = MTL::Size::Make(p.BX, p.BY, 1); 
    computeEncoder->dispatchThreadgroups(grid_size, thread_group_size);
    // cout << grid_size.width << "," << grid_size.height << endl;
    // cout << thread_group_size.width << "," << thread_group_size.height << endl;

    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void GEMMRunner::run_gemm_on_cpu(decltype(mat_multiply_naive) gemm_func) {
    Matrix<float> A(static_cast<float*>(m_device_buffer_A_ptr->contents()), {m_rows_X, m_cols_A});
    Matrix<float> B(static_cast<float*>(m_device_buffer_B_ptr->contents()), {m_cols_A, m_cols_X});
    Matrix<float> X_true(CPU_X, {m_rows_X, m_cols_X});
    gemm_func(X_true, A, B);
}

void GEMMRunner::check_results() {
    cout << "Verifying result for " << shader_name << ":";
    Matrix<float> X(static_cast<float*>(m_device_buffer_X_ptr->contents()), {m_rows_X, m_cols_X});
    Matrix<float> X_true(CPU_X, {m_rows_X, m_cols_X});
    // Show the contents if small.
    if (X.size() < 50) {
        cout << "X:\n" << X << endl;
    }
    
    const float max_allowable_error = 1e-4;
    const float max_error = assert_almost_equal_max_error(X, X_true, max_allowable_error);
    const float max_result_val = max_value(X_true);
    if (max_result_val == 0) {
        cout << "Max result magnitude was: " << max_result_val << "It is meaningless to verify unless some values are non-zero!" << endl;
        error_exit("exiting");
    }
    cout << "Passed! Max error was: " << max_error << endl;
}
