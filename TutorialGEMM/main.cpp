#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <chrono>
#include <vector>

#include "gemm_runner.h"

using namespace std;

// please make sure this count is larget than 100
static const int loop_count = 10;

int main() {
    float microsec_per_call;
    vector< tuple<uint, uint, uint> > matrix_size_vec = {
        {1024, 1024, 1024}, 
        // {64, 64, 64}, 
        // {72, 72, 72}, 
        // {72, 73, 74}, 
        // {74, 73, 72}, 
        // {128, 128, 128},
        // {64, 4096, 1024}
    };
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    for(const auto [rows_X, cols_X, inner_dim] : matrix_size_vec) {
        cout << "------------------------------" << endl;
        cout << "Problem Size" << rows_X << "\t" << cols_X << "\t" << inner_dim << endl;
        GEMMRunner runner(device);
        runner.initialize_data(rows_X, cols_X, inner_dim);

        // microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_cpu(mat_multiply_naive); });
        // runner.print_performance_result(microsec_per_call, "cpu naive");

        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_cpu(mat_multiply_blas); });
        runner.print_performance_result(microsec_per_call, "cpu blas");

        cout << "******************************" << endl;
        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_cpu(mat_multiply_vdsp); });
        runner.print_performance_result(microsec_per_call, "cpu vDSP");
        cout << "******************************" << endl;

        cout << "******************************" << endl;
        runner.change_shader("gemm_naive");
        GEMMParams naive_params{rows_X, cols_X, inner_dim};
        runner.run_gemm_on_gpu(naive_params); 
        runner.check_results();
        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_gpu(naive_params); });
        runner.print_performance_result(microsec_per_call, "m1 naive");
        cout << "******************************" << endl;

        cout << "******************************" << endl;
        runner.change_shader("gemm_shared");
        runner.run_gemm_on_gpu(naive_params); 
        runner.check_results();
        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_gpu(naive_params); });
        runner.print_performance_result(microsec_per_call, "m1 shared");
        cout << "******************************" << endl;

        // using the optimal setting by others
        cout << "******************************" << endl;
        runner.change_shader("gemm_opt");
        GEMMParams opt_params{rows_X, cols_X, inner_dim, 16, 8, 4, 8};
        runner.run_gemm_on_gpu(opt_params); 
        runner.check_results();
        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_gpu(opt_params); });
        runner.print_performance_result(microsec_per_call, "m1 opt");
        cout << "******************************" << endl;

        // using the optimal setting by others
        cout << "******************************" << endl;
        runner.change_shader("gemm_opt2");
        GEMMParams opt2_params{rows_X, cols_X, inner_dim, 16, 8, 4, 8};
        runner.run_gemm_on_gpu(opt2_params); 
        runner.check_results();
        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_gpu(opt2_params); });
        runner.print_performance_result(microsec_per_call, "m1 opt");
        cout << "******************************" << endl;

        // using the optimal setting by others
        cout << "******************************" << endl;
        runner.change_shader("gemm_tiling");
        GEMMParams tiling_params{rows_X, cols_X, inner_dim, 16, 8, 4, 8};
        runner.run_gemm_on_gpu(tiling_params); 
        runner.check_results();
        microsec_per_call = benchmark(loop_count, [&]() { runner.run_gemm_on_gpu(tiling_params); });
        runner.print_performance_result(microsec_per_call, "m1 tiling");
        cout << "******************************" << endl;
    }
    device->release();
}
