#include "utils.h"
#include <functional>
#include <cuda_runtime.h>

using std::clog;
using std::endl;

template<typename T, typename KERNEL_FUNC>
class Test {
    void print_performance_result(const duration<double>& duration, const string& name) {
        uint64_t run_time_us = duration_cast<microseconds>(duration).count();
        float gflops = float_calculation_num /(run_time_us * 1e3);
        clog << "[" << kernel_name + name << "]\tGFLOPS:" << gflops << "gflops\tTimeCost:" << duration.count() << "ms" << std::endl;
    }

    bool is_diff_single_item(T a, T b) {
        return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
    }

    bool is_diff_matrix(T* a, T* b, size_t msize) {
        for(int i = 0; i < msize; ++i) {
            if (is_diff_single_item(a[i], b[i])) {
                clog << "Find different is pos " << i << ": Expected: " << a[i] << ", Get: " << b[i] << endl;
                return true;
            }
        }
        return false;
    }

public:
    Test(size_t _input_length, size_t _output_length, size_t _weight_length, size_t _float_calculation_num, const string& _kernel_name): 
		float_calculation_num(_float_calculation_num), kernel_name(_kernel_name),
		input_length(_input_length), output_length(_output_length), weight_length(_weight_length),
		input_size(_input_length*sizeof(T)), output_size(_output_length*sizeof(T)), weight_size(_weight_length*sizeof(T)) {

		input = static_cast<T*>(malloc(input_size));
		output = static_cast<T*>(malloc(output_size));
		weight = static_cast<T*>(malloc(weight_size));

		auto sta = std::chrono::steady_clock::now();
		GenerateRandomMatrix(input, input_length);
		GenerateRandomMatrix(weight, weight_length);
		std::chrono::nanoseconds rand_duration = std::chrono::steady_clock::now() - sta;
		clog << "[Generate Random Matrix]\tTimeCost:" << std::chrono::duration_cast<microseconds>(rand_duration).count() << "us" << std::endl;
    }

	~Test() {
		free(input);
		free(output);
		free(weight);
	}

  void run_seq(KERNEL_FUNC sequential_kernel) {
    auto sta = std::chrono::steady_clock::now();
    sequential_kernel(input, weight, output);
    std::chrono::nanoseconds conv_seq_duration = std::chrono::steady_clock::now() - sta;
    print_performance_result(conv_seq_duration, "SEQ");
  }

  void test_cuda(KERNEL_FUNC cuda_kernel, void(*sizefunc)(dim3&, dim3&), const string& name,
                std::function<T*(const T*)> input_xform = nullptr, std::function<T*(const T*)> weight_xform = nullptr, std::function<T*(T*)> output_xform = nullptr) {
    T* xformed_input = input_xform ? input_xform(input) : input;
    T* xformed_weight = weight_xform ? weight_xform(weight) : weight;
    T* g_input, *g_weight, *g_output;
    cudaSetDevice(0);
    cudaMalloc((T**)&g_input, input_size);
    cudaMalloc((T**)&g_weight, weight_size);
    cudaMalloc((T**)&g_output, output_size);
    cudaMemcpy(g_input, xformed_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weight, xformed_weight, weight_size, cudaMemcpyHostToDevice);

    //GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid; dim3 block;
    sizefunc(grid, block);
    std::clog << "Block dims: " << block.x << ' ' << block.y << ' ' << block.z << '\n';
    std::clog << "Grid dims: " << grid.x << ' ' << grid.y << ' ' << grid.z << '\n';
    
    cudaEventRecord(start);
    cuda_kernel<<<grid, block>>>(g_input, g_weight, g_output);
    cudaError_t status = cudaGetLastError();
    cudaEventRecord(stop);

    T* cuda_output = static_cast<T*>(malloc(output_size));
    cudaMemcpy(cuda_output, g_output, output_size, cudaMemcpyDeviceToHost);

    if(status != cudaSuccess) {
      clog << "Error: " << cudaGetErrorString(status) << endl;
      return;
    }

    //GPU measurement end
	  cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::chrono::nanoseconds gpu_duration((uint64_t)(elapsed_ms * 1e6));
    print_performance_result(gpu_duration, name);

    CUDA_CHECK(cudaDeviceSynchronize());
    float* xformed_cuda_output = output_xform ? output_xform(cuda_output) : cuda_output;
    if (is_diff_matrix(output, xformed_cuda_output, output_length)) {
      clog << "FAIL" << endl;
    } else {
      clog << "PASS" << endl;
    }

    free(cuda_output);
    if (input_xform) { free(xformed_input); }
    if (weight_xform) { free(xformed_weight); }
    if (output_xform) { free(xformed_cuda_output); }
    cudaFree(g_input);
    cudaFree(g_output);
    cudaFree(g_weight);
  }

private:
    const string kernel_name;
    const uint64_t float_calculation_num;
    const uint64_t input_length;
    const uint64_t output_length;
    const uint64_t weight_length;
    const uint64_t input_size;
    const uint64_t output_size;
    const uint64_t weight_size;
    float* input;
    float* output;
    float* weight;
};

