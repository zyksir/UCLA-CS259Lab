#ifndef CNN_H_
#define CNN_H_

#include <stdexcept>
#include <string>

const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

#define CUDA_CHECK(err) cuda_check((err), __FILE__, __LINE__);
inline void cuda_check(cudaError_t err, const char* file, int line)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) + ": " + std::string(cudaGetErrorString(err)));
  }
}

// Utility function declarations.
void LoadData(const std::string& data_dir,
    float* input,
    float* weight, float* bias);
void kernel_wrapper(
    float* input,
    float* weight,
    float* bias, float* output);
void CnnSequential(
    const float* input,
    const float* weight, const float* bias,
    float* output);
int Verify(const std::string& data_dir,
    const float* output, int output_size);
#endif
