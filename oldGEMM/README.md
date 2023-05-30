# GEMM with Metal in C++

## Description of the included code

[This repo](https://github.com/bkvogel/metal_performance_testing) includes the following matrix multiplication shaders:

- [mat_mul_simple1](./mat_mul_simple1.metal): The most basic GPU implementation possible. This is essentially the same code as the inner loop of the CPU version. It simply computes the dot product over the inner dimension for its current thread. We will refer to this as the "naive" GPU implementation. Arbitrary matrix sizes are supported.
- [mat_mul_optimized_nv](./mat_mul_optimized_nv.metal): This version uses shared threadgroup memory with a tiled algorithm. I directly ported it to Metal from the corresponding CUDA kernel in NVIDIA's cuda-samples.
- [mat_mul_opt1](./mat_mul_opt1.metal): I wrote this to try to improve upon the naive version by giving each thread more work to do. Since each thread does its own tiling (using small 4x4 sub-matrices), it does not use shared threadgroup memory or synchronization barriers. Matrix sizes must be an integer multiple of 4. This also provides an example of using Metal's float4x4 matrix data type to simplify the shader code. Despite the simplicity, it is capable of over 3 TFLOPS on M1 MAx.
- [mat_mul_opt2](./mat_mul_opt2.metal): This is the current fastest shader in this repo. It uses the same method as mat_mul_opt1 but doubles the work performed by each thread by computing an 8x4 sub-matrix result of X. Its peak performance is around 3.8 TFLOPS on an M1 Max. The matrix dimensions must be an integer multiple of 4 or 8 depending on the dimension.
I also include the Matrix tensor (multidimensional array) class and a subset of the tensor Utilities from my old Kumozu framework. It contains CPU implementations of matrix multiplication including BLAS and some other simple tensor operations. This is only included because it simplified writing the examples. In your own code, you could replace this with your favorite tensoor library, provided that it is also capable of initilizing a tensor given an externally supplied float pointer.

The MatrixMultiplier class takes one of the above kernel names, initializes the shared memory matrices to be multiplied, and can then perform CPU/GPU matrix multiplication and a few other operations needed for the benchmarking experiments below.

The main.cpp file contains a function to run each experiment, which performs the benchmarking.

Note: This repo uses C++ and Metal only. If you are simply looking for Metal matrix multiplication shader examples to use in your Swift code, I suspect these will work (i.e., the .metal files), but you will then need to write your own Swift code to call them.

I have also included Metal-cpp in this repository (in the metal-cpp folder) so that you will be able to build and run the examples without having to download and install it as an additional step. Note that the license allows redistribution.

