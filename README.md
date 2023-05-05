# Task2: Performance Model of Convolution/GEMM on GPU

## [Task Description](https://polyarch.github.io/cs259/05-miniproj-2/)

This project aims to develop a model of GPU performance for the kernels. That is, given the problem size and tile sizes, we need to predict the performance.


## How to Run
```bash
make all
./gemm1
./gemm2
./conv1
./conv2
```

## reference
- [reference code for gemm code](https://github.com/siboehm/SGEMM_CUDA/tree/master/src), [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- useful commands in these lab
```bash
# general a set of useful information of the program
ncu --set full ./vectorAdd
# check stall reasons
ncu --metrics "regex:.*smsp__pcsamp_warps_issue.*" ./vectorAdd


# check which line of the code cause the illegal memory access
cuda-memcheck ./conv
# check race condition
cuda-memcheck --tool racecheck --racecheck-report analysis gemm1

```