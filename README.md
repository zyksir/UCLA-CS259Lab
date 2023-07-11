# Task1: Convolution/GEMM on GPU

## [Task Description](https://polyarch.github.io/cs259/05-miniproj-1/)

这个 branch 的主要目的是为了优化 GEMM 和 CONV 两个算子。而这篇文档则是来回答以下这个问题：
1. 对于 GEMM 和 CONV 优化，我到底做了什么？
2. 为什么我要做这些优化？
3. 这些优化起到了什么样的效果？

## Step1. Start from a naive version & Cublas
[conv source code](./src/kernels/conv_01_naive.cuh), [gemm source code](./src/kernels/gemm_01_naive.cuh)。

另外要注意的一点是，Cublas 的实现中，使用的是列存储(column major)，而我们使用的是行存储(row major)。因此调用 API 的时候需要做一个小小的改动。

## Step2. Coalescing
[conv source code](./src/kernels/conv_02_coalescing.cuh), [gemm source code](./src/kernels/gemm_02_coalescing.cuh).

使用 `ncu --set full ./conv1 > ncu_conv1_report.prof` 可以看到几个 kernel 的分析。我认为这一步最重要的分析 Memory Access Pattern, 每次尽量访问连续的数据。[this picture](https://siboehm.com/assets/img/CUDA-MMM/Naive_kernel_improved_access.png) 揭示了优化前后 Memory Access Pattern 的差异。

## Step3. Shared Memory
[conv source code](./src/kernels/conv_04_shared_memory.cuh), [gemm source code](./src/kernels/gemm_03_shared_memory.cuh).

这个步骤的本质是使用 Cache。

- ps: 注意在 conv 上先使用 block tiling，再使用 shared memory 优化才能看出每一步的效果。这是因为分析 prof 的结果显示 conv 的计算量更先成为瓶颈。

## Step4. Block Tiling
[conv source code](./src/kernels/conv_03_block_tiling.cuh), [gemm source code](./src/kernels/gemm_04_block_tiling.cuh).

这个步骤的本质是提升 Arithmetic Intensity. 

## Step5. Vectorize

[conv source code](./src/kernels/conv_05_vectorize.cuh), [gemm source code](./src/kernels/conv_05_vectorize.cuh).

这个步骤的本质也是提升 Arithmetic Intensity. 我们使用 float4/float2 来优化把多个 load 合并成一个 load。

## Step6: Future Work Warp Tiling / Double Buffer / Prefetch

Warp Tiling: 
- Main Idea: Since each warp contains 32 threads and the memory accesses to the same memory address in shared memory within the same warp can be coalesced, we introduce warp tiling. 

Double Buffer:
- Main Idea: 引入两个 shared memory cache, 在一个 cache 计算的时候去填充另一个 Cache，从而减少一半的 Sync。


## How to Run
```bash
make all
./gemm1
./gemm2
./conv1
./conv2
```

## reference
- [reference code for gemm code](https://github.com/siboehm/SGEMM_CUDA/tree/master/src), [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), [Optimizing-SGEMM-on-NVIDIA-Turing-GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)
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