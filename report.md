# Optimization of GEMM & Conv

### Introduction
- Overall Performance
```txt
[CONV1]: 
[CONV SEQ]Pass  GFLOPS:1.44013gflops    TimeCost:4.11005e+07us
[CONV CUDA NAIVE ]Pass  GFLOPS:1567.91gflops    TimeCost:37751.7us
[CONV CUDA COALESCING ]Pass     GFLOPS:1721.19gflops    TimeCost:34389.9us
[CONV CUDA BLOCK TILING ]Pass   GFLOPS:2058gflops       TimeCost:28761.6us
[CONV CUDA SHARED ]Pass GFLOPS:5239.45gflops    TimeCost:11297.6us
[GEMM2]: 
[GEMM SEQ]Pass  GFLOPS:1.68717gflops    TimeCost:1.94902e+06us
[GEMM CUDA NAIVE]Pass   GFLOPS:287.341gflops    TimeCost:11444.1us
[GEMM CUDA coalescing]Pass      GFLOPS:1385.85gflops    TimeCost:2372.83us
[GEMM CUDA NAIVE SHARED]Pass    GFLOPS:1846.76gflops    TimeCost:1780.69us
[GEMM CUDA SHARED]Pass  GFLOPS:2286.74gflops    TimeCost:1438.04us
[GEMM CUDA TILING]Pass  GFLOPS:2836.97gflops    TimeCost:1159.17us
[GEMM CUDA VECTERIZE]Pass       GFLOPS:2459.12gflops    TimeCost:1337.26us
```

- Machince Architecture
with `cd ~/cuda-samples && ./Samples/1_Utilities/deviceQuery/deviceQuery`, we can get some info
```python
MemoryBandwidth = 651 * (1024 ** 3) #B/s
MaxFLOPS = 13.8 * 1000 # GFLOPS
NumSP = 5120
MaxThreadPerSM = 2048
MaxThreadPerBlock = 1024
WarpSize = 32
L2CacheSize = 4718592 #Bytes
alpha = 1000**(-3)
DataTypeBytes = 4 # we use float
# B, Kx, Ky, Nx, Ny, Ni, Nn = 16, 3, 3, 224, 224, 64, 64
```

## Naive Model
```python
# Roofline model of naive way
# For Conv, each block caculate a batchSize of items in output
ConvOperationIntensity = 2*B*Ni*Kx*Ky/(B*Ni*Kx*Ky*2 + B) # 1 for conv1
naiveRoofline = min(MaxFLOPS, ConvOperationIntensity*MemoryBandwidth*alpha)
```

with running `ncu --set full` command, the bottleneck of both navie kernels is Memory.
- The main problem with conv is that it only use 2.3/4 info per cache line from L1TEX to L2 . 
- The Gemm mentions thread access 4 bytes per request but has 8.5 sectors per request per warp.(ps: optimal should be 4 sectors since each sector is 32 byte. 4*warpsize=4 sectors.) from L1TEX to L2 and global to L1TEX
- In addition, `ncu` mentions the kernel is too small. This can be easily upgraded by increasing thread nums per block.

## Memory Coalescing
The answer is using memory coalescing. That is, the 
- For Gemm, please refer to this image ![img](https://siboehm.com/assets/img/CUDA-MMM/Naive_kernel_improved_access.png). # 1 -> 2.5
- For Conv, we can transfer NCHW format to NHWC format to get memory coalescing. # 2.3/4 -> 3/4


## Shared Model
Before we get the naiveRoofline, we find the naive way of using shared memory cannot improve performance. After nvprof using `-m dram_read_throughput`, I find this is because the throughput doesn't increase. I think this is because the naive method has great locality if Strip is 1. Therefore, the cache locality of high and we can assume.
after run `ncu --metrics "regex:.*smsp__pcsamp_warps_issue.*" ./conv1` to see the stall reasons. One of the top reasons are "Stall MIO Throttle".


## Appendix
1. useful commands 
```bash
ncu --set full ./conv1 > ncu_conv1_report.prof
nvprof -e shared_ld_bank_conflict,shared_st_bank_conflict --metrics shared_efficiency,shared_load_transactions_per_request ./conv1
```

2. result running using `cd ~/cuda-samples && ./Samples/1_Utilities/deviceQuery/deviceQuery`, more info can be found [here](https://arxiv.org/pdf/1804.06826.pdf):
```txt
Device 0: "NVIDIA TITAN V"
  CUDA Driver Version / Runtime Version          11.7 / 11.7
  CUDA Capability Major/Minor version number:    7.0
  Total amount of global memory:                 12067 MBytes (12653035520 bytes)
  (080) Multiprocessors, (064) CUDA Cores/MP:    5120 CUDA Cores
  GPU Max Clock rate:                            1455 MHz (1.46 GHz)
  Memory Clock rate:                             850 Mhz
  Memory Bus Width:                              3072-bit
  L2 Cache Size:                                 4718592 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 7 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 175 / 0
```