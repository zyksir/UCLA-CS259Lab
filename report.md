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
```python
alpha = 1000**(-3)
DataTypeBytes = 4
MemoryBandwidth = 651 * (1024 ** 3) #B/s
MaxFLOPS = 13.8 * 1000 # GFLOPS
L2CacheSize = 4718592 #Bytes
# B, Kx, Ky, Nx, Ny, Ni, Nn = 16, 3, 3, 224, 224, 64, 64
```

## Naive Model
```python
# Roofline model of naive way
# For Conv, each block caculate a batchSize of items in output
ConvOperationIntensity = 2*B*Ni*Kx*Ky/(B*Ni*Kx*Ky*2 + B) # 1 for conv1
naiveRoofline = min(MaxFLOPS, ConvOperationIntensity*MemoryBandwidth*alpha) # 698 GFLOPS -> 253 measured
```

## Shared Model
Before we get the naiveRoofline, we find the naive way of using shared memory cannot improve performance. After nvprof using `-m dram_read_throughput`, I find this is because the throughput doesn't increase. I think this is because the naive method has great locality if Strip is 1. Therefore, the cache locality of high and we can assume.
after run `ncu --set full ./conv1` and `ncu --metrics "regex:.*smsp__pcsamp_warps_issue.*" ./conv1` to see the stall reasons.

