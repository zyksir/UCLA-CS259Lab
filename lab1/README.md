# Task1: CUDA Version of Conv2D

Implement and evaluate a CUDA version of “Conv2D” convolution kernel (which is a 2D convolution + extended in the channel dimension) kernel and “classifier” (i.e. a fully connected layer / matrix-vector multiply). Use these parameters from VGG:
- Conv1: Nx=224 Ny=224 Kx=3 Ky=3 Ni=64 Nn=64 (stride=1)
- Conv2: Nx=14 Ny=14 Kx=3 Ky=3 Ni=512 Nn=512 (stride=1)
- Class1: Ni=25088 Nn=4096
- Class2: Ni=4096 Nn=1024

Param Definitions:

- Ni/Nn – Number of input/output channels/feature-maps
- Nx/Ny – Image/feature-map width/height
- Kx/Ky – Kernel size

sequence code can be refered from [here](https://github.com/PolyArch/fp-diannao).
The starter code of cnn can get from [ucla cs-133-22spring lab](https://github.com/UCLA-VAST/cs-133-22)

## Appendix 1: GPU Architecture
output of running `./Samples/1_Utilities/deviceQuery/deviceQuery
`.
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
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.7, CUDA Runtime Version = 11.7, NumDevs = 1
Result = PASS
```