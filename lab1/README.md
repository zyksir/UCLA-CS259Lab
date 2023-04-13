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