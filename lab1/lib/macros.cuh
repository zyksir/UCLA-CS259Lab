#pragma once

/******** input/output size ***********/
#ifndef BatchSize
    #define BatchSize 1
#endif
// Nx: Width, Ny: Height
#ifndef Nx
    #define Nx 224
    #define Ny 224
#endif

#ifndef Kx
    #define Kx 3
    #define Ky 3
#endif

#ifndef Sx
    #define Sx 1
    #define Sy 1
#endif

#ifndef Ni
    #define Ni 64
    #define Nn 64
#endif

constexpr int NyPAD = (Ny + Ky - 1);
constexpr int NxPAD = (Nx + Kx - 1);
constexpr int NySCL = (Ny/Sy);
constexpr int NxSCL = (Nx/Sx);
/******** input/output size end ***********/

/******** CUDA params ***********/
#ifndef BLOCKSIZEX
    #define BLOCKSIZEX 16   
#endif

#ifndef BLOCKSIZEY
    #define BLOCKSIZEY 16   
#endif

#ifndef BLOCK_CHANNEL
    #define BLOCK_CHANNEL 4
    #define GRIDDIMZ 64
#endif

#ifndef BLOCKSIZEZ
    #define BLOCKSIZEZ 1
#endif
/******** CUDA params end ***********/
