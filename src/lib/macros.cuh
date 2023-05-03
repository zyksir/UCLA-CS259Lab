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


#define Val(matrix, x, y, Y) matrix[(x)*Y+(y)]
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))