#pragma once

/******** input/output size ***********/
// Nx: Width, Ny: Height
#ifndef Nx
#define Nx 224
#define Ny 224
#define Kx 3
#define Ky 3
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
#define BLOCK_CHANNEL 4
#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define GRIDDIMZ 64
#endif

#define BLOCKSIZEZ 1
/******** CUDA params end ***********/
