#pragma once

/******** input/output size ***********/
// Nx: Width, Ny: Height
#define Nx 224
#define Ny 224
#define Kx 3
#define Ky 3
#define Ni 64
#define Nn 64
#define Sx 1
#define Sy 1

constexpr int NyPAD = (Ny + Ky - 1);
constexpr int NxPAD = (Nx + Kx - 1);
constexpr int NySCL = (Ny/Sy);
constexpr int NxSCL = (Nx/Sx);
/******** input/output size end ***********/

/******** CUDA params ***********/
constexpr int BLOCK_CHANNEL = 4;
#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define BLOCKSIZEZ 1
#define GRIDDIMZ 64

constexpr int GRIDDIMX = (Nx / BLOCKSIZEX);
constexpr int GRIDDIMY = (Ny / BLOCKSIZEY);
/******** CUDA params end ***********/

#define input(ni, h, w) input[(ni)*NyPAD*NxPAD + (h)*NxPAD + (w)]
#define output(nn, h, w) output[(nn)*NySCL*NxSCL + (h)*NxSCL + (w)]
#define weight(nn, ni, p, q) weight[(nn)*Ni*Ky*Kx + (ni)*Ky*Kx + (p)*Kx + (q)]
