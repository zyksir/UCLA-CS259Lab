#pragma once

/******** input/output size ***********/
#ifndef BatchSize
    #define BatchSize 1
#endif
// Nx: Width, Ny: Height
#ifndef Nxx
    #define Nxx 224
    #define Nyy 224
#endif

#ifndef Kxx
    #define Kxx 3
    #define Kyy 3
#endif

#ifndef Sxx
    #define Sxx 1
    #define Syy 1
#endif

#ifndef Nii
    #define Nii 64
    #define Nnn 64
#endif
/******** input/output size end ***********/

#define Val(matrix, x, y, Y) matrix[(x)*Y+(y)]
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define Val4D(matrix, a, b, c, d, B, C, D) matrix[(((a)*B + b)*C + c)*D + d]