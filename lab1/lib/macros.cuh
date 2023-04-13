#pragma once

#define kNum 256
#define kInImSize 228
#define kImSize 224
#define kOutImSize 112
#define kKernel 5

/******** config params ***********/
constexpr int BLOCK_CHANNEL = 4;
#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define BLOCKSIZEZ 1
#define GRIDDIMZ 64
/******** config end ***********/

constexpr int GRIDDIMX = (kOutImSize / BLOCKSIZEX);
constexpr int GRIDDIMY = (kOutImSize / BLOCKSIZEY);

#define input(i, h, w) input[(i)*kInImSize * kInImSize + (h)*kInImSize + (w)]
#define output(i, h, w) output[(i)*kOutImSize * kOutImSize + (h)*kOutImSize + (w)]
#define weight(i, j, p, q) weight[(i)*kNum * kKernel * kKernel + (j)*kKernel * kKernel + (p)*kKernel + (q)]
