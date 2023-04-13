#pragma once
__global__ void cnn_gpu(float* input,
    float* weight,
    float* bias, float* output);
