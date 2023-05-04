#pragma once


// Sequential CNN implementation
// input: B * Ni * NyPAD * NxPAD
// weight: Nn * Ni * Ky * Kx
// output: BatchSize * Nn * NySCL * NxSCL
template<const uint B, const uint Ni, const uint Nn, const uint Nx, const uint Ny,
  const uint Kx, const uint Ky, 
  const uint Sx=1, const uint Sy=1, 
  const uint NyPAD=Ny+Ky-1, const uint NxPAD=Nx+Kx-1, 
  const uint NySCL=Ny/Sy, const uint NxSCL=Nx/Sx
>
void conv_seq(const float *input,
    const float *weight,
    float *output) {

  for(int b = 0; b < BatchSize; ++b) {
    for(int nn = 0; nn < Nn; ++nn) {
      for(int ny = 0; ny < Ny; ny += Sy) {
        for(int nx = 0; nx < Nx; nx += Sx) {
          int xout = nx / Sx;
          int yout = ny / Sy;
          float sum = 0.0f;

          for(int ni = 0; ni < Ni; ++ni) {
            for(int ky = 0; ky < Ky; ++ky) {
              for(int kx = 0; kx < Kx; ++kx) {
                sum += Val4D(weight, nn, ni, ky, kx, Ni, Ky, Kx) * Val4D(input, b, ni, ny+ky, nx+kx, Ni, NyPAD, NxPAD);
                // sum += weight(nn, ni, ky, kx) * input(b, ni, ny+ky, nx+kx);
              }
            }
          }

          // Perform Relu
          Val4D(output, b, nn, yout, xout, Nn, NySCL, NxSCL) = max(0.0f, sum);
        }
      }
    }
  }
}