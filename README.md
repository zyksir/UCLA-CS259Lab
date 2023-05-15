# Project of UCLA CS259: Accelerator for M1

The repository records all the lab codes and project codes for [UCLA CS259](https://polyarch.github.io/cs259/). I implemented a CONV kernel and a GEMM kernel in lab and the code is in lab2 branch. In this branch I will use MSL(Metal Shading Language) to develop some common machine learning kernels.

## Requirement

Please install `llvm`, `libomp` using brew. Install Xcode in App store and then change its path.

```bash
brew install llvm
brew install libomp
xcode-select --install 
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```



## Reference

- [m1-gpu-cpp](https://github.com/larsgeb/m1-gpu-cpp/tree/main) and the blogs he wrote is the tutorial for beginners about how to write MSL in cpp.
- 