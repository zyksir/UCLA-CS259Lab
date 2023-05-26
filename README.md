# Project of UCLA CS259: Accelerator for M1

The repository records all the lab codes and project codes for [UCLA CS259](https://polyarch.github.io/cs259/). I implemented a CONV kernel and a GEMM kernel in lab and the code is in lab2 branch. In this branch I will use MSL(Metal Shading Language) to develop some common machine learning kernels.

## Prerequisites

### Step1. Install necessary compilation tools

Please install `llvm`, `libomp` using brew. Install Xcode in App store and then change its path.

First, try to run `xcrun metal`.If you run into the following error: `xcrun: error: unable to find utility "metal", not a developer tool or in PATH`. On you local machine, install XCode from App store; On aws, you need to download xcode.xip and xip it using `xip -x Xcode11.xip` under `/Application` dir. Last, run the following command to fix it:

```bash
xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

Second, install `llvm` and `libomp`.

```bash
brew install llvm
brew install libomp
```

By now, you should be able to compile for `TutorialCode` and `TutorialGEMM`. You can go into `TutorialGEMM` and run `cd TutorialGEMM && make main && ./main.x`. The code should run without error.

### Step2. Install python dependencies

All the other environment is in `environment.yaml` and we can use conda to install it. We still need to install pytorch and pytest for pytorch. If you mac has M1 chip, make sure you install preview version of pytorch from [here](https://pytorch.org/get-started/locally/); if the mac is x86, you should install stable version of pytorch. Then, install `pytest-pytorch`.

```bash
conda env create --file environment.yaml
conda activate dlsys-needle-m1
# this is the stable version since I am x86 chip
pip3 install torch torchvision torchaudio
# for M1 chip, this is the command I found in the website above
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu 
pip3 install pytest-pytorch
```

Now, you are good to go into next Step

## Step up

### Compile

Run `make` can compile the project.

### Check Correctness

Run `pytest` can check the correctness of your implementation. The whole `pytest` takes lots of time and these commands should work.

```bash
pytest tests/test_ndarray.py
pytest tests/test_nd_backend.py
pytest tests/test_sequence_models.py
```

### Check Performance

Run this command should check the performance for difference operations. **For Now, I have only implemented test for GEMM. Most code is copied from tests.**

```bash
python experiments/run_tester.py
```

### TODO List

These are future work to do:

- add test for more operations. It's easy, just modify `tester.py` file and add more functions. For example, we can check conv performance.

- improve performace for GEMM operation. The most hard one. For now, a straightforward way is to combine shared memory and tiling.

- try to implemetation more operations. For now, they use GEMM to implemetation conv. Maybe a new way to implemet conv will improve the performance.

## Reference

- [m1-gpu-cpp](https://github.com/larsgeb/m1-gpu-cpp/tree/main) and the blogs he wrote is the tutorial for beginners about how to write MSL in cpp.
- To create M1 instance in AWS, follow [this instruction](https://aws.amazon.com/blogs/aws/use-amazon-ec2-m1-mac-instances-to-build-test-macos-ios-ipados-tvos-and-watchos-apps/)
- [GEMM example](https://github.com/bkvogel/metal_performance_testing) has can naive implemetation of optimizations.
- [dlsys-needle-m1](https://github.com/wenjunsun/dlsys-needle-m1/) implements the framework of a python library which can use M1. With this framework, all we need to is add tester for performance and add metal code.
