CXX = nvcc # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS) # -g -G # --ptxas-options=-v
LIBS = src/lib/*

conv1: src/conv_runner.cu src/kernels/conv* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNxx=224 -DNyy=224 -DNii=64 -DNnn=64 -DBatchSize=16

conv2: src/conv_runner.cu src/kernels/conv* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNxx=14 -DNyy=14 -DNii=512 -DNnn=512 -DBatchSize=16

conv3: src/conv_runner.cu src/kernels/conv* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNxx=224 -DNyy=224 -DNii=64 -DNnn=64 -DBatchSize=1

conv4: src/conv_runner.cu src/kernels/conv* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNxx=14 -DNyy=14 -DNii=512 -DNnn=512 -DBatchSize=1

gemm1: src/gemm_runner.cu src/kernels/gemm* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNii=4096 -DNnn=1024 -DBatchSize=16

gemm2: src/gemm_runner.cu src/kernels/gemm* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNii=25088 -DNnn=4096 -DBatchSize=16

all: conv1 conv2 gemm1 gemm2

clean:
	$(RM) conv1 conv2 gemm1 gemm2
