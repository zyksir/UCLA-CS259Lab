CXX = nvcc # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS)
LIBS = src/lib/*

conv1: src/conv.cu $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNx=224 -DNy=224 -DNi=64 -DNn=64 -DBatchSize=16

conv2: src/conv.cu $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNx=14 -DNy=14 -DNi=512 -DNn=512 -DBatchSize=16

gemm1: src/gemm_runner.cu src/kernels/gemm* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=4096 -DNn=1024 -DBatchSize=16

gemm2: src/gemm_runner.cu src/kernels/gemm* $(LIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=25088 -DNn=4096 -DBatchSize=16

all: conv1 conv2 gemm1 gemm2

clean:
	$(RM) conv1 conv2 gemm1 gemm2
