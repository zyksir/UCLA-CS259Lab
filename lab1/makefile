CXX = nvcc # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS)
SRCS = lib/utils.h lib/test.hpp lib/macros.cuh

conv1: conv.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNx=224 -DNy=224 -DNi=64 -DNn=64 -DBatchSize=16 -DBLOCKSIZEX=32 -DBLOCKSIZEY=32 -DBLOCKSIZEZ=64

conv2: conv.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNx=14 -DNy=14 -DNi=512 -DNn=512 -DBatchSize=16 -DBLOCKSIZEX=14 -DBLOCKSIZEY=14 -DBLOCKSIZEZ=512

fc1: fc.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=4096 -DNn=1024

fc2: fc.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=25088 -DNn=4096

gemm1: gemm.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=4096 -DNn=1024 -DBatchSize=16 -DBLOCKSIZEZ=512 -DBLOCKSIZEY=64 -DBLOCKSIZEX=16

gemm2: gemm.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=25088 -DNn=4096 -DBatchSize=16 -DBLOCKSIZEZ=512 -DBLOCKSIZEY=64 -DBLOCKSIZEX=16

all: fc1 fc2 conv1 conv2 gemm1 gemm2

clean:
	$(RM) fc1 fc2 conv1 conv2 gemm1 gemm2
