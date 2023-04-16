CXX = nvcc # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS)
SRCS = lib/utils.h lib/utils.cpp lib/macros.cuh

conv1: conv.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNx=224 -DNy=224 -DNi=64 -DNn=64 -DBatchSize=16

conv2: conv.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNx=14 -DNy=14 -DNi=512 -DNn=512 -DBLOCKSIZEX=14 -DBLOCKSIZEY=14 -DBatchSize=16

fc1: fc.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=4096 -DNn=1024

fc2: fc.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=25088 -DNn=4096

gemm1: gemm.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=4096 -DNn=1024 -DBatchSize=16 -DBLOCKSIZEZ=64 -DBLOCKSIZEY=8 -DBLOCKSIZEX=4

gemm2: gemm.cu $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o %.cpp, $^) -DNi=25088 -DNn=4096 -DBatchSize=16 -DBLOCKSIZEZ=64 -DBLOCKSIZEY=8 -DBLOCKSIZEX=4

all: fc1 fc2 conv1 conv2

clean:
	$(RM) fc1 fc2 conv1 conv2
