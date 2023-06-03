OPTPATH=/opt/homebrew/opt
CXX = $(OPTPATH)/llvm/bin/clang++
LDFLAGS = -L$(OPTPATH)/libomp/lib -I../metal-cpp # specify your library linking options here
CXX := /usr/local/Cellar/llvm/16.0.4/bin/clang++
LDFLAGS := -I../metal-cpp # -L/usr/local/Cellar/libomp/16.0.4/lib 
FRAMEWORK := -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
CXXFLAGS := -std=c++17 -stdlib=libc++ -O3 -ffast-math -fopenmp $(LDFLAGS) -fno-objc-arc $(FRAMEWORK) -g
SRCS := $(wildcard *.cpp) $(wildcard *.h) $(wildcard *.hpp)

all: main
	
gemm_naive: ./gemm_naive.metal
	xcrun -sdk macosx metal -O3 -c $@.metal -o $@.air

gemm_shared: ./gemm_shared.metal
	xcrun -sdk macosx metal -O3 -c $@.metal -o $@.air

gemm_opt: ./gemm_opt.metal
	xcrun -sdk macosx metal -O3 -c $@.metal -o $@.air

src: $(SRCS)

main: gemm_naive src gemm_shared gemm_opt
	xcrun -sdk macosx metal-ar r default.metalar *.air
	xcrun -sdk macosx metallib default.metalar -o default.metallib
	$(CXX) $(CXXFLAGS) $(filter %.cpp, $(SRCS)) -o $@.x

clean:
	rm -rf main *.dSYM *.x *.air *.metallib

