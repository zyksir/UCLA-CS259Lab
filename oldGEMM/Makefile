OPTPATH=/opt/homebrew/opt
CXX = $(OPTPATH)/llvm/bin/clang++
LDFLAGS = -L$(OPTPATH)/libomp/lib -I../metal-cpp # specify your library linking options here
CXX := /usr/local/Cellar/llvm/16.0.4/bin/clang++
LDFLAGS := -L/usr/local/Cellar/libomp/16.0.4/lib -I../metal-cpp 
FRAMEWORK := -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
CXXFLAGS := -std=c++17 -stdlib=libc++ -O3 -ffast-math -fopenmp $(LDFLAGS) -fno-objc-arc $(FRAMEWORK) -g
SRCS := $(wildcard *.cpp) $(wildcard *.h)

all: main
	
mat_mul_opt1: ./mat_mul_opt1.metal
	xcrun -sdk macosx metal -c $@.metal -o $@.air
	# xcrun -sdk macosx metallib $@.air -o $@.metallib

mat_mul_opt2: ./mat_mul_opt2.metal
	xcrun -sdk macosx metal -c $@.metal -o $@.air

mat_mul_opt3: ./mat_mul_opt3.metal
	xcrun -sdk macosx metal -c $@.metal -o $@.air

mat_mul_optimized_nv: ./mat_mul_optimized_nv.metal
	xcrun -sdk macosx metal -c $@.metal -o $@.air

mat_mul_simple1: ./mat_mul_simple1.metal
	xcrun -sdk macosx metal -c $@.metal -o $@.air

src: $(SRCS)


main: mat_mul_opt1 mat_mul_opt2 mat_mul_opt3 mat_mul_optimized_nv mat_mul_simple1 src
	xcrun -sdk macosx metal-ar r default.metalar *.air
	xcrun -sdk macosx metallib default.metalar -o default.metallib
	$(CXX) $(CXXFLAGS) $(filter %.cpp, $(SRCS)) -o $@.x

clean:
	rm -rf main *.dSYM *.x *.air *.metallib

