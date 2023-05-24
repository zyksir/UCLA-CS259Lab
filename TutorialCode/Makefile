CXX = /usr/local/Cellar/llvm/16.0.4/bin/clang++
LDFLAGS = -L/usr/local/Cellar/libomp/16.0.3/lib -I../metal-cpp # specify your library linking options here
FRAMEWORK = -framework Metal -framework Foundation -framework MetalKit
CXXFLAGS = -std=c++17 -stdlib=libc++ -O3 -fopenmp $(LDFLAGS) -fno-objc-arc $(FRAMEWORK) -g

main: ./*.cpp ./*.hpp ./*.metal
	$(CXX) $(CXXFLAGS) $(filter %.cpp, $^) -o $@.x
	xcrun -sdk macosx metal -c ops.metal -o MyLibrary.air
	xcrun -sdk macosx metallib MyLibrary.air -o ops.metallib

all:
	main

clean:
	rm -rf main *.dSYM *.x *.air *.metallib

