SRCS = lib/macros.cuh lib/cnn.h lib/cnn.cu lib/wrapper.cu lib/main.cu cnn_gpu.cu

REQUIRED_FILES = cnn.h params.sh lab3-report.pdf
export KERNEL ?= cnn_gpu.cu

test: cnn $(KERNEL)
	. ./params.sh; ./$<

test-seq:
	@$(MAKE) --no-print-directory test SEQUENTIAL=

conv_seq: lib/macros.cuh lib/conv_seq.h lib/conv_gpu.h lib/conv_gpu.cu lib/conv_seq.cu
	$(CXX) $(CXXFLAGS) -o $@ lib/conv_seq.cu lib/conv_gpu.cu

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cu %.a %.o, $^) $(LDFLAGS)

clean:
	$(RM) cnn

CXX = nvcc # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS)
