# Build for the signalling microbenchmark.
#
#   make                     # default: sm_120a (Blackwell consumer, CUDA >= 12.8)
#   make ARCH=sm_120         # if the toolkit rejects the family-specific arch
#   make ARCH=sm_90a         # Hopper (H20)

NVCC     ?= nvcc
ARCH     ?= sm_120a
NVCCFLAGS := -O3 -std=c++17 -lineinfo -arch=$(ARCH)

TARGETS := signal_fanin

.PHONY: all clean

all: $(TARGETS)

signal_fanin: signal_fanin.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) $(addsuffix .exe,$(TARGETS))
