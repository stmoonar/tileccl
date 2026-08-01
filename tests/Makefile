# Build for the P2P microbenchmark.
#
#   make                     # default: sm_120a (Blackwell consumer, CUDA >= 12.8)
#   make ARCH=sm_120         # if the toolkit rejects the family-specific arch
#   make ARCH=sm_90a         # Hopper, for cross-checking the TMA path
#
# sm_120a is the architecture-specific variant; it is required for some
# Blackwell-only instructions and is a safe default here since the resulting
# cubin only has to run on the sm_120 pair under test.

NVCC     ?= nvcc
ARCH     ?= sm_120a
NVCCFLAGS := -O3 -std=c++17 -lineinfo -arch=$(ARCH)

TARGET := p2p_ce_vs_tma

.PHONY: all clean ptx

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Handy when checking that cp.async.bulk actually survives to the final PTX.
ptx: $(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $(TARGET).ptx $<

clean:
	rm -f $(TARGET) $(TARGET).exe $(TARGET).ptx
