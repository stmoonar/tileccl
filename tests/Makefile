# Build for the P2P microbenchmarks.
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

TARGETS := p2p_ce_vs_tma pk_bw_sweep interference_matrix sync_cost intra_sm_matrix

.PHONY: all clean ptx

all: $(TARGETS)

p2p_ce_vs_tma: p2p_ce_vs_tma.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

pk_bw_sweep: pk_bw_sweep.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

interference_matrix: interference_matrix.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

sync_cost: sync_cost.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

intra_sm_matrix: intra_sm_matrix.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Handy when checking that cp.async.bulk actually survives to the final PTX.
ptx: p2p_ce_vs_tma.cu pk_bw_sweep.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o p2p_ce_vs_tma.ptx p2p_ce_vs_tma.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o pk_bw_sweep.ptx pk_bw_sweep.cu

clean:
	rm -f $(TARGETS) $(addsuffix .exe,$(TARGETS)) $(addsuffix .ptx,$(TARGETS))
