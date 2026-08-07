# comm_comp: CE/GEMM interference + AG+GEMM granularity + remote-epilogue
# microbenchmarks for Hopper (4xH800 NVLink).
#
#   make               # sm_90a (required: WGMMA + TMA need the 'a' suffix)
#   make -j3           # the three CUTLASS TUs compile independently
#
# Needs CUDA >= 12.4 (CUTLASS 4.x); tested against the CUDA 12.9 toolkit.
# CUTLASS comes from the repo submodule (git submodule update --init).

NVCC    ?= nvcc
ARCH    ?= sm_90a
CUTLASS := ../third_party/cutlass

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -DNDEBUG \
             --expt-relaxed-constexpr \
             -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include \
             --threads 4 \
             -Xcompiler -Wno-psabi

TARGETS := exp1_ce_interference exp2_ag_gemm_granularity exp3_epilogue_remote \
           exp3_gemm_rs_fused

.PHONY: all clean

all: $(TARGETS)

exp1_ce_interference: exp1_ce_interference.cu common.cuh gemm_sm90.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

exp2_ag_gemm_granularity: exp2_ag_gemm_granularity.cu common.cuh gemm_sm90.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

exp3_epilogue_remote: exp3_epilogue_remote.cu common.cuh gemm_sm90.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# flux-faithful fused GEMM+RS: multi-process (fork + cudaIPC), Linux-only
exp3_gemm_rs_fused: exp3_gemm_rs_fused.cu common.cuh gemm_sm90.cuh \
                    rs_barrier.cuh rs_dma_sm90.cuh rs_gemm_kernel_sm90.cuh \
                    ipc_multiproc.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) $(addsuffix .exe,$(TARGETS))
