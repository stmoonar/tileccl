# Build for the peer-L2 probe.
#
#   make                     # default: sm_90 (Hopper, the NVLink box)
#   make ARCH=sm_120         # the local Blackwell PCIe pair
#   make ARCH=sm_90a         # also fine; nothing here needs the 'a' variant
#
# Two binaries from one source. They differ only in the L1 policy for ordinary
# global loads, which is the one place where remote-data caching behaviour is
# genuinely in question:
#
#   peer_l2_probe        default   (ld.global -> may allocate in L1)
#   peer_l2_probe_nol1   -dlcm=cg  (L1 bypassed, L2/beyond only)
#
# Run both and the L1 contribution separates from everything else. The L2
# conclusion is read off the _nol1 build; the default build is what a real
# kernel would do.

NVCC     ?= nvcc
ARCH     ?= sm_90
NVCCFLAGS := -O3 -std=c++17 -lineinfo -arch=$(ARCH)

TARGETS := peer_l2_probe peer_l2_probe_nol1

.PHONY: all clean ptx

all: $(TARGETS)

peer_l2_probe: peer_l2_probe.cu
	$(NVCC) $(NVCCFLAGS) -DBUILD_TAG='"ca"' -o $@ $<

peer_l2_probe_nol1: peer_l2_probe.cu
	$(NVCC) $(NVCCFLAGS) -Xptxas -dlcm=cg -DBUILD_TAG='"cg"' -o $@ $<

# Confirms the loads are plain `ld.global.v4.u32` and not `ld.global.nc`
# (the read-only path), which would change what is being measured.
ptx: peer_l2_probe.cu
	$(NVCC) $(NVCCFLAGS) -DBUILD_TAG='"ca"' -ptx -o peer_l2_probe.ptx $<
	grep -c 'ld.global.nc' peer_l2_probe.ptx || echo "no ld.global.nc: good"

# -dlcm is a ptxas decision, so it is invisible in PTX; the two builds diverge
# only in SASS. Expect the LDG in the read loop to differ between these dumps.
sass: $(TARGETS)
	cuobjdump -sass peer_l2_probe      > peer_l2_probe.sass
	cuobjdump -sass peer_l2_probe_nol1 > peer_l2_probe_nol1.sass
	@echo "diff the stream_read_kernel bodies in *.sass"

clean:
	rm -f $(TARGETS) $(addsuffix .exe,$(TARGETS)) *.ptx *.sass
