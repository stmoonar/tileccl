// peer_l2_probe.cu
//
// Question: when GPU A loads from memory owned by GPU B over NVLink (or PCIe)
// P2P, is that data cached in **A's** L2 -- the requester's / receiver's L2?
//
// The architectural answer is "no": NVIDIA's L2 is a *memory-side* cache. The
// slices sit behind the XBAR, paired with the memory partitions/controllers of
// the GPU that owns the DRAM, not on the core side. A peer load travels
// SM -> L1 -> XBAR -> NVLink -> (peer hub) -> peer XBAR -> peer L2 -> peer HBM,
// so the L2 that serves it is the *owner's*. A memory-side cache can only cache
// the memory it fronts, and no remote address maps to any local memory
// partition, so the requester's L2 has nothing to cache. There is no NVIDIA
// document that states this in one sentence; the support is (a) every
// architecture whitepaper since Fermi drawing L2 slices paired with memory
// controllers, (b) the absence of any inter-GPU L2 coherence protocol, which
// makes locally caching a remote line semantically unsound, and (c) independent
// measurements -- Lutz et al., "Pump Up the Volume" (SIGMOD 2020) observe that
// a small hash table does not enter the GPU's L2 over NVLink 2.0 and name
// memory-side caching as the reason; the DGX-1 "Spy in the GPU-box" side
// channel works only because remote reads contend in the *remote* GPU's L2.
//
// This binary is that claim, measured on this box, with a control. It runs the
// exact same kernel on the exact same GPU against two buffers that differ only
// in which device owns them:
//
//   target=local   buffer in the reader's own HBM   -> caching IS expected
//   target=peer    buffer in the peer's HBM         -> caching is the question
//
// Three independent instruments, in increasing directness:
//
//   mode=bw    Achieved read bandwidth vs working-set size, swept across the
//              L2 capacity. `local` must show a step: working sets below L2
//              run at cache bandwidth, above it at HBM bandwidth. That step is
//              the CONTROL -- it is the instrument proving it can see caching
//              at all. `peer` is expected to be flat at link bandwidth: a 4 MB
//              peer buffer read a thousand times is no faster than a 400 MB one.
//
//   mode=lat   Dependent-load pointer chase (128 B random cycle, one thread)
//              vs working-set size. Same logic on the latency axis, and it is
//              the instrument that separates "cached locally" from "not cached
//              but link-limited": if the requester's L2 held those lines, a
//              small peer working set would chase at local-L2 latency (~10^2
//              ns), not at link latency (~10^3 ns).
//
//   mode=wire  The direct byte count. Read an S-byte peer buffer K times and
//              ask the NVLink hardware counters how many bytes actually
//              crossed the link. No caching <=> wire bytes ~= K * S. Counters
//              come from `nvidia-smi nvlink -gt d` bracketed tightly around
//              the loop, on both the reader (RX) and the owner (TX), with an
//              idle-drift baseline subtracted. `target=local` is the control:
//              it must move ~0 bytes on the wire.
//
//   mode=once  One deterministic timed launch per (target, size), for an
//              external profiler:
//                ncu --section Nvlink --launch-skip 1 --launch-count 1 \
//                    ./peer_l2_probe --modes once --targets peer --sizes 8M
//              (NCU's "peer traffic" counters only cover PCIe-attached GPUs;
//              NVLink needs the Nvlink section/metrics.)
//
// The L1 nuance: L1 may well cache remote lines -- there are third-party
// measurements suggesting it does -- which is why this file is built a second
// time with `-Xptxas -dlcm=cg` (L1 bypassed for ordinary global loads) as
// `peer_l2_probe_nol1`. Running both separates "L1 helped" from "L2 helped",
// and the default size sweep starts below the L1 capacity so an L1 step and an
// L2 step land in different places on the x axis. It does not change the L2
// conclusion, and it does not matter much for a fused GEMM anyway: L1 is
// per-SM, small, and not shared between CTAs, so it cannot serve the cross-CTA
// reuse of a broadcast A-tile.
//
// Loads are plain `ld.global` -- the pointer is deliberately not `__restrict__`
// so nvcc does not promote them to `ld.global.nc` / the read-only path. That is
// the same instruction a GEMM operand load would use.
//
// Two orthogonal axes on top of that, because "ld.global reading peer memory"
// is one of four cells and not the one this repo leans on hardest:
//
//   --dir write   Same question for the PUSH side: does a store into peer
//                 memory touch the SENDER's L2? Same reading -- `local` must
//                 step, `peer` must not, the link must carry every byte -- but
//                 with no L1 confound, since global stores have been
//                 write-through in L1 since Volta. The link direction flips
//                 (push = sender TX / owner RX), the kernels end with a
//                 system-scope drain so the clock covers delivery rather than
//                 issue, and the result is verified ON THE OWNER: did the
//                 bytes land in its HBM, not did the sender think it sent them.
//
//   --via tma     Same questions for cp.async.bulk. This is the engine
//                 exp3_gemm_rs_fused actually uses to pull peer tiles and the
//                 one tests/p2p_ce_vs_tma uses to push them, so an answer
//                 measured on the LSU path does not automatically transfer.
//                 In particular: if TMA does not populate L1 the way ld.global
//                 demonstrably does, the L1 result above does not apply to the
//                 fused kernel at all.
//
// mode=lat stays on ld.global in both cases: a dependent-load chase has no TMA
// or store equivalent.
//
// Build:  make                      (see l2_peer_cache/Makefile)
// Run:    ./peer_l2_probe --reader 0 --owner 1
//         ./peer_l2_probe --modes bw --csv out/l2peer
//         ./run_all.sh              (everything, both builds, + packaging)

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

// Set by the Makefile: "ca" = default L1 caching, "cg" = -Xptxas -dlcm=cg.
#ifndef BUILD_TAG
#define BUILD_TAG "ca"
#endif

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      std::fprintf(stderr, "[CUDA] %s:%d: %s -> %s\n", __FILE__, __LINE__,    \
                   #expr, cudaGetErrorString(_err));                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

static uint64_t parse_bytes(const std::string& s) {
  char* end = nullptr;
  uint64_t v = std::strtoull(s.c_str(), &end, 10);
  if (end && *end) {
    switch (*end) {
      case 'k': case 'K': v <<= 10; break;
      case 'm': case 'M': v <<= 20; break;
      case 'g': case 'G': v <<= 30; break;
      default:
        std::fprintf(stderr, "bad size suffix in '%s'\n", s.c_str());
        std::exit(1);
    }
  }
  return v;
}

static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  size_t beg = 0;
  while (beg <= s.size()) {
    size_t end = s.find(',', beg);
    if (end == std::string::npos) end = s.size();
    if (end > beg) out.push_back(s.substr(beg, end - beg));
    beg = end + 1;
  }
  return out;
}

static std::string fmt_bytes(uint64_t b) {
  char buf[32];
  if (b >= (1ull << 20)) {
    const double m = (double)b / 1048576.0;
    std::snprintf(buf, sizeof(buf), (m == (double)(uint64_t)m) ? "%.0fM" : "%.2fM", m);
  } else {
    std::snprintf(buf, sizeof(buf), "%.0fK", (double)b / 1024.0);
  }
  return buf;
}

static bool has(const std::vector<std::string>& v, const char* s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

static constexpr int kLineBytes = 128;  // pointer-chase granularity
static constexpr int kMaxBlocks = 4096;

static __global__ void fill_kernel(uint32_t* p, size_t n, uint32_t salt) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += step) {
    // murmur3 finalizer: full avalanche, so a truncated or aliased read cannot
    // accidentally reproduce the right checksum.
    uint32_t h = (uint32_t)i + salt * 0x9E3779B9u;
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    p[i] = h;
  }
}

__device__ __forceinline__ uint32_t block_reduce_add(uint32_t v) {
  __shared__ uint32_t s[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarp = (blockDim.x + 31) >> 5;
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) s[wid] = v;
  __syncthreads();
  uint32_t t = 0;
  if (wid == 0) {
    t = (lane < nwarp) ? s[lane] : 0u;
    for (int o = 16; o; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
  }
  return t;  // meaningful on thread 0 only
}

// Streaming read of an n_vec-element (16 B each) buffer: `iters` 128-bit loads
// per thread, wrapping inside the buffer so the *working set* stays exactly
// n_vec*16 bytes no matter how large the grid is.
//
// Why wrap instead of a plain grid-stride loop: at a 512 KB working set a
// grid-stride loop over 500+ CTAs leaves all but a handful of CTAs with zero
// work, so the small end of the sweep would measure "8 SMs hitting L2" instead
// of "the whole GPU hitting L2" -- and the local L2 step is exactly the control
// that must not be understated. With the wrap, every thread does `iters` loads
// at every size, and the only thing that changes across the sweep is how much
// distinct memory those loads touch.
//
// The address chain (i += stride; conditional wrap) never depends on loaded
// data, so loads still pipeline; memory-level parallelism is preserved.
//
// The accumulator is block-reduced and stored so nothing can be dead-code
// eliminated.
static __global__ __launch_bounds__(1024) void stream_read_kernel(
    const uint4* p, size_t n_vec, size_t stride_mod, uint64_t iters,
    uint32_t* sink) {
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t i = tid % n_vec;
  uint4 acc = make_uint4(0, 0, 0, 0);
#pragma unroll 4
  for (uint64_t k = 0; k < iters; ++k) {
    const uint4 v = p[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    i += stride_mod;
    if (i >= n_vec) i -= n_vec;
  }
  const uint32_t s = block_reduce_add(acc.x + acc.y + acc.z + acc.w);
  if (threadIdx.x == 0) sink[blockIdx.x] = s;
}

// Streaming *write* of the same wrapped working set: `iters` 128-bit stores
// per thread, identical addressing to the read kernel so the two sweeps are
// directly comparable. Every thread stores the same constant, so the writes
// that overlap because of the wrap are race-free and the result is still
// verifiable.
//
// Global stores have been write-through in L1 since Volta, so unlike the read
// sweep there is no L1 confound here: whatever step this curve shows is L2.
//
// The trailing system-scope fence is what makes the timing honest for
// target=peer. Without it the kernel could retire with posted writes still in
// flight on the link, and the "bandwidth" would be an issue rate rather than a
// delivery rate. It costs one drain per kernel, amortised over `iters`.
static __global__ __launch_bounds__(1024) void stream_write_kernel(
    uint4* p, size_t n_vec, size_t stride_mod, uint64_t iters, uint4 val) {
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t i = tid % n_vec;
#pragma unroll 4
  for (uint64_t k = 0; k < iters; ++k) {
    p[i] = val;
    i += stride_mod;
    if (i >= n_vec) i -= n_vec;
  }
  __threadfence_system();
}

// Counts words that are not `val`. Run on whichever device OWNS the buffer, so
// for target=peer this asks the owner whether the bytes actually landed in its
// HBM -- not whether the writer thinks it sent them.
static __global__ void check_const_kernel(const uint32_t* p, size_t n,
                                          uint32_t val, uint32_t* bad) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  uint32_t c = 0;
  for (; i < n; i += step) c += (p[i] != val) ? 1u : 0u;
  const uint32_t t = block_reduce_add(c);
  if (threadIdx.x == 0 && t) atomicAdd(bad, t);
}

// ---------------------------------------------------------------------------
// TMA (cp.async.bulk) path
// ---------------------------------------------------------------------------
//
// Everything above measures the LSU path (ld.global / st.global). TMA is a
// different engine with its own request path, and it is the path this repo
// actually depends on: exp3_gemm_rs_fused pulls peer tiles with SM90_TMA_LOAD,
// and tests/p2p_ce_vs_tma pushes them with cp.async.bulk.global.shared::cta. If
// a bulk copy behaved differently -- allocated in the requester's L2, or did
// not fill L1 the way ordinary loads demonstrably do -- every conclusion drawn
// from the ld/st sweeps would have to be re-derived for it. So it gets the same
// instrument, the same working-set sweep and the same control.
//
// The PTX below is lifted from tests/p2p_ce_vs_tma.cu, where these sequences
// are already exercised on both an NVLink and a PCIe pair. Bodies are guarded
// on __CUDA_ARCH__ so the file still compiles for pre-Hopper targets; main()
// refuses --via tma there rather than launching a stub.

static constexpr int kTmaStages = 4;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)),
               "r"(count));
#else
  (void)bar; (void)count;
#endif
}

// Makes prior generic-proxy shared-memory writes (mbarrier init, SM stores into
// the staging buffer) visible to the async proxy that TMA operates in.
__device__ __forceinline__ void fence_proxy_async_shared() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_addr(bar)),
               "r"(bytes)
               : "memory");
#else
  (void)bar; (void)bytes;
#endif
}

// Branch-free try_wait: avoids duplicated PTX labels when inlined more than
// once in a single kernel.
__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* bar,
                                                  uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
      "selp.b32 %0, 1, 0, P;\n"
      "}\n"
      : "=r"(ok)
      : "r"(smem_addr(bar)), "r"(phase));
  return ok != 0;
#else
  (void)bar; (void)phase;
  return true;
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  while (!mbarrier_try_wait(bar, phase)) {
  }
}

// global -> shared::cta. `gmem_src` may be a peer device pointer.
__device__ __forceinline__ void tma_load_1d(void* smem_dst, const void* gmem_src,
                                            uint32_t bytes, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(smem_addr(smem_dst)),
      "l"(gmem_src), "r"(bytes), "r"(smem_addr(bar))
      : "memory");
#else
  (void)smem_dst; (void)gmem_src; (void)bytes; (void)bar;
#endif
}

// shared::cta -> global. `gmem_dst` may be a peer device pointer.
__device__ __forceinline__ void tma_store_1d(void* gmem_dst,
                                             const void* smem_src,
                                             uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                   "l"(gmem_dst),
               "r"(smem_addr(smem_src)), "r"(bytes)
               : "memory");
#else
  (void)gmem_dst; (void)smem_src; (void)bytes;
#endif
}

__device__ __forceinline__ void tma_store_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

// Waits until at most N bulk-store groups are outstanding END TO END, i.e. the
// writes have landed at the destination. The `.read` variant would only tell us
// the source buffer can be recycled, which for a peer target would stop the
// clock before the link work is done.
template <int N>
__device__ __forceinline__ void tma_store_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
#endif
}

// gmem -> smem bulk copies, wrapping inside an n_tiles-tile working set, so the
// working set is controlled exactly as in the ld/st sweep and the two curves
// share an x axis. One issuing thread per CTA (that is how TMA is driven in
// real code -- cute::elect_one_sync); pipeline depth, not thread count, is what
// keeps the link busy.
//
// Thread 0 reads one word out of every tile that lands: the data has to be
// consumed for the number to mean anything, and it proves the copy completed
// rather than merely having been issued.
static __global__ __launch_bounds__(1024) void tma_stream_read_kernel(
    const uint8_t* __restrict__ p, uint32_t n_tiles, uint32_t stride_tiles,
    uint64_t iters, uint32_t tile_bytes, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* buf = smem;
  uint64_t* bar =
      reinterpret_cast<uint64_t*>(smem + (size_t)kTmaStages * tile_bytes);

  if (threadIdx.x != 0) return;

#pragma unroll
  for (int s = 0; s < kTmaStages; ++s) mbarrier_init(&bar[s], 1);
  fence_proxy_async_shared();

  uint32_t t = blockIdx.x % n_tiles;
  auto issue = [&](uint32_t slot) {
    mbarrier_arrive_expect_tx(&bar[slot], tile_bytes);
    tma_load_1d(buf + (size_t)slot * tile_bytes, p + (size_t)t * tile_bytes,
                tile_bytes, &bar[slot]);
    t += stride_tiles;
    if (t >= n_tiles) t -= n_tiles;
  };

  uint32_t phase = 0, acc = 0;
  uint64_t issued = 0;
  for (; issued < (uint64_t)kTmaStages && issued < iters; ++issued)
    issue((uint32_t)(issued % kTmaStages));

  for (uint64_t done = 0; done < iters; ++done) {
    const uint32_t slot = (uint32_t)(done % kTmaStages);
    mbarrier_wait(&bar[slot], (phase >> slot) & 1u);
    phase ^= (1u << slot);
    acc += *reinterpret_cast<const uint32_t*>(buf + (size_t)slot * tile_bytes);
    if (issued < iters) {
      issue(slot);
      ++issued;
    }
  }
  sink[blockIdx.x] = acc;
}

// smem -> gmem bulk copies with the same wrap. The staging buffer is filled
// once with the verification constant and never rewritten, so any number of
// stores may be in flight from it; the rolling wait_group bound only keeps the
// queue finite. The final wait_group 0 means the kernel does not retire until
// the writes have landed at the destination.
static __global__ __launch_bounds__(1024) void tma_stream_write_kernel(
    uint8_t* __restrict__ p, uint32_t n_tiles, uint32_t stride_tiles,
    uint64_t iters, uint32_t tile_bytes, uint32_t fill) {
  extern __shared__ __align__(128) uint8_t smem[];
  for (uint32_t o = threadIdx.x * 4; o < tile_bytes; o += blockDim.x * 4)
    *reinterpret_cast<uint32_t*>(smem + o) = fill;
  __syncthreads();
  fence_proxy_async_shared();
  if (threadIdx.x != 0) return;

  uint32_t t = blockIdx.x % n_tiles;
  for (uint64_t k = 0; k < iters; ++k) {
    tma_store_1d(p + (size_t)t * tile_bytes, smem, tile_bytes);
    tma_store_commit();
    tma_store_wait_all<8>();
    t += stride_tiles;
    if (t >= n_tiles) t -= n_tiles;
  }
  tma_store_wait_all<0>();
}

// Reads every element exactly once (order-independent 32-bit wrapping sum), so
// the reader-over-peer total can be compared against the owner-over-local
// total. This is the --verify path; it is not timed.
static __global__ __launch_bounds__(1024) void sum_kernel(const uint4* p,
                                                          size_t n_vec,
                                                          uint32_t* sink) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (; i < n_vec; i += step) {
    const uint4 v = p[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  const uint32_t s = block_reduce_add(acc.x + acc.y + acc.z + acc.w);
  if (threadIdx.x == 0) sink[blockIdx.x] = s;
}

// Writes next[i] at the start of line i, turning the buffer into a linked
// cycle over lines. Runs on whichever device owns the buffer.
static __global__ void scatter_chain_kernel(uint32_t* p, const uint32_t* next,
                                            size_t lines) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  for (; i < lines; i += step) p[i * (kLineBytes / 4)] = next[i];
}

// One thread, one dependent load per hop: no MLP, so the measured time is pure
// round-trip latency. Prefetchers cannot help -- the cycle is a random
// permutation of lines built on the host.
static __global__ void chase_kernel(const uint32_t* p, uint32_t start,
                                    uint64_t hops, uint32_t* sink) {
  uint32_t idx = start;
  for (uint64_t h = 0; h < hops; ++h) idx = p[(size_t)idx * (kLineBytes / 4)];
  sink[0] = idx;
}

// ---------------------------------------------------------------------------
// NVLink hardware counters, via nvidia-smi
// ---------------------------------------------------------------------------
//
// `nvidia-smi nvlink -gt d` reads the per-link cumulative *data* (payload)
// counters, in KiB, and needs neither root nor the profiler-permission setting
// NCU needs. Payload is the right thing to count: a read request carries no
// payload, so a pull shows up as RX on the requester and TX on the owner --
// which is itself a consistency check on the counters, since the two must
// agree.
//
// Indexing note: nvidia-smi enumerates in NVML order, which is not CUDA order
// unless CUDA_DEVICE_ORDER=PCI_BUS_ID. Always address the GPU by PCI bus id.

struct NvlCounters {
  bool ok = false;
  uint64_t rx = 0, tx = 0;
  int fields = 0;
};

static bool accumulate_field(const char* line, const char* key, uint64_t* out) {
  const char* q = std::strstr(line, key);
  if (!q) return false;
  q += std::strlen(key);
  while (*q == ' ' || *q == '\t' || *q == ':') ++q;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(q, &end, 10);
  if (end == q) return false;  // "N/A"
  while (*end == ' ') ++end;
  uint64_t mul = 1024;  // documented unit is KiB; parse it anyway
  if (!std::strncmp(end, "KiB", 3)) mul = 1024ull;
  else if (!std::strncmp(end, "MiB", 3)) mul = 1024ull << 10;
  else if (!std::strncmp(end, "GiB", 3)) mul = 1024ull << 20;
  else if (end[0] == 'B') mul = 1;
  *out += (uint64_t)v * mul;
  return true;
}

static NvlCounters read_nvlink_counters(int dev) {
  NvlCounters c;
  char bus[80] = {0};
  if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) return c;
  char cmd[256];
  std::snprintf(cmd, sizeof(cmd), "nvidia-smi nvlink -gt d -i %s 2>/dev/null",
                bus);
  FILE* f = POPEN(cmd, "r");
  if (!f) return c;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    if (accumulate_field(line, "Data Rx", &c.rx)) c.fields++;
    if (accumulate_field(line, "Data Tx", &c.tx)) c.fields++;
  }
  PCLOSE(f);
  c.ok = c.fields > 0;
  return c;
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct Config {
  int reader = 0;  // GPU that issues the loads (every timed kernel runs here)
  int owner = 1;   // GPU that owns the "peer" buffer
  std::vector<std::string> modes{"bw", "lat", "wire"};
  std::vector<std::string> targets{"local", "peer"};
  std::string via = "ldst";   // transfer engine: ldst | tma
  std::string dir = "read";   // direction:       read | write
  uint64_t tma_tile = 8 << 10;  // bytes per cp.async.bulk copy
  std::vector<uint64_t> sizes;  // empty -> derived from the reader's L2 size
  uint64_t total = 4ull << 30;  // logical bytes read per bw/once point
  int reps = 5;                 // timed repetitions per bw point (median)
  int blocks = 0;               // 0 = 4 x SM count
  int threads = 256;
  uint64_t hops = 20000;             // pointer-chase hops per lat point
  std::vector<uint64_t> wire_sizes;  // empty -> {1M, 8M, 64M}
  uint64_t wire_total = 64ull << 30;
  int wire_launches = 8;
  int idle_ms = 500;  // idle window for the counter-drift baseline
  bool counters = true;
  bool verify = true;
  std::string csv;
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --reader N       GPU issuing the loads (default 0)\n"
      "  --owner N        GPU owning the peer buffer (default 1)\n"
      "  --modes LIST     bw,lat,wire,once (default bw,lat,wire)\n"
      "  --targets LIST   local,peer (default both; 'local' is the control)\n"
      "  --via ENGINE     ldst | tma  (default ldst). tma uses cp.async.bulk,\n"
      "                   the path exp3_gemm_rs_fused actually takes; sm_90+\n"
      "  --dir DIR        read | write (default read). write measures the\n"
      "                   PUSH side: does a store to peer memory touch the\n"
      "                   SENDER's L2? applies to bw/wire/once, not lat\n"
      "  --tma-tile B     bytes per cp.async.bulk copy (default 8K)\n"
      "  --sizes LIST     working sets, e.g. 1M,8M,64M (default: 128K..512K\n"
      "                   plus 1/64x..8x the reader's L2 capacity)\n"
      "  --total BYTES    logical bytes read per bw/once point (default 4G)\n"
      "  --reps N         timed repetitions per bw point (default 5, median)\n"
      "  --blocks N       CTAs, 0 = 4 x SM count (default 0)\n"
      "  --threads N      threads per CTA (default 256)\n"
      "  --hops N         pointer-chase hops per lat point (default 20000)\n"
      "  --wire-sizes L   working sets for the wire test (default 1M,8M,64M)\n"
      "  --wire-total B   logical bytes per wire point (default 64G)\n"
      "  --wire-launches N  kernel launches per wire point (default 8)\n"
      "  --idle-ms N      idle counter-drift baseline window (default 500)\n"
      "  --no-counters    skip the nvidia-smi NVLink counter reads\n"
      "  --no-verify      skip the peer-vs-owner checksum comparison\n"
      "  --csv PREFIX     write PREFIX_bw.csv / _lat.csv / _wire.csv\n",
      prog);
}

static Config parse_args(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--reader") c.reader = std::atoi(next().c_str());
    else if (a == "--owner") c.owner = std::atoi(next().c_str());
    else if (a == "--modes") c.modes = split_csv(next());
    else if (a == "--targets") c.targets = split_csv(next());
    else if (a == "--via") c.via = next();
    else if (a == "--dir") c.dir = next();
    else if (a == "--tma-tile") c.tma_tile = parse_bytes(next());
    else if (a == "--sizes") {
      c.sizes.clear();
      for (const auto& t : split_csv(next())) c.sizes.push_back(parse_bytes(t));
    } else if (a == "--total") c.total = parse_bytes(next());
    else if (a == "--reps") c.reps = std::atoi(next().c_str());
    else if (a == "--blocks") c.blocks = std::atoi(next().c_str());
    else if (a == "--threads") c.threads = std::atoi(next().c_str());
    else if (a == "--hops") c.hops = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--wire-sizes") {
      c.wire_sizes.clear();
      for (const auto& t : split_csv(next()))
        c.wire_sizes.push_back(parse_bytes(t));
    } else if (a == "--wire-total") c.wire_total = parse_bytes(next());
    else if (a == "--wire-launches") c.wire_launches = std::atoi(next().c_str());
    else if (a == "--idle-ms") c.idle_ms = std::atoi(next().c_str());
    else if (a == "--no-counters") c.counters = false;
    else if (a == "--no-verify") c.verify = false;
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (c.reader == c.owner) {
    std::fprintf(stderr, "--reader and --owner must differ\n");
    std::exit(1);
  }
  if (c.threads < 32 || c.threads > 1024 || c.threads % 32) {
    std::fprintf(stderr, "--threads must be a multiple of 32 in [32,1024]\n");
    std::exit(1);
  }
  if (c.reps < 1) c.reps = 1;
  if (c.via != "ldst" && c.via != "tma") {
    std::fprintf(stderr, "--via must be ldst or tma\n");
    std::exit(1);
  }
  if (c.dir != "read" && c.dir != "write") {
    std::fprintf(stderr, "--dir must be read or write\n");
    std::exit(1);
  }
  if (c.tma_tile < 128 || (c.tma_tile & (c.tma_tile - 1))) {
    std::fprintf(stderr, "--tma-tile must be a power of two >= 128\n");
    std::exit(1);
  }
  return c;
}

// Written by every write-mode kernel and checked by check_const_kernel. Any
// value works as long as the fill pattern cannot produce it by accident.
static constexpr uint32_t kFillWord = 0x5A5AA5A5u;

// ---------------------------------------------------------------------------
// harness state
// ---------------------------------------------------------------------------

struct Ctx {
  Config cfg;
  int sm_count = 0;
  uint64_t l2_bytes = 0;
  uint64_t max_size = 0;
  int blocks = 0;
  uint32_t* buf_local = nullptr;  // on reader
  uint32_t* buf_peer = nullptr;   // on owner, reached from reader through UVA
  uint32_t* sink_r = nullptr;     // on reader
  uint32_t* sink_o = nullptr;     // on owner
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
  FILE* csv_bw = nullptr;
  FILE* csv_lat = nullptr;
  FILE* csv_wire = nullptr;

  uint32_t* buf(const std::string& target) const {
    return target == "peer" ? buf_peer : buf_local;
  }
};

static FILE* open_csv(const std::string& prefix, const char* suffix,
                      const char* header) {
  const std::string path = prefix + suffix;
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return nullptr;
  }
  std::fprintf(f, "%s\n", header);
  return f;
}

// Working sets straddling both cache capacities. The small fixed sizes sit
// below the per-SM L1 (128-256 KB on recent parts) so an L1 step and an L2 step
// land in different places on the x axis and cannot be confused; the fractions
// of L2 put the L2 knee on the 1.0x point.
static std::vector<uint64_t> default_sizes(uint64_t l2) {
  static const double frac[] = {1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8,
                                1.0 / 4,  1.0 / 2,  0.75,     1.0,
                                1.5,      2.0,      4.0,      8.0};
  std::vector<uint64_t> out{64ull << 10, 128ull << 10, 256ull << 10,
                            512ull << 10};
  for (double f : frac) {
    uint64_t s = (uint64_t)(l2 * f);
    s &= ~(uint64_t)((1 << 17) - 1);  // round down to a 128 KiB multiple
    if (s >= (1ull << 19) && s > out.back()) out.push_back(s);
  }
  return out;
}

// Per-thread load count for a given working set: large enough to (a) move
// `total` bytes and (b) traverse the whole buffer at least once.
struct Shape {
  size_t n_vec = 0, stride_mod = 1;  // ld/st path: 16 B elements, per thread
  uint32_t tile_bytes = 0, n_tiles = 0, stride_tiles = 1, smem_bytes = 0;  // tma
  uint64_t iters = 0, logical_bytes = 0;  // both
};

// The tma variant wraps at TILE granularity across CTAs where the ld/st variant
// wraps at 16 B granularity across threads. Both hold the working set at exactly
// `size_bytes` at every point of the sweep and both cover it completely (the
// `cover` floor on `iters` is what guarantees that), so the two curves are read
// off the same x axis.
static Shape make_shape(const Ctx& x, uint64_t size_bytes, uint64_t total) {
  Shape s;
  if (x.cfg.via == "tma") {
    uint32_t tile = (uint32_t)std::min<uint64_t>(x.cfg.tma_tile, size_bytes);
    while (tile > 128 && (size_bytes % tile)) tile >>= 1;
    const uint64_t blocks = (uint64_t)x.blocks;
    s.tile_bytes = tile;
    s.n_tiles = (uint32_t)(size_bytes / tile);
    s.stride_tiles = (uint32_t)(blocks % s.n_tiles);
    if (s.stride_tiles == 0) s.stride_tiles = 1;
    const uint64_t cover = (s.n_tiles + blocks - 1) / blocks;
    const uint64_t want = total / (blocks * tile);
    s.iters = std::max<uint64_t>(std::max<uint64_t>(cover, want), 4);
    s.logical_bytes = blocks * s.iters * tile;
    // reads need STAGES buffers plus their mbarriers; writes re-issue from one
    // never-modified staging tile, so a single buffer is enough
    s.smem_bytes = (x.cfg.dir == "write")
                       ? tile
                       : (uint32_t)kTmaStages * tile + (uint32_t)kTmaStages * 8u;
    return s;
  }
  s.n_vec = size_bytes / 16;
  const size_t nthreads = (size_t)x.blocks * x.cfg.threads;
  s.stride_mod = nthreads % s.n_vec;
  if (s.stride_mod == 0) s.stride_mod = 1;
  const uint64_t cover = (s.n_vec + nthreads - 1) / nthreads;
  const uint64_t want = total / (nthreads * 16);
  s.iters = std::max<uint64_t>(std::max<uint64_t>(cover, want), 4);
  s.logical_bytes = (uint64_t)nthreads * s.iters * 16;
  return s;
}

// One timed launch of whichever (engine, direction) pair this run selected.
// The caller is responsible for having the reader device current.
static double time_op(Ctx& x, uint32_t* p, const Shape& s) {
  float ms = 0;
  const bool write = x.cfg.dir == "write";
  CUDA_CHECK(cudaEventRecord(x.ev0));
  if (x.cfg.via == "tma") {
    if (write)
      tma_stream_write_kernel<<<x.blocks, x.cfg.threads, s.smem_bytes>>>(
          (uint8_t*)p, s.n_tiles, s.stride_tiles, s.iters, s.tile_bytes,
          kFillWord);
    else
      tma_stream_read_kernel<<<x.blocks, x.cfg.threads, s.smem_bytes>>>(
          (const uint8_t*)p, s.n_tiles, s.stride_tiles, s.iters, s.tile_bytes,
          x.sink_r);
  } else if (write) {
    const uint4 v = make_uint4(kFillWord, kFillWord, kFillWord, kFillWord);
    stream_write_kernel<<<x.blocks, x.cfg.threads>>>(
        (uint4*)p, s.n_vec, s.stride_mod, s.iters, v);
  } else {
    stream_read_kernel<<<x.blocks, x.cfg.threads>>>(
        (const uint4*)p, s.n_vec, s.stride_mod, s.iters, x.sink_r);
  }
  CUDA_CHECK(cudaEventRecord(x.ev1));
  CUDA_CHECK(cudaEventSynchronize(x.ev1));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
  return (double)ms;
}

static uint32_t checksum(int dev, const uint32_t* p, size_t n_vec,
                         uint32_t* sink, int blocks, int threads) {
  CUDA_CHECK(cudaSetDevice(dev));
  sum_kernel<<<blocks, threads>>>((const uint4*)p, n_vec, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint32_t> h(blocks);
  CUDA_CHECK(cudaMemcpy(h.data(), sink, blocks * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  uint32_t t = 0;
  for (uint32_t v : h) t += v;
  return t;
}

// After a write sweep, ask the buffer's OWNER whether every word in the region
// that was written now holds kFillWord. For target=peer that is the question
// that matters: not "did the sender issue the stores" but "did the bytes land
// in the owner's HBM". Checked at the largest working set of the sweep, which
// covers every smaller one.
static bool verify_writes(Ctx& x, const std::string& target, uint64_t bytes) {
  if (!x.cfg.verify || bytes == 0) return true;
  const int dev = (target == "peer") ? x.cfg.owner : x.cfg.reader;
  uint32_t* bad = (target == "peer") ? x.sink_o : x.sink_r;
  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaMemset(bad, 0, sizeof(uint32_t)));
  check_const_kernel<<<256, 256>>>(x.buf(target), bytes / 4, kFillWord, bad);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  uint32_t n = 0;
  CUDA_CHECK(cudaMemcpy(&n, bad, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaSetDevice(x.cfg.reader));
  std::printf("  verify %s writes over %s (checked on GPU %d): %u bad words "
              "-> %s\n",
              target.c_str(), fmt_bytes(bytes).c_str(), dev, n,
              n ? "MISMATCH" : "OK");
  return n == 0;
}

static void refill(Ctx& x) {
  const size_t n = x.max_size / 4;
  CUDA_CHECK(cudaSetDevice(x.cfg.owner));
  fill_kernel<<<256, 256>>>(x.buf_peer, n, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaSetDevice(x.cfg.reader));
  fill_kernel<<<256, 256>>>(x.buf_local, n, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
}

// Bandwidth at the largest working set that still fits comfortably inside L2,
// over bandwidth at the smallest one that comfortably does not. This -- not
// first-point-over-last-point -- is the L2 knee: it deliberately excludes the
// sub-L1 sizes at the small end, which move for a different reason.
struct Knee {
  double in_l2 = 0, out_l2 = 0, ratio = 0;
  uint64_t in_bytes = 0, out_bytes = 0;
  bool valid = false;
};

static Knee find_knee(const std::vector<uint64_t>& sizes,
                      const std::vector<double>& gbps, uint64_t l2) {
  Knee k;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] <= l2 / 4 && sizes[i] > k.in_bytes) {
      k.in_bytes = sizes[i];
      k.in_l2 = gbps[i];
    }
    if (sizes[i] >= 2 * l2 && (k.out_bytes == 0 || sizes[i] < k.out_bytes)) {
      k.out_bytes = sizes[i];
      k.out_l2 = gbps[i];
    }
  }
  k.valid = k.in_bytes && k.out_bytes && k.out_l2 > 0;
  if (k.valid) k.ratio = k.in_l2 / k.out_l2;
  return k;
}

// ---------------------------------------------------------------------------
// mode: bw (and mode: once, which is the same thing with a single timed launch)
// ---------------------------------------------------------------------------

static void run_bw(Ctx& x, const char* mode_name, bool single_launch) {
  Config& cfg = x.cfg;
  refill(x);
  CUDA_CHECK(cudaSetDevice(cfg.reader));

  const bool write = cfg.dir == "write";
  std::printf("=== mode %s: %s %s bandwidth vs working set ===\n", mode_name,
              cfg.via.c_str(), write ? "write" : "read");
  std::printf("%-6s %10s %8s %11s %12s %10s %10s\n", "target", "workset",
              "x L2", write && cfg.via == "tma" ? "tiles/CTA" : "iters/thr",
              "logical GiB", "ms", "GB/s");

  for (const auto& target : cfg.targets) {
    std::vector<uint64_t> sizes_done;
    std::vector<double> gbps_done;
    uint64_t last_size = 0;
    for (uint64_t size : cfg.sizes) {
      if (size < 4096) continue;
      const Shape s = make_shape(x, size, cfg.total);
      uint32_t* p = x.buf(target);
      last_size = size;

      time_op(x, p, s);  // warmup: populate caches / TLB
      std::vector<double> ms;
      const int reps = single_launch ? 1 : cfg.reps;
      for (int r = 0; r < reps; ++r) ms.push_back(time_op(x, p, s));
      std::sort(ms.begin(), ms.end());
      const double med = ms[ms.size() / 2];
      const double gbps = (double)s.logical_bytes / (med * 1e-3) / 1e9;
      sizes_done.push_back(size);
      gbps_done.push_back(gbps);

      std::printf("%-6s %10s %8.3f %11llu %12.2f %10.3f %10.1f\n",
                  target.c_str(), fmt_bytes(size).c_str(),
                  (double)size / (double)x.l2_bytes,
                  (unsigned long long)s.iters,
                  (double)s.logical_bytes / 1073741824.0, med, gbps);
      std::fflush(stdout);

      if (x.csv_bw)
        std::fprintf(x.csv_bw,
                     "%s,%s,%llu,%.5f,%llu,%d,%d,%llu,%.4f,%.4f,%.2f,%s,%s,"
                     "%u\n",
                     BUILD_TAG, target.c_str(), (unsigned long long)size,
                     (double)size / (double)x.l2_bytes,
                     (unsigned long long)s.iters, x.blocks, cfg.threads,
                     (unsigned long long)s.logical_bytes, med, ms.front(),
                     gbps, cfg.via.c_str(), cfg.dir.c_str(), s.tile_bytes);
    }
    const Knee k = find_knee(sizes_done, gbps_done, x.l2_bytes);
    if (k.valid)
      std::printf("  -> %-5s L2 knee: %.1f GB/s at %s (in L2) vs %.1f GB/s at "
                  "%s (out) = %.2fx  %s\n",
                  target.c_str(), k.in_l2, fmt_bytes(k.in_bytes).c_str(),
                  k.out_l2, fmt_bytes(k.out_bytes).c_str(), k.ratio,
                  target == "peer"
                      ? (write ? "(~1.00 => the sender's L2 never held it)"
                               : "(~1.00 => the reader's L2 never held it)")
                      : "(must be >1, or the instrument is blind)");
    if (write && !verify_writes(x, target, last_size))
      std::fprintf(stderr,
                   "[!] %s writes did not land -- every number above for this "
                   "target is meaningless\n",
                   target.c_str());
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// mode: lat
// ---------------------------------------------------------------------------

static void build_chain(Ctx& x, int dev, uint32_t* buf, size_t lines) {
  std::vector<uint32_t> perm(lines), next(lines);
  for (size_t i = 0; i < lines; ++i) perm[i] = (uint32_t)i;
  // Sattolo's algorithm; the randomness is what defeats prefetchers, and
  // walking the permutation in order gives a single cycle over every line.
  std::mt19937 rng(12345);
  for (size_t i = lines - 1; i > 0; --i) {
    const size_t j = rng() % i;  // strictly less than i
    std::swap(perm[i], perm[j]);
  }
  for (size_t i = 0; i < lines; ++i) next[perm[i]] = perm[(i + 1) % lines];

  CUDA_CHECK(cudaSetDevice(dev));
  uint32_t* d_next = nullptr;
  CUDA_CHECK(cudaMalloc(&d_next, lines * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_next, next.data(), lines * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  scatter_chain_kernel<<<256, 256>>>(buf, d_next, lines);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_next));
  CUDA_CHECK(cudaSetDevice(x.cfg.reader));
}

static void run_lat(Ctx& x) {
  Config& cfg = x.cfg;
  std::printf("=== mode lat: dependent-load latency vs working set ===\n");
  std::printf("(128 B random cycle, one thread, no MLP; large sizes also pay\n"
              " TLB misses, so read the local/peer *comparison*, not absolutes)\n");
  if (cfg.via != "ldst" || cfg.dir != "read")
    std::printf("[note] lat always uses ld.global reads: a dependent-load\n"
                "       chase has no TMA or store equivalent. --via/--dir do\n"
                "       not apply here.\n");
  std::printf("%-6s %10s %8s %12s %10s %10s\n", "target", "workset", "x L2",
              "lines", "ms", "ns/hop");

  for (const auto& target : cfg.targets) {
    const int dev = (target == "peer") ? cfg.owner : cfg.reader;
    uint32_t* buf = x.buf(target);
    double ns_small = 0, ns_large = 0;
    bool first = true;
    for (uint64_t size : cfg.sizes) {
      const size_t lines = size / kLineBytes;
      if (lines < 64) continue;
      build_chain(x, dev, buf, lines);

      CUDA_CHECK(cudaSetDevice(cfg.reader));
      float ms = 0;
      chase_kernel<<<1, 1>>>(buf, 0, cfg.hops / 10 + 1, x.sink_r);  // warmup
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(x.ev0));
      chase_kernel<<<1, 1>>>(buf, 0, cfg.hops, x.sink_r);
      CUDA_CHECK(cudaEventRecord(x.ev1));
      CUDA_CHECK(cudaEventSynchronize(x.ev1));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
      const double ns = (double)ms * 1e6 / (double)cfg.hops;
      if (first) { ns_small = ns; first = false; }
      ns_large = ns;

      std::printf("%-6s %10s %8.3f %12zu %10.3f %10.1f\n", target.c_str(),
                  fmt_bytes(size).c_str(), (double)size / (double)x.l2_bytes,
                  lines, (double)ms, ns);
      std::fflush(stdout);

      if (x.csv_lat)
        std::fprintf(x.csv_lat, "%s,%s,%llu,%.5f,%zu,%llu,%.4f,%.2f\n",
                     BUILD_TAG, target.c_str(), (unsigned long long)size,
                     (double)size / (double)x.l2_bytes, lines,
                     (unsigned long long)cfg.hops, (double)ms, ns);
    }
    if (ns_small > 0)
      std::printf("  -> %-5s smallest %.0f ns, largest %.0f ns (%.2fx)\n",
                  target.c_str(), ns_small, ns_large, ns_large / ns_small);
  }
  std::printf(
      "  A peer working set that fit in the reader's L2 would chase at the\n"
      "  reader's L2 latency. Check whether it does.\n\n");
  refill(x);  // the chain overwrote the checksum pattern
}

// ---------------------------------------------------------------------------
// mode: wire
// ---------------------------------------------------------------------------

static void run_wire(Ctx& x) {
  Config& cfg = x.cfg;
  refill(x);
  CUDA_CHECK(cudaSetDevice(cfg.reader));

  const bool write = cfg.dir == "write";
  // A pull shows up as RX on the requester and TX on the owner; a push is the
  // other way round. `near` is always the counter on the GPU running the
  // kernel, `far` the one on the GPU that owns the buffer -- they must agree
  // whichever direction the payload travels.
  auto near_ctr = [&](const NvlCounters& c) { return write ? c.tx : c.rx; };
  auto far_ctr = [&](const NvlCounters& c) { return write ? c.rx : c.tx; };
  const char* near_name = write ? "TX" : "RX";
  const char* far_name = write ? "RX" : "TX";

  std::printf("=== mode wire: bytes on the link vs bytes the SMs asked for "
              "(%s %s) ===\n",
              cfg.via.c_str(), write ? "write" : "read");

  // Idle drift: what these counters accumulate with the box doing nothing
  // (background NVLink chatter, other tenants). Subtracted from every
  // measurement and reported, so a small non-zero `local` reading is not
  // mistaken for signal.
  double idle_rx_bps = 0, idle_tx_bps = 0;
  bool counters_ok = false;
  if (cfg.counters) {
    const NvlCounters a = read_nvlink_counters(cfg.reader);
    const NvlCounters ao = read_nvlink_counters(cfg.owner);
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.idle_ms));
    const NvlCounters b = read_nvlink_counters(cfg.reader);
    const NvlCounters bo = read_nvlink_counters(cfg.owner);
    counters_ok = a.ok && b.ok && ao.ok && bo.ok;
    if (counters_ok) {
      idle_rx_bps = (double)(near_ctr(b) - near_ctr(a)) / (cfg.idle_ms * 1e-3);
      idle_tx_bps = (double)(far_ctr(bo) - far_ctr(ao)) / (cfg.idle_ms * 1e-3);
      std::printf("idle drift over %d ms: reader %s %.1f MB/s, owner %s "
                  "%.1f MB/s\n", cfg.idle_ms, near_name, idle_rx_bps / 1e6,
                  far_name, idle_tx_bps / 1e6);
    } else {
      std::printf(
          "[!] nvidia-smi NVLink data counters unavailable on this box (no\n"
          "    NVLink, or the driver reports N/A). The bandwidth and latency\n"
          "    sweeps still stand; for a byte count fall back to\n"
          "    `ncu --section Nvlink` -- run_all.sh does that too.\n");
    }
  }

  char near_hdr[16], far_hdr[16], ratio_hdr[16];
  std::snprintf(near_hdr, sizeof(near_hdr), "wire %s GiB", near_name);
  std::snprintf(far_hdr, sizeof(far_hdr), "wire %s GiB", far_name);
  std::snprintf(ratio_hdr, sizeof(ratio_hdr), "%s/asked", near_name);
  std::printf("%-6s %9s %11s %9s %9s %11s %11s %9s\n", "target", "workset",
              "asked GiB", "ms", "GB/s", near_hdr, far_hdr, ratio_hdr);

  uint64_t last_size = 0;
  for (uint64_t size : cfg.wire_sizes) {
    if (size < 4096 || size > x.max_size) continue;
    const uint64_t per_launch =
        cfg.wire_total / (uint64_t)std::max(1, cfg.wire_launches);
    const Shape s = make_shape(x, size, per_launch);
    const uint64_t asked = s.logical_bytes * (uint64_t)cfg.wire_launches;
    last_size = size;

    for (const auto& target : cfg.targets) {
      uint32_t* p = x.buf(target);
      time_op(x, p, s);  // warmup, outside the bracket

      // The drift correction has to use the *counter* window, not the kernel
      // time: the nvidia-smi calls themselves take ~100 ms each and sit inside
      // the bracket.
      const auto t0 = std::chrono::steady_clock::now();
      const NvlCounters c0 =
          cfg.counters ? read_nvlink_counters(cfg.reader) : NvlCounters{};
      const NvlCounters o0 =
          cfg.counters ? read_nvlink_counters(cfg.owner) : NvlCounters{};
      double ms_sum = 0;
      for (int l = 0; l < cfg.wire_launches; ++l) ms_sum += time_op(x, p, s);
      const NvlCounters c1 =
          cfg.counters ? read_nvlink_counters(cfg.reader) : NvlCounters{};
      const NvlCounters o1 =
          cfg.counters ? read_nvlink_counters(cfg.owner) : NvlCounters{};
      const double wall_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();

      const bool ok = counters_ok && c0.ok && c1.ok && o0.ok && o1.ok;
      const double rx =
          ok ? std::max(0.0, (double)(near_ctr(c1) - near_ctr(c0)) -
                                 idle_rx_bps * wall_s)
             : 0;
      const double tx =
          ok ? std::max(0.0, (double)(far_ctr(o1) - far_ctr(o0)) -
                                 idle_tx_bps * wall_s)
             : 0;
      const double ratio_rx = ok ? rx / (double)asked : -1.0;
      const double ratio_tx = ok ? tx / (double)asked : -1.0;
      const double gbps = (double)asked / (ms_sum * 1e-3) / 1e9;

      char ratio_str[16];
      if (ok) std::snprintf(ratio_str, sizeof(ratio_str), "%.3f", ratio_rx);
      else std::snprintf(ratio_str, sizeof(ratio_str), "n/a");

      std::printf("%-6s %9s %11.2f %9.1f %9.1f %11.2f %11.2f %9s\n",
                  target.c_str(), fmt_bytes(size).c_str(),
                  (double)asked / 1073741824.0, ms_sum, gbps,
                  rx / 1073741824.0, tx / 1073741824.0, ratio_str);
      std::fflush(stdout);

      if (x.csv_wire)
        std::fprintf(x.csv_wire,
                     "%s,%s,%llu,%d,%llu,%llu,%.3f,%.2f,%.0f,%.0f,%.4f,%.4f,"
                     "%.0f,%.0f,%d,%s,%s,%u\n",
                     BUILD_TAG, target.c_str(), (unsigned long long)size,
                     cfg.wire_launches, (unsigned long long)s.iters,
                     (unsigned long long)asked, ms_sum, gbps, rx, tx, ratio_rx,
                     ratio_tx, idle_rx_bps, idle_tx_bps, ok ? 1 : 0,
                     cfg.via.c_str(), cfg.dir.c_str(), s.tile_bytes);
    }
  }
  if (write)
    for (const auto& target : cfg.targets) verify_writes(x, target, last_size);
  std::printf(
      "\n%s/asked ~= 1.00 on `peer`  => every logical byte crossed the link,\n"
      "                               nothing was absorbed by a cache on the\n"
      "                               kernel's own GPU.\n"
      "%s/asked ~= 0    on `local` => the counters measure what we think.\n"
      "The near-end and far-end counters must agree; if they do not, the\n"
      "counters (not the hypothesis) are the thing to distrust.\n"
      "CSV note: wire_rx_bytes always holds the NEAR-end counter and\n"
      "wire_tx_bytes the FAR-end one, so for dir=write they are the sender's\n"
      "TX and the owner's RX respectively.\n\n",
      near_name, near_name);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Ctx x;
  x.cfg = parse_args(argc, argv);
  Config& cfg = x.cfg;

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (cfg.reader >= ndev || cfg.owner >= ndev) {
    std::fprintf(stderr, "need GPUs %d and %d, found %d\n", cfg.reader,
                 cfg.owner, ndev);
    return 1;
  }

  cudaDeviceProp pr{}, po{};
  CUDA_CHECK(cudaGetDeviceProperties(&pr, cfg.reader));
  CUDA_CHECK(cudaGetDeviceProperties(&po, cfg.owner));
  x.sm_count = pr.multiProcessorCount;
  int l2 = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, cfg.reader));
  x.l2_bytes = (uint64_t)l2;
  x.blocks = cfg.blocks > 0 ? std::min(cfg.blocks, kMaxBlocks)
                            : std::min(kMaxBlocks, 4 * x.sm_count);

  int can = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can, cfg.reader, cfg.owner));
  if (!can) {
    std::fprintf(stderr,
                 "GPU %d cannot peer-access GPU %d -- this experiment needs\n"
                 "P2P (NVLink, or PCIe through BAR1); nothing to measure.\n",
                 cfg.reader, cfg.owner);
    return 1;
  }
  int perf_rank = -1, atomics = -1;
  cudaDeviceGetP2PAttribute(&perf_rank, cudaDevP2PAttrPerformanceRank,
                            cfg.reader, cfg.owner);
  cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported,
                            cfg.reader, cfg.owner);
  cudaGetLastError();  // these are informational; do not poison later checks

  std::printf("=== peer_l2_probe (build %s) ===\n", BUILD_TAG);
  std::printf("reader GPU %d: %s  sm_%d%d  %d SMs  L2 %.1f MiB\n", cfg.reader,
              pr.name, pr.major, pr.minor, pr.multiProcessorCount,
              x.l2_bytes / 1048576.0);
  std::printf("owner  GPU %d: %s  sm_%d%d  %d SMs\n", cfg.owner, po.name,
              po.major, po.minor, po.multiProcessorCount);
  std::printf("P2P: perf rank %d, native atomics %d  (rank 0 with atomics is\n"
              "     the NVLink signature; PCIe P2P usually reports atomics 0)\n",
              perf_rank, atomics);
  std::printf("grid: %d CTAs x %d threads\n", x.blocks, cfg.threads);
  std::printf("path: via=%s dir=%s%s\n", cfg.via.c_str(), cfg.dir.c_str(),
              cfg.via == "tma" ? " (cp.async.bulk)" : " (ld/st.global)");
  if (cfg.via == "tma") {
    if (pr.major < 9) {
      std::fprintf(stderr,
                   "--via tma needs cp.async.bulk (sm_90+); reader GPU %d is "
                   "sm_%d%d\n",
                   cfg.reader, pr.major, pr.minor);
      return 1;
    }
    std::printf("      tma tile %s, %d-stage pipeline, 1 issuing thread/CTA\n",
                fmt_bytes(cfg.tma_tile).c_str(), kTmaStages);
  }

  if (cfg.sizes.empty()) cfg.sizes = default_sizes(x.l2_bytes);
  if (cfg.wire_sizes.empty())
    cfg.wire_sizes = {1ull << 20, 8ull << 20, 64ull << 20};
  std::sort(cfg.sizes.begin(), cfg.sizes.end());

  x.max_size = 0;
  for (uint64_t s : cfg.sizes) x.max_size = std::max(x.max_size, s);
  for (uint64_t s : cfg.wire_sizes) x.max_size = std::max(x.max_size, s);
  x.max_size = (x.max_size + 4095) & ~4095ull;
  std::printf("buffers: %.1f MiB on each of GPU %d and GPU %d\n\n",
              x.max_size / 1048576.0, cfg.reader, cfg.owner);

  // allocate + enable peer access
  CUDA_CHECK(cudaSetDevice(cfg.owner));
  CUDA_CHECK(cudaMalloc(&x.buf_peer, x.max_size));
  CUDA_CHECK(cudaMalloc(&x.sink_o, kMaxBlocks * sizeof(uint32_t)));
  CUDA_CHECK(cudaSetDevice(cfg.reader));
  CUDA_CHECK(cudaMalloc(&x.buf_local, x.max_size));
  CUDA_CHECK(cudaMalloc(&x.sink_r, kMaxBlocks * sizeof(uint32_t)));
  {
    const cudaError_t e = cudaDeviceEnablePeerAccess(cfg.owner, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    else CUDA_CHECK(e);
  }
  CUDA_CHECK(cudaEventCreate(&x.ev0));
  CUDA_CHECK(cudaEventCreate(&x.ev1));

  // Default --tma-tile keeps the staging buffers under the 48 KiB static
  // limit; a larger one needs the opt-in. Failure here is not fatal -- the
  // launch would report it, and only oversized tiles can hit it.
  if (cfg.via == "tma") {
    const int want = (int)std::min<uint64_t>(
        200u << 10, (uint64_t)kTmaStages * cfg.tma_tile + kTmaStages * 8u);
    cudaFuncSetAttribute(tma_stream_read_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, want);
    cudaFuncSetAttribute(tma_stream_write_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, want);
    cudaGetLastError();
  }

  // The peer pointer must really live on the owner: a silent fallback to a
  // local mapping would make every number below meaningless.
  cudaPointerAttributes pa{};
  CUDA_CHECK(cudaPointerGetAttributes(&pa, x.buf_peer));
  if (pa.device != cfg.owner) {
    std::fprintf(stderr, "peer buffer reports device %d, expected %d\n",
                 pa.device, cfg.owner);
    return 1;
  }

  refill(x);

  if (cfg.verify) {
    const size_t n_vec = x.max_size / 16;
    const uint32_t sum_owner =
        checksum(cfg.owner, x.buf_peer, n_vec, x.sink_o, 256, 256);
    const uint32_t sum_reader =
        checksum(cfg.reader, x.buf_peer, n_vec, x.sink_r, 256, 256);
    const uint32_t sum_local =
        checksum(cfg.reader, x.buf_local, n_vec, x.sink_r, 256, 256);
    const bool ok = (sum_owner == sum_reader) && (sum_owner == sum_local);
    std::printf("verify: owner-over-local 0x%08x, reader-over-peer 0x%08x, "
                "reader-over-local 0x%08x -> %s\n\n",
                sum_owner, sum_reader, sum_local, ok ? "OK" : "MISMATCH");
    if (!ok) {
      std::fprintf(stderr,
                   "peer reads do not return the owner's data -- fix that "
                   "before reading anything else here\n");
      return 1;
    }
  }
  CUDA_CHECK(cudaSetDevice(cfg.reader));

  if (!cfg.csv.empty()) {
    // New columns are appended, never inserted: run_all.sh's verdict reduces
    // these files positionally and older bundles must stay readable.
    x.csv_bw = open_csv(cfg.csv, "_bw.csv",
                        "build,target,bytes,frac_l2,iters_per_thread,blocks,"
                        "threads,logical_bytes,ms_med,ms_min,gbps,via,dir,"
                        "tma_tile_bytes");
    x.csv_lat = open_csv(cfg.csv, "_lat.csv",
                         "build,target,bytes,frac_l2,lines,hops,ms,ns_per_hop");
    x.csv_wire = open_csv(
        cfg.csv, "_wire.csv",
        "build,target,bytes,launches,iters_per_thread,asked_bytes,ms,gbps,"
        "wire_rx_bytes,wire_tx_bytes,rx_over_asked,tx_over_asked,"
        "idle_rx_bps,idle_tx_bps,counters_ok,via,dir,tma_tile_bytes");
  }

  if (has(cfg.modes, "bw")) run_bw(x, "bw", false);
  if (has(cfg.modes, "lat")) run_lat(x);
  if (has(cfg.modes, "wire")) run_wire(x);
  if (has(cfg.modes, "once")) run_bw(x, "once", true);

  std::printf(
      "How to read this run\n"
      "  1. `local` bandwidth must rise as the working set drops below L2 and\n"
      "     `local` latency must fall. That step is the control: the\n"
      "     instrument proving it can see an L2 when there is one.\n"
      "  2. If `peer` shows the same step at the same place, the reader's L2\n"
      "     IS caching remote lines and the memory-side story is wrong here.\n"
      "  3. If `peer` is flat -- same bandwidth, same latency, whether the\n"
      "     remote buffer is 1/64 of L2 or 8x L2 -- the reader's L2 is not\n"
      "     holding it, and mode=wire says the same thing in bytes.\n"
      "  4. A step in `peer` that appears only below ~256 KB, only in the\n"
      "     default build, and vanishes in peer_l2_probe_nol1 is L1, not L2.\n"
      "  5. `--dir write` asks the same question of the PUSH side: does a\n"
      "     store into peer memory touch the SENDER's L2? Same reading --\n"
      "     `local` must step, `peer` must not, and the link must carry every\n"
      "     byte. Global stores are write-through in L1, so no L1 confound.\n"
      "  6. `--via tma` re-asks all of it for cp.async.bulk, the engine\n"
      "     exp3_gemm_rs_fused actually uses. If the tma `peer` curve is flat\n"
      "     in BOTH builds where the ld/st one steps below 256 KB in the\n"
      "     default build, then TMA does not populate L1 and the L1 result\n"
      "     above does not transfer to the fused kernel.\n");

  if (x.csv_bw) std::fclose(x.csv_bw);
  if (x.csv_lat) std::fclose(x.csv_lat);
  if (x.csv_wire) std::fclose(x.csv_wire);
  CUDA_CHECK(cudaSetDevice(cfg.reader));
  cudaFree(x.buf_local);
  cudaFree(x.sink_r);
  CUDA_CHECK(cudaSetDevice(cfg.owner));
  cudaFree(x.buf_peer);
  cudaFree(x.sink_o);
  return 0;
}
