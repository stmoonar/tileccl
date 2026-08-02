// p2p_ce_vs_tma.cu
//
// Microbenchmark: Copy Engine (cudaMemcpyPeerAsync) vs TMA (cp.async.bulk)
// for inter-GPU transfers over PCIe, targeting sm_120 (Blackwell consumer).
//
// Purpose: decide at what tile granularity compute/communication fusion should
// happen. We measure a [tokens x hidden] activation tensor moving from GPU A to
// GPU B with four different transfer mechanisms:
//
// PUSH variants (executed by the source GPU; PCIe posted writes):
//   ce         cudaMemcpyPeerAsync            -- DMA copy engine, zero SM cost
//   sm         kernel with uint4 ld/st        -- SM-driven stores into peer VA
//   tma        gmem(local) -> smem -> gmem(peer) via cp.async.bulk
//   tma_store  smem -> gmem(peer) via cp.async.bulk
//
// PULL variants (same mechanisms executed by the destination GPU; PCIe
// non-posted reads -- each read is a request/completion round trip, so
// throughput is bounded by how many reads the engine keeps in flight):
//   ce_pull    cudaMemcpyPeerAsync enqueued on a dst-device stream
//   sm_pull    kernel on dst: uint4 loads from peer VA, stores to local gmem
//   tma_pull   gmem(peer) -> smem -> gmem(local) via cp.async.bulk, on dst
//   tma_load   gmem(peer) -> smem only -- the fused *consumer* model
//
// `tma` is the apples-to-apples comparison against `ce` (both are gmem->gmem).
// `tma_store` models the *fused producer*: the tile was just produced in
// shared memory by a compute kernel and is pushed straight to the peer without
// a round trip through local global memory. It is an upper bound; its payload
// is intentionally not a faithful copy of the source, so verification is
// skipped. `tma_load` is its pull-side mirror: the consumer fetches the remote
// tile straight into shared memory (like a GEMM operand load from a peer);
// nothing is written to gmem, so verification is skipped there too.
//
// Key assumption under test: a peer device pointer (valid in the unified VA
// space after cudaDeviceEnablePeerAccess) can be used as the destination of
// `cp.async.bulk.global.shared::cta`. If the TMA unit cannot route through the
// PCIe P2P path, expect either an illegal-address fault or a verification
// failure -- both are meaningful results for this study.
//
// Build:  make            (see tests/Makefile)
// Run:    ./p2p_ce_vs_tma --tokens 1,16,256,4096 --hidden 7168 --dtype bf16

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

// Non-fatal variant: used around kernels that may legitimately fault if the
// hardware refuses TMA over the P2P path.
static bool cuda_ok(const char* what) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[FAIL] %s -> %s\n"
                 "       (if this is a sticky fault the CUDA context is now\n"
                 "        unusable; rerun the remaining methods separately via\n"
                 "        --methods)\n",
                 what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// PTX helpers for cp.async.bulk / mbarrier (sm_90+, valid on sm_120)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)),
               "r"(count));
}

// Makes prior generic-proxy shared-memory writes (mbarrier init, SM stores)
// visible to the async proxy that TMA operates in.
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar,
                                                          uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_addr(bar)),
               "r"(bytes)
               : "memory");
}

// Branch-free try_wait: avoids duplicated PTX labels when inlined more than
// once in a single kernel.
__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* bar,
                                                  uint32_t phase) {
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
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  while (!mbarrier_try_wait(bar, phase)) {
  }
}

// global -> shared::cta, completion signalled through `bar`'s tx count.
__device__ __forceinline__ void tma_load_1d(void* smem_dst, const void* gmem_src,
                                            uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(smem_addr(smem_dst)),
      "l"(gmem_src), "r"(bytes), "r"(smem_addr(bar))
      : "memory");
}

// shared::cta -> global. `gmem_dst` may be a peer device pointer.
__device__ __forceinline__ void tma_store_1d(void* gmem_dst,
                                             const void* smem_src,
                                             uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                   "l"(gmem_dst),
               "r"(smem_addr(smem_src)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Waits until at most N bulk-store groups still have the *source* smem live.
// Cheaper than full completion; only tells us the buffer can be recycled.
template <int N>
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// Waits until at most N bulk-store groups are outstanding *end to end*, i.e.
// the writes have landed at the destination. This is what we need before the
// kernel exits, otherwise we would not be timing the PCIe write at all.
template <int N>
__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

constexpr uint32_t kTmaThreads = 128;
constexpr uint32_t kSmThreads = 256;

// gmem(local) -> smem -> gmem(peer), software pipelined over STAGES buffers.
//
// Buffer accounting (steady state): STAGES = L + 1 + W, where
//   L = loads in flight              = STAGES - 2
//   1 = buffer whose store we issue this iteration
//   W = stores still reading their buffer = 1  (cp.async.bulk.wait_group.read 1)
// Hence STAGES >= 3.
template <int STAGES>
__global__ __launch_bounds__(kTmaThreads) void tma_p2p_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint64_t total_bytes, uint32_t tile_bytes, uint32_t tile_stride,
    uint32_t num_tiles) {
  static_assert(STAGES >= 3, "need at least 3 stages for the pipeline");
  constexpr uint32_t kInflight = STAGES - 2;

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* buf = smem;
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + STAGES * (size_t)tile_stride);

  // Lane 0 owns the whole pipeline: the barriers and the staging buffers are
  // touched by nobody else, so no __syncthreads() is needed anywhere here.
  if (threadIdx.x != 0) return;

#pragma unroll
  for (int s = 0; s < STAGES; ++s) mbarrier_init(&bar[s], 1);
  fence_proxy_async_shared();

  // Tiles owned by this CTA: blockIdx.x, blockIdx.x + gridDim.x, ...
  const uint32_t nblk = gridDim.x;
  const uint32_t my_tiles =
      (num_tiles > blockIdx.x) ? ((num_tiles - blockIdx.x + nblk - 1) / nblk) : 0;
  if (my_tiles == 0) return;

  auto tile_off = [&](uint32_t k) -> uint64_t {
    return (uint64_t)(blockIdx.x + k * nblk) * tile_bytes;
  };
  auto tile_len = [&](uint64_t off) -> uint32_t {
    uint64_t rem = total_bytes - off;
    return (uint32_t)(rem < tile_bytes ? rem : tile_bytes);
  };
  auto issue_load = [&](uint32_t k) {
    const uint32_t s = k % STAGES;
    const uint64_t off = tile_off(k);
    const uint32_t nb = tile_len(off);
    mbarrier_arrive_expect_tx(&bar[s], nb);
    tma_load_1d(buf + (size_t)s * tile_stride, src + off, nb, &bar[s]);
  };

  uint32_t phase_bits = 0;  // bit s = expected parity of bar[s]
  uint32_t issued = 0;
  for (; issued < kInflight && issued < my_tiles; ++issued) issue_load(issued);

  for (uint32_t done = 0; done < my_tiles; ++done) {
    const uint32_t s = done % STAGES;
    mbarrier_wait(&bar[s], (phase_bits >> s) & 1u);
    phase_bits ^= (1u << s);

    const uint64_t off = tile_off(done);
    tma_store_1d(dst + off, buf + (size_t)s * tile_stride, tile_len(off));
    tma_store_commit();

    if (issued < my_tiles) {
      // Recycle the oldest buffer only after its store has drained it.
      tma_store_wait_read<1>();
      issue_load(issued++);
    }
  }

  // Drain: wait for the writes to actually land on the peer, not just for the
  // source buffers to be readable again.
  tma_store_wait_all<0>();
}

// smem -> gmem(peer) only. Models a fused kernel pushing a freshly computed
// tile straight out. Payload is a repeated copy of this CTA's first tile, so
// the destination contents are NOT a faithful copy -- verification is skipped.
template <int STAGES>
__global__ __launch_bounds__(kTmaThreads) void tma_store_only_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint64_t total_bytes, uint32_t tile_bytes, uint32_t tile_stride,
    uint32_t num_tiles) {
  static_assert(STAGES >= 2, "need at least 2 stages");
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* buf = smem;

  const uint32_t nblk = gridDim.x;
  const uint32_t my_tiles =
      (num_tiles > blockIdx.x) ? ((num_tiles - blockIdx.x + nblk - 1) / nblk) : 0;
  if (my_tiles == 0) return;

  // Prime every stage buffer with real data (SM path, one-off, outside timing
  // relevance for large transfers but included in the reported number).
  {
    const uint64_t off0 = (uint64_t)blockIdx.x * tile_bytes;
    const uint32_t nb0 =
        (uint32_t)((total_bytes - off0) < tile_bytes ? (total_bytes - off0)
                                                     : tile_bytes);
    const uint4* gsrc = reinterpret_cast<const uint4*>(src + off0);
    const uint32_t nvec = nb0 / sizeof(uint4);
    for (uint32_t s = 0; s < STAGES; ++s) {
      uint4* sdst = reinterpret_cast<uint4*>(buf + (size_t)s * tile_stride);
      for (uint32_t i = threadIdx.x; i < nvec; i += blockDim.x) sdst[i] = gsrc[i];
    }
  }
  // SM wrote shared memory in the generic proxy; make it visible to TMA.
  fence_proxy_async_shared();
  __syncthreads();

  if (threadIdx.x == 0) {
    for (uint32_t k = 0; k < my_tiles; ++k) {
      // Buffer k%STAGES was last used by tile k-STAGES; allowing STAGES-1
      // outstanding groups guarantees that one has drained.
      tma_store_wait_read<STAGES - 1>();
      const uint64_t off = (uint64_t)(blockIdx.x + k * nblk) * tile_bytes;
      const uint64_t rem = total_bytes - off;
      const uint32_t nb = (uint32_t)(rem < tile_bytes ? rem : tile_bytes);
      tma_store_1d(dst + off, buf + (size_t)(k % STAGES) * tile_stride, nb);
      tma_store_commit();
    }
    tma_store_wait_all<0>();
  }
}

// gmem(peer) -> smem only. Models a fused consumer pulling a remote tile
// straight into shared memory for compute. Nothing reaches local gmem, so
// verification is skipped; the PCIe traffic is still the full payload.
template <int STAGES>
__global__ __launch_bounds__(kTmaThreads) void tma_load_only_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint64_t total_bytes, uint32_t tile_bytes, uint32_t tile_stride,
    uint32_t num_tiles) {
  static_assert(STAGES >= 2, "need at least 2 stages");
  (void)dst;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* buf = smem;
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + STAGES * (size_t)tile_stride);

  if (threadIdx.x != 0) return;

#pragma unroll
  for (int s = 0; s < STAGES; ++s) mbarrier_init(&bar[s], 1);
  fence_proxy_async_shared();

  const uint32_t nblk = gridDim.x;
  const uint32_t my_tiles =
      (num_tiles > blockIdx.x) ? ((num_tiles - blockIdx.x + nblk - 1) / nblk) : 0;
  if (my_tiles == 0) return;

  auto issue_load = [&](uint32_t k) {
    const uint32_t s = k % STAGES;
    const uint64_t off = (uint64_t)(blockIdx.x + k * nblk) * tile_bytes;
    const uint64_t rem = total_bytes - off;
    const uint32_t nb = (uint32_t)(rem < tile_bytes ? rem : tile_bytes);
    mbarrier_arrive_expect_tx(&bar[s], nb);
    tma_load_1d(buf + (size_t)s * tile_stride, src + off, nb, &bar[s]);
  };

  // No store side: all STAGES buffers can hold loads in flight. A buffer is
  // reused only after its barrier fired, i.e. the previous load into it landed.
  uint32_t phase_bits = 0;
  uint32_t issued = 0;
  for (; issued < (uint32_t)STAGES && issued < my_tiles; ++issued)
    issue_load(issued);

  for (uint32_t done = 0; done < my_tiles; ++done) {
    const uint32_t s = done % STAGES;
    mbarrier_wait(&bar[s], (phase_bits >> s) & 1u);
    phase_bits ^= (1u << s);
    if (issued < my_tiles) issue_load(issued++);
  }
}

// Plain SM-driven copy: 128-bit loads from local gmem, 128-bit stores into the
// peer's VA range.
__global__ __launch_bounds__(kSmThreads) void sm_p2p_kernel(
    const uint4* __restrict__ src, uint4* __restrict__ dst, uint64_t n_vec) {
  uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
  for (; idx < n_vec; idx += stride) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// config / CLI
// ---------------------------------------------------------------------------

struct Config {
  std::vector<uint64_t> token_list{1, 4, 16, 64, 256, 1024, 4096};
  uint64_t hidden = 7168;
  uint32_t dtype_bytes = 2;
  std::string dtype_name = "bf16";
  uint32_t tile_tokens = 0;  // 0 = auto
  int stages = 4;
  int blocks = 0;  // 0 = auto
  int iters = 100;
  int warmup = 20;
  int src_dev = 0;
  int dst_dev = 1;
  bool check = true;
  std::vector<std::string> methods{"ce",      "ce_pull",  "sm",
                                   "sm_pull", "tma",      "tma_pull",
                                   "tma_store", "tma_load"};
};

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

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --tokens LIST     comma-separated token counts (default 1,4,16,64,256,1024,4096)\n"
      "  --hidden N        hidden dimension (default 7168)\n"
      "  --dtype T         bf16|fp16|fp32|fp8 (default bf16)\n"
      "  --tile-tokens N   tokens per TMA tile, 0 = auto-fit shared memory (default 0)\n"
      "  --stages N        TMA pipeline depth, >=3 (default 4)\n"
      "  --blocks N        CTAs for the kernel paths, 0 = auto (default 0)\n"
      "  --iters N         timed iterations (default 100)\n"
      "  --warmup N        warmup iterations (default 20)\n"
      "  --src N           source GPU index (default 0)\n"
      "  --dst N           destination GPU index (default 1)\n"
      "  --methods LIST    subset of ce,ce_pull,sm,sm_pull,tma,tma_pull,\n"
      "                    tma_store,tma_load (default all; *_pull and\n"
      "                    tma_load execute on the dst GPU)\n"
      "  --no-check        skip correctness verification\n",
      prog);
}

static Config parse_args(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--tokens") {
      c.token_list.clear();
      for (auto& t : split_csv(next())) c.token_list.push_back(std::strtoull(t.c_str(), nullptr, 10));
    } else if (a == "--hidden") {
      c.hidden = std::strtoull(next().c_str(), nullptr, 10);
    } else if (a == "--dtype") {
      c.dtype_name = next();
      if (c.dtype_name == "bf16" || c.dtype_name == "fp16") c.dtype_bytes = 2;
      else if (c.dtype_name == "fp32") c.dtype_bytes = 4;
      else if (c.dtype_name == "fp8") c.dtype_bytes = 1;
      else { std::fprintf(stderr, "unknown dtype %s\n", c.dtype_name.c_str()); std::exit(1); }
    } else if (a == "--tile-tokens") {
      c.tile_tokens = (uint32_t)std::strtoul(next().c_str(), nullptr, 10);
    } else if (a == "--stages") {
      c.stages = std::atoi(next().c_str());
    } else if (a == "--blocks") {
      c.blocks = std::atoi(next().c_str());
    } else if (a == "--iters") {
      c.iters = std::atoi(next().c_str());
    } else if (a == "--warmup") {
      c.warmup = std::atoi(next().c_str());
    } else if (a == "--src") {
      c.src_dev = std::atoi(next().c_str());
    } else if (a == "--dst") {
      c.dst_dev = std::atoi(next().c_str());
    } else if (a == "--methods") {
      c.methods = split_csv(next());
    } else if (a == "--no-check") {
      c.check = false;
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (c.stages < 3) {
    std::fprintf(stderr, "--stages must be >= 3\n");
    std::exit(1);
  }
  return c;
}

// ---------------------------------------------------------------------------
// timing harness
// ---------------------------------------------------------------------------

struct Result {
  bool ok = false;
  double avg_us = 0;     // back-to-back, device time / iters
  double p50_us = 0;     // one-shot latency, median
  double p99_us = 0;     // one-shot latency, tail (p99, or max for few samples)
  double gbps = 0;       // from avg_us
  int verify = -1;       // -1 skipped, 0 mismatch, 1 ok
  std::string note;
};

template <typename Launch>
static bool time_transfer(Launch&& launch, uint64_t bytes, cudaStream_t stream,
                          const Config& cfg, const char* tag, Result* out) {
  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < cfg.warmup; ++i) launch(stream);
  if (!cuda_ok(tag)) {
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return false;
  }

  // Back-to-back throughput: launches overlap with execution, so this is the
  // number that matters for sustained streaming.
  CUDA_CHECK(cudaEventRecord(beg, stream));
  for (int i = 0; i < cfg.iters; ++i) launch(stream);
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  out->avg_us = (double)ms * 1000.0 / cfg.iters;
  out->gbps = (double)bytes / (out->avg_us * 1e-6) / 1e9;

  // One-shot latency: fully serialized, includes the launch->completion path.
  const int nlat = std::min(cfg.iters, 64);
  std::vector<double> lat;
  lat.reserve(nlat);
  for (int i = 0; i < nlat; ++i) {
    CUDA_CHECK(cudaEventRecord(beg, stream));
    launch(stream);
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    lat.push_back((double)ms * 1000.0);
  }
  std::sort(lat.begin(), lat.end());
  out->p50_us = lat[lat.size() / 2];
  // Tail latency: the number a fused pipeline's consumer actually waits on.
  out->p99_us = lat[std::min(lat.size() - 1, lat.size() * 99 / 100)];

  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  return cuda_ok(tag);
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

struct TileSetup {
  uint32_t tile_bytes;
  uint32_t tile_stride;  // 128B-aligned, for shared memory
  uint32_t num_tiles;
  size_t smem_bytes;
  int stages;
  int blocks;
  bool token_aligned;
};

static uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// Pipeline depths we instantiate the TMA kernels for.
static bool stages_supported(int s) {
  return s == 3 || s == 4 || s == 5 || s == 6 || s == 8;
}

// Picks tile size + pipeline depth that fit into the opt-in shared memory
// budget. Falls back to sub-token tiles if a single token is too large.
static bool plan_tiles(const Config& cfg, uint64_t total_bytes,
                       uint32_t hidden_bytes, size_t smem_budget, int num_sms,
                       TileSetup* out, std::string* err) {
  int stages = cfg.stages;
  uint32_t tile_bytes = 0;
  bool token_aligned = true;

  for (; stages >= 3; --stages) {
    if (!stages_supported(stages)) continue;
    // Reserve room for the mbarrier array plus alignment slack.
    const size_t payload = smem_budget > (size_t)stages * 8 + 256
                               ? smem_budget - (size_t)stages * 8 - 256
                               : 0;
    const size_t per_stage = payload / stages;
    if (per_stage < 128) continue;

    if (cfg.tile_tokens > 0) {
      const uint64_t want = (uint64_t)cfg.tile_tokens * hidden_bytes;
      if (want > per_stage) continue;  // try fewer stages
      tile_bytes = (uint32_t)want;
    } else if (hidden_bytes <= per_stage) {
      uint64_t ntok = per_stage / hidden_bytes;
      ntok = std::min<uint64_t>(ntok, total_bytes / hidden_bytes);
      ntok = std::max<uint64_t>(ntok, 1);
      tile_bytes = (uint32_t)(ntok * hidden_bytes);
    } else {
      // One token does not fit: split within a token, 16B granularity.
      tile_bytes = (uint32_t)(per_stage / 16 * 16);
      token_aligned = false;
    }
    if (tile_bytes >= 16) break;
  }

  if (stages < 3 || tile_bytes < 16) {
    *err = "cannot fit a tile into shared memory; lower --tile-tokens or --stages";
    return false;
  }
  if (tile_bytes > total_bytes) tile_bytes = (uint32_t)total_bytes;

  out->stages = stages;
  out->tile_bytes = tile_bytes;
  out->tile_stride = round_up(tile_bytes, 128);
  out->num_tiles = (uint32_t)((total_bytes + tile_bytes - 1) / tile_bytes);
  out->smem_bytes = (size_t)out->tile_stride * stages + (size_t)stages * 8;
  out->token_aligned = token_aligned;
  out->blocks = cfg.blocks > 0
                    ? cfg.blocks
                    : (int)std::min<uint64_t>(out->num_tiles, (uint64_t)num_sms);
  if (out->blocks < 1) out->blocks = 1;
  if (out->smem_bytes > smem_budget) {
    *err = "shared memory plan overflowed the budget";
    return false;
  }
  return true;
}

using TmaKernel = void (*)(const uint8_t*, uint8_t*, uint64_t, uint32_t,
                           uint32_t, uint32_t);

enum class TmaMode { kFullCopy, kStoreOnly, kLoadOnly };

template <int STAGES>
static TmaKernel tma_kernel_for(TmaMode mode) {
  switch (mode) {
    case TmaMode::kStoreOnly: return tma_store_only_kernel<STAGES>;
    case TmaMode::kLoadOnly: return tma_load_only_kernel<STAGES>;
    default: return tma_p2p_kernel<STAGES>;
  }
}

static TmaKernel pick_tma_kernel(int stages, TmaMode mode) {
  switch (stages) {
    case 3: return tma_kernel_for<3>(mode);
    case 4: return tma_kernel_for<4>(mode);
    case 5: return tma_kernel_for<5>(mode);
    case 6: return tma_kernel_for<6>(mode);
    case 8: return tma_kernel_for<8>(mode);
    default: return nullptr;
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static bool wants(const Config& cfg, const char* m) {
  for (auto& s : cfg.methods)
    if (s == m) return true;
  return false;
}

static void fill_pattern(std::vector<uint32_t>& h, uint64_t nwords) {
  h.resize(nwords);
  for (uint64_t i = 0; i < nwords; ++i)
    h[i] = (uint32_t)(i * 2654435761ull) ^ 0x9e3779b9u;
}

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return 1;
  }
  if (cfg.src_dev >= ndev || cfg.dst_dev >= ndev || cfg.src_dev == cfg.dst_dev) {
    std::fprintf(stderr, "bad --src/--dst\n");
    return 1;
  }

  cudaDeviceProp psrc, pdst;
  CUDA_CHECK(cudaGetDeviceProperties(&psrc, cfg.src_dev));
  CUDA_CHECK(cudaGetDeviceProperties(&pdst, cfg.dst_dev));
  char bus_src[64], bus_dst[64];
  CUDA_CHECK(cudaDeviceGetPCIBusId(bus_src, sizeof(bus_src), cfg.src_dev));
  CUDA_CHECK(cudaDeviceGetPCIBusId(bus_dst, sizeof(bus_dst), cfg.dst_dev));

  int can_peer = 0, perf_rank = -1, native_atomic = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, cfg.src_dev, cfg.dst_dev));
  if (can_peer) {
    CUDA_CHECK(cudaDeviceGetP2PAttribute(&perf_rank, cudaDevP2PAttrPerformanceRank,
                                         cfg.src_dev, cfg.dst_dev));
    CUDA_CHECK(cudaDeviceGetP2PAttribute(&native_atomic,
                                         cudaDevP2PAttrNativeAtomicSupported,
                                         cfg.src_dev, cfg.dst_dev));
  }

  std::printf("=== P2P transfer microbenchmark: Copy Engine vs TMA ===\n");
  std::printf("src GPU %d  %-28s sm_%d%d  %s\n", cfg.src_dev, psrc.name,
              psrc.major, psrc.minor, bus_src);
  std::printf("dst GPU %d  %-28s sm_%d%d  %s\n", cfg.dst_dev, pdst.name,
              pdst.major, pdst.minor, bus_dst);
  std::printf("P2P access: %s", can_peer ? "yes" : "NO");
  if (can_peer)
    std::printf("   perf-rank=%d  native-atomics=%d", perf_rank, native_atomic);
  std::printf("\n");

  if (!can_peer) {
    std::printf(
        "\n[warn] direct peer access unavailable. `ce` will stage through the\n"
        "       host (much slower); `sm`/`tma`/`tma_store` cannot run at all.\n");
  }

  // Enable bidirectional peer access so both the copy engine and the SM/TMA
  // paths can address the remote allocation directly.
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  if (can_peer) {
    cudaError_t e = cudaDeviceEnablePeerAccess(cfg.dst_dev, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    else cudaGetLastError();
  }
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  if (can_peer) {
    cudaError_t e = cudaDeviceEnablePeerAccess(cfg.src_dev, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    else cudaGetLastError();
  }
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));

  size_t smem_budget = 0;
  {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      cfg.src_dev));
    smem_budget = (size_t)v;
  }
  std::printf("max opt-in shared memory / CTA: %zu B    SMs: %d\n", smem_budget,
              psrc.multiProcessorCount);

  const uint32_t hidden_bytes = (uint32_t)(cfg.hidden * cfg.dtype_bytes);
  if (hidden_bytes % 16 != 0) {
    std::fprintf(stderr,
                 "hidden*sizeof(dtype) = %u B is not 16B-aligned; cp.async.bulk\n"
                 "requires 16B granularity. Pick a different hidden/dtype.\n",
                 hidden_bytes);
    return 1;
  }
  std::printf("dtype=%s (%u B)  hidden=%llu  -> %u B/token\n\n", cfg.dtype_name.c_str(),
              cfg.dtype_bytes, (unsigned long long)cfg.hidden, hidden_bytes);

  // Allocate once for the largest requested size and reuse.
  uint64_t max_tokens = 0;
  for (auto t : cfg.token_list) max_tokens = std::max(max_tokens, t);
  const uint64_t max_bytes = max_tokens * hidden_bytes;

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaMalloc(&d_src, max_bytes));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaMalloc(&d_dst, max_bytes));
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));

  std::vector<uint32_t> h_ref, h_out;
  fill_pattern(h_ref, max_bytes / 4);
  h_out.resize(max_bytes / 4);
  CUDA_CHECK(cudaMemcpy(d_src, h_ref.data(), max_bytes, cudaMemcpyHostToDevice));

  // One stream per device: work enqueued on a stream is executed by the GPU
  // that owns it, which is exactly what distinguishes push from pull.
  cudaStream_t stream, stream_dst;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_dst, cudaStreamNonBlocking));
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));

  for (uint64_t tokens : cfg.token_list) {
    const uint64_t bytes = tokens * hidden_bytes;
    if (bytes == 0) continue;

    TileSetup ts{};
    std::string err;
    if (!plan_tiles(cfg, bytes, hidden_bytes, smem_budget,
                    psrc.multiProcessorCount, &ts, &err)) {
      std::printf("tokens=%llu : %s\n", (unsigned long long)tokens, err.c_str());
      continue;
    }

    const std::string tile_desc =
        ts.token_aligned ? std::to_string(ts.tile_bytes / hidden_bytes) + " tok"
                         : std::string("sub-token");
    std::printf("--- tokens=%llu   payload=%.3f MiB   tile=%u B (%s)   "
                "tiles=%u   stages=%d   blocks=%d   smem=%zu B ---\n",
                (unsigned long long)tokens, bytes / 1048576.0, ts.tile_bytes,
                tile_desc.c_str(), ts.num_tiles, ts.stages, ts.blocks,
                ts.smem_bytes);
    std::printf("%-11s %12s %12s %12s %12s %8s\n", "method", "avg(us)",
                "p50(us)", "p99(us)", "BW(GB/s)", "verify");

    auto report = [&](const char* name, const Result& r) {
      if (!r.ok) {
        std::printf("%-11s %12s %12s %12s %12s %8s  %s\n", name, "-", "-", "-",
                    "-", "-", r.note.empty() ? "failed" : r.note.c_str());
        return;
      }
      const char* v = r.verify < 0 ? "skip" : (r.verify ? "ok" : "MISMATCH");
      std::printf("%-11s %12.2f %12.2f %12.2f %12.2f %8s%s%s\n", name, r.avg_us,
                  r.p50_us, r.p99_us, r.gbps, v, r.note.empty() ? "" : "  ",
                  r.note.c_str());
    };

    // Verifies d_dst against the reference pattern for `bytes` bytes.
    auto verify_dst = [&]() -> int {
      if (!cfg.check) return -1;
      CUDA_CHECK(cudaMemcpy(h_out.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
      return std::memcmp(h_out.data(), h_ref.data(), bytes) == 0 ? 1 : 0;
    };
    auto clear_dst = [&]() {
      CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
      CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    };

    // ---- ce / ce_pull ------------------------------------------------------
    // Same API call either way; the stream's device decides which GPU's copy
    // engine executes it: src stream -> src engine writes (push), dst stream
    // -> dst engine reads (pull).
    for (int pull = 0; pull < 2; ++pull) {
      const char* name = pull ? "ce_pull" : "ce";
      if (!wants(cfg, name)) continue;
      clear_dst();
      const int exec_dev = pull ? cfg.dst_dev : cfg.src_dev;
      cudaStream_t xs = pull ? stream_dst : stream;
      CUDA_CHECK(cudaSetDevice(exec_dev));
      Result r;
      auto launch = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, cfg.dst_dev, d_src, cfg.src_dev,
                                       bytes, s));
      };
      r.ok = time_transfer(launch, bytes, xs, cfg, name, &r);
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      if (r.ok) r.verify = verify_dst();
      report(name, r);
    }

    // ---- sm / sm_pull ------------------------------------------------------
    // Identical kernel; launched on dst it loads over PCIe (non-posted reads)
    // instead of storing over PCIe (posted writes).
    for (int pull = 0; pull < 2; ++pull) {
      const char* name = pull ? "sm_pull" : "sm";
      if (!wants(cfg, name)) continue;
      if (!can_peer) {
        Result r;
        r.note = "needs P2P";
        report(name, r);
        continue;
      }
      clear_dst();
      const uint64_t nvec = bytes / sizeof(uint4);
      const int nsm =
          pull ? pdst.multiProcessorCount : psrc.multiProcessorCount;
      int blocks = cfg.blocks > 0
                       ? cfg.blocks
                       : (int)std::min<uint64_t>(
                             (nvec + kSmThreads - 1) / kSmThreads,
                             (uint64_t)nsm * 4);
      if (blocks < 1) blocks = 1;
      const int exec_dev = pull ? cfg.dst_dev : cfg.src_dev;
      cudaStream_t xs = pull ? stream_dst : stream;
      CUDA_CHECK(cudaSetDevice(exec_dev));
      Result r;
      auto launch = [&](cudaStream_t s) {
        sm_p2p_kernel<<<blocks, kSmThreads, 0, s>>>(
            reinterpret_cast<const uint4*>(d_src),
            reinterpret_cast<uint4*>(d_dst), nvec);
      };
      r.ok = time_transfer(launch, bytes, xs, cfg, name, &r);
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      if (r.ok) r.verify = verify_dst();
      report(name, r);
    }

    // ---- tma family --------------------------------------------------------
    struct TmaVariant {
      const char* name;
      TmaMode mode;
      bool pull;              // executes on the dst GPU
      const char* skip_note;  // non-null => verification skipped by design
    };
    const TmaVariant tma_variants[] = {
        {"tma", TmaMode::kFullCopy, false, nullptr},
        {"tma_pull", TmaMode::kFullCopy, true, nullptr},
        {"tma_store", TmaMode::kStoreOnly, false,
         "payload not a faithful copy by design"},
        {"tma_load", TmaMode::kLoadOnly, true, "payload stays in smem by design"},
    };
    for (const auto& v : tma_variants) {
      if (!wants(cfg, v.name)) continue;
      if (!can_peer) {
        Result r;
        r.note = "needs P2P";
        report(v.name, r);
        continue;
      }
      TmaKernel k = pick_tma_kernel(ts.stages, v.mode);
      if (!k) {
        Result r;
        r.note = "no instantiation for this --stages";
        report(v.name, r);
        continue;
      }
      clear_dst();
      const int exec_dev = v.pull ? cfg.dst_dev : cfg.src_dev;
      CUDA_CHECK(cudaSetDevice(exec_dev));
      // The opt-in smem attribute is per-device state; set it on the device
      // that will run the kernel.
      cudaError_t e = cudaFuncSetAttribute(
          k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ts.smem_bytes);
      if (e != cudaSuccess) {
        Result r;
        r.note = cudaGetErrorString(e);
        report(v.name, r);
        cudaGetLastError();
        CUDA_CHECK(cudaSetDevice(cfg.src_dev));
        continue;
      }
      Result r;
      auto launch = [&](cudaStream_t s) {
        k<<<ts.blocks, kTmaThreads, ts.smem_bytes, s>>>(
            d_src, d_dst, bytes, ts.tile_bytes, ts.tile_stride, ts.num_tiles);
      };
      r.ok = time_transfer(launch, bytes, v.pull ? stream_dst : stream, cfg,
                           v.name, &r);
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      if (r.ok) {
        if (v.skip_note) {
          r.verify = -1;
          r.note = v.skip_note;
        } else {
          r.verify = verify_dst();
        }
      }
      report(v.name, r);
    }
    std::printf("\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaStreamDestroy(stream_dst));
  CUDA_CHECK(cudaFree(d_dst));
  return 0;
}
