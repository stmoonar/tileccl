// pk_bw_sweep.cu
//
// Microbenchmark reproducing the transfer-mechanism study from the
// ParallelKittens paper (arXiv:2511.13940, Section 3.1.2):
//
//   Figure 2 : achieved P2P bandwidth vs message granularity, for a fixed
//              total payload (default 1 GiB) split into messages of size S.
//   Figure 3 : achieved P2P bandwidth vs number of SMs driving the transfer,
//              at a fixed message size.
//   Table 1  : the large-message, all-SM end points of both sweeps.
//
// Mechanisms measured (all push: executed by the source GPU):
//   ce   host-initiated copy engine  -- one cudaMemcpyPeerAsync per message
//   tma  device-initiated cp.async.bulk: gmem(local) -> smem -> gmem(peer)
//   reg  device-initiated uint4 ld/st through the peer VA mapping
//
// The paper runs on NVLink (H100/B200); this runs the same sweeps over
// whatever links the two GPUs here (PCIe for sm_120 consumer parts), so the
// absolute numbers differ but the curve shapes are directly comparable.
//
// Message-to-worker mapping, chosen to model each mechanism's real usage:
//   ce   the host enqueues one async copy per message on one stream. For tiny
//        messages the per-call overhead is the point of the experiment, so the
//        message count per iteration is capped (--max-ce-msgs) and bandwidth
//        is computed over the bytes actually moved (flagged '*').
//   tma  messages are distributed round-robin over CTAs; within a CTA a single
//        thread drives an mbarrier-based load->store pipeline. Messages are
//        bounded by shared memory (the paper's "227 KB" analog). Messages that
//        fit a >=3-stage pipeline run fully overlapped; larger ones fall back
//        to a single-buffer load->store loop (flagged '^'); beyond the smem
//        budget the cell is '-' (the paper holds the curve flat there).
//   reg  one warp per message when messages are plentiful; when there are
//        fewer messages than warps, warps gang up on messages (degenerating
//        to a plain grid-stride copy for a single huge message). All accesses
//        are 16 B vectors, coalesced across lanes.
//
// Build:  make            (see tests/Makefile)
// Run:    ./pk_bw_sweep                                  # both sweeps, 1 GiB
//         ./pk_bw_sweep --mode size --sizes 128,2K,64K,2M,64M,1G
//         ./pk_bw_sweep --mode sms --sms 1,2,4,8,16,32 --msg-reg 16K

#include <cuda_runtime.h>

#include <algorithm>
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
    std::fprintf(stderr, "[FAIL] %s -> %s\n", what, cudaGetErrorString(err));
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

__device__ __forceinline__ void tma_load_1d(void* smem_dst, const void* gmem_src,
                                            uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(smem_addr(smem_dst)),
      "l"(gmem_src), "r"(bytes), "r"(smem_addr(bar))
      : "memory");
}

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

template <int N>
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

template <int N>
__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

constexpr uint32_t kTmaThreads = 128;
constexpr uint32_t kRegMaxThreads = 1024;

// Deterministic fill / verify on device, so 1 GiB payloads never cross PCIe
// to the host just for checking.
__device__ __forceinline__ uint32_t pattern_word(uint64_t i) {
  return (uint32_t)(i * 2654435761ull) ^ 0x9e3779b9u;
}

__global__ void fill_pattern_kernel(uint32_t* p, uint64_t nwords) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
  for (; i < nwords; i += stride) p[i] = pattern_word(i);
}

__global__ void verify_pattern_kernel(const uint32_t* p, uint64_t nwords,
                                      unsigned long long* mismatches) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  const uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
  unsigned long long bad = 0;
  for (; i < nwords; i += stride)
    if (p[i] != pattern_word(i)) ++bad;
  if (bad) atomicAdd(mismatches, bad);
}

// Register-op path: plain 128-bit ld/st into the peer VA range, message
// granularity modeled by the warp<->message mapping described in the header.
__global__ __launch_bounds__(kRegMaxThreads) void reg_p2p_kernel(
    const uint4* __restrict__ src, uint4* __restrict__ dst, uint64_t total_vec,
    uint32_t msg_vec) {
  const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t nwarps = (gridDim.x * blockDim.x) >> 5;
  const uint64_t num_msgs = total_vec / msg_vec;
  if (num_msgs >= nwarps) {
    // One warp per message; consecutive lanes read/write consecutive vectors.
    for (uint64_t m = warp; m < num_msgs; m += nwarps) {
      const uint64_t base = m * (uint64_t)msg_vec;
      for (uint32_t i = lane; i < msg_vec; i += 32) dst[base + i] = src[base + i];
    }
  } else {
    // Fewer messages than warps: gang wpm warps onto each message. With a
    // single message this degenerates to a whole-grid grid-stride copy.
    const uint32_t wpm = nwarps / (uint32_t)num_msgs;
    const uint64_t m = warp / wpm;
    if (m >= num_msgs) return;
    const uint64_t base = m * (uint64_t)msg_vec;
    const uint64_t step = (uint64_t)wpm * 32;
    for (uint64_t i = (uint64_t)(warp % wpm) * 32 + lane; i < msg_vec; i += step)
      dst[base + i] = src[base + i];
  }
}

// TMA path, pipelined: gmem(local) -> smem -> gmem(peer) over STAGES buffers.
// Same accounting as tests/p2p_ce_vs_tma.cu: STAGES = loads-in-flight + 1
// buffer being stored + 1 buffer still draining, hence STAGES >= 3.
template <int STAGES>
__global__ __launch_bounds__(kTmaThreads) void tma_pipe_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint64_t total_bytes, uint32_t msg_bytes, uint32_t msg_stride,
    uint32_t num_msgs) {
  static_assert(STAGES >= 3, "need at least 3 stages for the pipeline");
  constexpr uint32_t kInflight = STAGES - 2;

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* buf = smem;
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + STAGES * (size_t)msg_stride);

  // Lane 0 owns the whole pipeline; nobody else touches the buffers.
  if (threadIdx.x != 0) return;

#pragma unroll
  for (int s = 0; s < STAGES; ++s) mbarrier_init(&bar[s], 1);
  fence_proxy_async_shared();

  const uint32_t nblk = gridDim.x;
  const uint32_t my_msgs =
      (num_msgs > blockIdx.x) ? ((num_msgs - blockIdx.x + nblk - 1) / nblk) : 0;
  if (my_msgs == 0) return;

  auto msg_off = [&](uint32_t k) -> uint64_t {
    return (uint64_t)(blockIdx.x + k * nblk) * msg_bytes;
  };
  auto msg_len = [&](uint64_t off) -> uint32_t {
    uint64_t rem = total_bytes - off;
    return (uint32_t)(rem < msg_bytes ? rem : msg_bytes);
  };
  auto issue_load = [&](uint32_t k) {
    const uint32_t s = k % STAGES;
    const uint64_t off = msg_off(k);
    const uint32_t nb = msg_len(off);
    mbarrier_arrive_expect_tx(&bar[s], nb);
    tma_load_1d(buf + (size_t)s * msg_stride, src + off, nb, &bar[s]);
  };

  uint32_t phase_bits = 0;
  uint32_t issued = 0;
  for (; issued < kInflight && issued < my_msgs; ++issued) issue_load(issued);

  for (uint32_t done = 0; done < my_msgs; ++done) {
    const uint32_t s = done % STAGES;
    mbarrier_wait(&bar[s], (phase_bits >> s) & 1u);
    phase_bits ^= (1u << s);

    const uint64_t off = msg_off(done);
    tma_store_1d(dst + off, buf + (size_t)s * msg_stride, msg_len(off));
    tma_store_commit();

    if (issued < my_msgs) {
      tma_store_wait_read<1>();
      issue_load(issued++);
    }
  }
  // Ensure the writes actually landed on the peer before the kernel exits.
  tma_store_wait_all<0>();
}

// TMA path, single-buffer fallback for messages too large to triple-buffer.
// Load and store serialize per message, but the local gmem->smem load is
// orders of magnitude faster than the P2P store, so the cost is small.
__global__ __launch_bounds__(kTmaThreads) void tma_single_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    uint64_t total_bytes, uint32_t msg_bytes, uint32_t msg_stride,
    uint32_t num_msgs) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* buf = smem;
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + msg_stride);

  if (threadIdx.x != 0) return;
  mbarrier_init(bar, 1);
  fence_proxy_async_shared();

  const uint32_t nblk = gridDim.x;
  const uint32_t my_msgs =
      (num_msgs > blockIdx.x) ? ((num_msgs - blockIdx.x + nblk - 1) / nblk) : 0;

  uint32_t phase = 0;
  for (uint32_t k = 0; k < my_msgs; ++k) {
    const uint64_t off = (uint64_t)(blockIdx.x + k * nblk) * msg_bytes;
    const uint64_t rem = total_bytes - off;
    const uint32_t nb = (uint32_t)(rem < msg_bytes ? rem : msg_bytes);
    mbarrier_arrive_expect_tx(bar, nb);
    tma_load_1d(buf, src + off, nb, bar);
    mbarrier_wait(bar, phase);
    phase ^= 1;
    tma_store_1d(dst + off, buf, nb);
    tma_store_commit();
    // The buffer is recycled next iteration; only wait for the read side.
    tma_store_wait_read<0>();
  }
  tma_store_wait_all<0>();
}

// ---------------------------------------------------------------------------
// config / CLI
// ---------------------------------------------------------------------------

struct Config {
  uint64_t total = 1ull << 30;  // payload per measurement
  std::vector<uint64_t> sizes;  // message sizes; empty = default sweep
  std::vector<int> sms;         // SM counts; empty = auto
  std::string mode = "both";    // size | sms | both
  std::vector<std::string> methods{"ce", "tma", "reg"};
  int blocks = 0;           // size sweep CTA count; 0 = one per SM
  uint64_t msg_tma = 0;     // sms sweep message size; 0 = auto (max pipelined)
  uint64_t msg_reg = 16384; // sms sweep message size for reg
  int stages = 4;           // preferred TMA pipeline depth (3,4,6,8)
  int reg_threads = 1024;   // threads per CTA for reg (multiple of 32)
  int iters = 10;
  int warmup = 3;
  int src_dev = 0;
  int dst_dev = 1;
  uint64_t max_ce_msgs = 16384;  // per-iteration cap on host-side enqueues
  bool check = true;
};

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

static std::string fmt_size(uint64_t b) {
  char buf[32];
  if (b >= (1ull << 30) && b % (1ull << 30) == 0)
    std::snprintf(buf, sizeof(buf), "%llu GiB", (unsigned long long)(b >> 30));
  else if (b >= (1ull << 20) && b % (1ull << 20) == 0)
    std::snprintf(buf, sizeof(buf), "%llu MiB", (unsigned long long)(b >> 20));
  else if (b >= 1024 && b % 1024 == 0)
    std::snprintf(buf, sizeof(buf), "%llu KiB", (unsigned long long)(b >> 10));
  else
    std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
  return buf;
}

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --mode M          size | sms | both (default both)\n"
      "  --total BYTES     payload per measurement, K/M/G suffix ok (default 1G)\n"
      "  --sizes LIST      message sizes for the size sweep (default\n"
      "                    128,512,2K,8K,32K,64K,128K,512K,2M,8M,32M,128M,512M,1G)\n"
      "  --sms LIST        SM counts for the SM sweep (default auto)\n"
      "  --methods LIST    subset of ce,tma,reg (default all)\n"
      "  --blocks N        CTAs for the size sweep, 0 = one per SM (default 0)\n"
      "  --msg-tma BYTES   TMA message size for the SM sweep, 0 = auto\n"
      "  --msg-reg BYTES   reg message size for the SM sweep (default 16K)\n"
      "  --stages N        preferred TMA pipeline depth: 3,4,6,8 (default 4)\n"
      "  --reg-threads N   threads per CTA for reg, multiple of 32 (default 1024)\n"
      "  --iters N         timed iterations (default 10)\n"
      "  --warmup N        warmup iterations (default 3)\n"
      "  --max-ce-msgs N   cap on copy-engine calls per iteration (default 16384)\n"
      "  --src N / --dst N GPU indices (default 0 / 1)\n"
      "  --no-check        skip verification\n",
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
    if (a == "--mode") {
      c.mode = next();
      if (c.mode != "size" && c.mode != "sms" && c.mode != "both") {
        std::fprintf(stderr, "bad --mode %s\n", c.mode.c_str());
        std::exit(1);
      }
    } else if (a == "--total") {
      c.total = parse_bytes(next());
    } else if (a == "--sizes") {
      for (auto& t : split_csv(next())) c.sizes.push_back(parse_bytes(t));
    } else if (a == "--sms") {
      for (auto& t : split_csv(next())) c.sms.push_back(std::atoi(t.c_str()));
    } else if (a == "--methods") {
      c.methods = split_csv(next());
    } else if (a == "--blocks") {
      c.blocks = std::atoi(next().c_str());
    } else if (a == "--msg-tma") {
      c.msg_tma = parse_bytes(next());
    } else if (a == "--msg-reg") {
      c.msg_reg = parse_bytes(next());
    } else if (a == "--stages") {
      c.stages = std::atoi(next().c_str());
    } else if (a == "--reg-threads") {
      c.reg_threads = std::atoi(next().c_str());
    } else if (a == "--iters") {
      c.iters = std::atoi(next().c_str());
    } else if (a == "--warmup") {
      c.warmup = std::atoi(next().c_str());
    } else if (a == "--max-ce-msgs") {
      c.max_ce_msgs = std::strtoull(next().c_str(), nullptr, 10);
    } else if (a == "--src") {
      c.src_dev = std::atoi(next().c_str());
    } else if (a == "--dst") {
      c.dst_dev = std::atoi(next().c_str());
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
  if (c.stages != 3 && c.stages != 4 && c.stages != 6 && c.stages != 8) {
    std::fprintf(stderr, "--stages must be one of 3,4,6,8\n");
    std::exit(1);
  }
  if (c.reg_threads < 32 || c.reg_threads > (int)kRegMaxThreads ||
      c.reg_threads % 32 != 0) {
    std::fprintf(stderr, "--reg-threads must be a multiple of 32 in [32,1024]\n");
    std::exit(1);
  }
  if (c.total < 16 || c.total % 16 != 0) {
    std::fprintf(stderr, "--total must be a multiple of 16\n");
    std::exit(1);
  }
  return c;
}

// ---------------------------------------------------------------------------
// TMA planning
// ---------------------------------------------------------------------------

static uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

struct TmaPlan {
  bool ok = false;
  bool pipelined = false;
  int stages = 0;
  uint32_t stride = 0;
  size_t smem = 0;
};

// Picks the deepest pipeline (<= want_stages) that fits the smem budget, or
// falls back to a single buffer, or fails if even one message does not fit.
static TmaPlan plan_tma(uint32_t msg_bytes, int want_stages, size_t budget) {
  TmaPlan p;
  const uint32_t stride = round_up(msg_bytes, 128);
  static const int cand[] = {8, 6, 4, 3};
  for (int s : cand) {
    if (s > want_stages) continue;
    const size_t need = (size_t)stride * s + 8u * s;
    if (need <= budget) {
      p.ok = true; p.pipelined = true; p.stages = s;
      p.stride = stride; p.smem = need;
      return p;
    }
  }
  const size_t need = (size_t)stride + 8;
  if (need <= budget) {
    p.ok = true; p.pipelined = false; p.stages = 1;
    p.stride = stride; p.smem = need;
  }
  return p;
}

using TmaKernel = void (*)(const uint8_t*, uint8_t*, uint64_t, uint32_t,
                           uint32_t, uint32_t);

static TmaKernel pick_tma_kernel(const TmaPlan& p) {
  if (!p.pipelined) return tma_single_kernel;
  switch (p.stages) {
    case 3: return tma_pipe_kernel<3>;
    case 4: return tma_pipe_kernel<4>;
    case 6: return tma_pipe_kernel<6>;
    case 8: return tma_pipe_kernel<8>;
    default: return nullptr;
  }
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------

// One table cell: a bandwidth number plus a one-character qualifier.
//   ' ' clean   '*' ce payload capped   '^' tma single-buffer   '!' verify fail
struct Cell {
  bool ran = false;
  double gbps = 0;
  double avg_us = 0;
  char flag = ' ';
  std::string note;  // shown when !ran
};

template <typename Launch>
static bool time_avg_us(Launch&& launch, cudaStream_t stream, int warmup,
                        int iters, const char* tag, double* out_us) {
  for (int i = 0; i < warmup; ++i) launch(stream);
  if (!cuda_ok(tag)) return false;

  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(beg, stream));
  for (int i = 0; i < iters; ++i) launch(stream);
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  *out_us = (double)ms * 1000.0 / iters;
  return cuda_ok(tag);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static bool wants(const Config& cfg, const char* m) {
  for (auto& s : cfg.methods)
    if (s == m) return true;
  return false;
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

  int can_peer = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, cfg.src_dev, cfg.dst_dev));

  std::printf("=== PK transfer-mechanism sweep (paper Fig.2 / Fig.3) ===\n");
  std::printf("src GPU %d  %-28s sm_%d%d  %d SMs\n", cfg.src_dev, psrc.name,
              psrc.major, psrc.minor, psrc.multiProcessorCount);
  std::printf("dst GPU %d  %-28s sm_%d%d\n", cfg.dst_dev, pdst.name, pdst.major,
              pdst.minor);
  std::printf("P2P access: %s   payload/measurement: %s\n",
              can_peer ? "yes" : "NO", fmt_size(cfg.total).c_str());
  if (!can_peer)
    std::printf("[warn] no direct peer access: ce stages through the host,\n"
                "       tma/reg are skipped entirely.\n");

  // Enable bidirectional peer access (harmless if already enabled).
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
  std::printf("max opt-in shared memory / CTA: %zu B\n\n", smem_budget);

  const int num_sms = psrc.multiProcessorCount;

  // Buffers + pattern fill (on-device, no host round trip).
  uint8_t *d_src = nullptr, *d_dst = nullptr;
  unsigned long long* d_bad = nullptr;
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaMalloc(&d_src, cfg.total));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaMalloc(&d_dst, cfg.total));
  CUDA_CHECK(cudaMalloc(&d_bad, sizeof(unsigned long long)));
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  {
    const uint64_t nwords = cfg.total / 4;
    const int blk = (int)std::min<uint64_t>((nwords + 511) / 512, 4096);
    fill_pattern_kernel<<<blk, 512>>>(reinterpret_cast<uint32_t*>(d_src), nwords);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto clear_dst = [&](uint64_t bytes) {
    CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
    CUDA_CHECK(cudaMemset(d_dst, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  };

  // Returns 1 ok, 0 mismatch, -1 skipped. Runs on the dst GPU.
  auto verify_dst = [&](uint64_t bytes) -> int {
    if (!cfg.check) return -1;
    CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
    CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(unsigned long long)));
    const uint64_t nwords = bytes / 4;
    const int blk = (int)std::min<uint64_t>((nwords + 511) / 512, 4096);
    verify_pattern_kernel<<<blk, 512>>>(reinterpret_cast<const uint32_t*>(d_dst),
                                        nwords, d_bad);
    unsigned long long bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    return bad == 0 ? 1 : 0;
  };

  // ---- per-method benchmark closures --------------------------------------

  auto bench_ce = [&](uint64_t S) -> Cell {
    Cell c;
    uint64_t bytes = cfg.total / S * S;
    if (bytes == 0) { c.note = "msg > total"; return c; }
    const uint64_t max_bytes = S * cfg.max_ce_msgs;
    if (bytes > max_bytes) { bytes = max_bytes; c.flag = '*'; }
    clear_dst(bytes);
    auto launch = [&](cudaStream_t s) {
      for (uint64_t off = 0; off < bytes; off += S)
        CUDA_CHECK(cudaMemcpyPeerAsync(d_dst + off, cfg.dst_dev, d_src + off,
                                       cfg.src_dev, S, s));
    };
    if (!time_avg_us(launch, stream, cfg.warmup, cfg.iters, "ce", &c.avg_us))
      { c.note = "failed"; return c; }
    c.ran = true;
    c.gbps = (double)bytes / (c.avg_us * 1e-6) / 1e9;
    if (verify_dst(bytes) == 0) c.flag = '!';
    return c;
  };

  auto bench_tma = [&](uint64_t S, int blocks) -> Cell {
    Cell c;
    if (!can_peer) { c.note = "needs P2P"; return c; }
    if (S < 16 || S % 16 != 0) { c.note = "not 16B-aligned"; return c; }
    const uint64_t bytes = cfg.total / S * S;
    if (bytes == 0) { c.note = "msg > total"; return c; }
    if (S > 0xffffffffull) { c.note = "> smem"; return c; }
    TmaPlan plan = plan_tma((uint32_t)S, cfg.stages, smem_budget);
    if (!plan.ok) { c.note = "> smem"; return c; }
    TmaKernel k = pick_tma_kernel(plan);
    const uint32_t num_msgs = (uint32_t)(bytes / S);
    if (blocks <= 0) blocks = num_sms;
    blocks = (int)std::min<uint32_t>((uint32_t)blocks, num_msgs);
    if (!plan.pipelined) c.flag = '^';
    clear_dst(bytes);
    cudaError_t e = cudaFuncSetAttribute(
        k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)plan.smem);
    if (e != cudaSuccess) { c.note = cudaGetErrorString(e); cudaGetLastError(); return c; }
    auto launch = [&](cudaStream_t s) {
      k<<<blocks, kTmaThreads, plan.smem, s>>>(d_src, d_dst, bytes, (uint32_t)S,
                                               plan.stride, num_msgs);
    };
    if (!time_avg_us(launch, stream, cfg.warmup, cfg.iters, "tma", &c.avg_us))
      { c.note = "failed"; c.flag = ' '; return c; }
    c.ran = true;
    c.gbps = (double)bytes / (c.avg_us * 1e-6) / 1e9;
    if (verify_dst(bytes) == 0) c.flag = '!';
    return c;
  };

  auto bench_reg = [&](uint64_t S, int blocks) -> Cell {
    Cell c;
    if (!can_peer) { c.note = "needs P2P"; return c; }
    if (S < 16 || S % 16 != 0) { c.note = "not 16B-aligned"; return c; }
    const uint64_t bytes = cfg.total / S * S;
    if (bytes == 0) { c.note = "msg > total"; return c; }
    if (blocks <= 0) blocks = num_sms;
    clear_dst(bytes);
    auto launch = [&](cudaStream_t s) {
      reg_p2p_kernel<<<blocks, cfg.reg_threads, 0, s>>>(
          reinterpret_cast<const uint4*>(d_src),
          reinterpret_cast<uint4*>(d_dst), bytes / 16, (uint32_t)(S / 16));
    };
    if (!time_avg_us(launch, stream, cfg.warmup, cfg.iters, "reg", &c.avg_us))
      { c.note = "failed"; return c; }
    c.ran = true;
    c.gbps = (double)bytes / (c.avg_us * 1e-6) / 1e9;
    if (verify_dst(bytes) == 0) c.flag = '!';
    return c;
  };

  auto print_cell = [](const Cell& c) {
    if (c.ran)
      std::printf(" %11.2f%c", c.gbps, c.flag);
    else
      std::printf(" %11s ", c.note.empty() ? "-" : c.note.c_str());
  };

  // ---- Fig. 2: bandwidth vs message size ----------------------------------

  if (cfg.mode == "size" || cfg.mode == "both") {
    std::vector<uint64_t> sizes = cfg.sizes;
    if (sizes.empty())
      sizes = {128,        512,        2048,      8192,      32768,
               65536,      131072,     524288,    2097152,   8388608,
               33554432,   134217728,  536870912, 1073741824ull};
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(),
                               [&](uint64_t s) { return s > cfg.total; }),
                sizes.end());

    std::printf("--- Fig.2: bandwidth (GB/s) vs message size"
                "   [blocks=%d, stages<=%d] ---\n",
                cfg.blocks > 0 ? cfg.blocks : num_sms, cfg.stages);
    std::printf("%-10s", "msg_size");
    for (auto& m : cfg.methods) std::printf(" %12s", m.c_str());
    std::printf("\n");

    for (uint64_t S : sizes) {
      std::printf("%-10s", fmt_size(S).c_str());
      for (auto& m : cfg.methods) {
        Cell c;
        if (m == "ce") c = bench_ce(S);
        else if (m == "tma") c = bench_tma(S, cfg.blocks);
        else if (m == "reg") c = bench_reg(S, cfg.blocks);
        else c.note = "?";
        print_cell(c);
        std::fflush(stdout);
      }
      std::printf("\n");
    }
    std::printf("flags: * ce payload capped at --max-ce-msgs messages/iter"
                "   ^ tma single-buffer (no pipelining)   ! verify FAILED\n\n");
  }

  // ---- Fig. 3: bandwidth vs SM count --------------------------------------

  if (cfg.mode == "sms" || cfg.mode == "both") {
    std::vector<int> sms = cfg.sms;
    if (sms.empty()) {
      static const int cand[] = {1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32,
                                 40, 48, 64, 80, 96, 112, 128, 144, 160, 176};
      for (int s : cand)
        if (s < num_sms) sms.push_back(s);
      sms.push_back(num_sms);
    }
    sms.erase(std::remove_if(sms.begin(), sms.end(),
                             [&](int s) { return s < 1 || s > num_sms; }),
              sms.end());

    // Auto TMA message: the largest that still fits the preferred pipeline.
    uint64_t msg_tma = cfg.msg_tma;
    if (msg_tma == 0) {
      const size_t per_stage = (smem_budget - 8u * cfg.stages) / cfg.stages;
      msg_tma = (uint64_t)(per_stage / 128 * 128);
    }
    msg_tma = std::min(msg_tma, cfg.total);
    const uint64_t msg_reg = std::min(cfg.msg_reg, cfg.total);

    const bool do_tma = wants(cfg, "tma");
    const bool do_reg = wants(cfg, "reg");

    std::printf("--- Fig.3: bandwidth (GB/s) vs SM count"
                "   [msg: tma=%s, reg=%s, reg_threads=%d] ---\n",
                fmt_size(msg_tma).c_str(), fmt_size(msg_reg).c_str(),
                cfg.reg_threads);
    if (wants(cfg, "ce")) {
      Cell c = bench_ce(cfg.total);  // single full-size copy, 0 SMs
      if (c.ran)
        std::printf("copy-engine reference (1 call, %s, 0 SMs): %.2f GB/s\n",
                    fmt_size(cfg.total).c_str(), c.gbps);
    }
    std::printf("%-6s", "sms");
    if (do_tma) std::printf(" %12s", "tma");
    if (do_reg) std::printf(" %12s", "reg");
    std::printf("\n");

    for (int n : sms) {
      std::printf("%-6d", n);
      if (do_tma) { print_cell(bench_tma(msg_tma, n)); std::fflush(stdout); }
      if (do_reg) { print_cell(bench_reg(msg_reg, n)); std::fflush(stdout); }
      std::printf("\n");
    }
    std::printf("flags: ^ tma single-buffer (no pipelining)   ! verify FAILED\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaFree(d_bad));
  CUDA_CHECK(cudaFree(d_dst));
  return 0;
}
