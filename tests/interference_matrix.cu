// interference_matrix.cu
//
// Quantifies the *mutual* interference between P2P communication and compute,
// per (communication mechanism x compute bottleneck) pair. This is the number
// a fusion autotuner actually needs: not "which mover has the highest raw
// bandwidth", but "how much compute does each byte moved through mechanism M
// steal from a kernel bound on resource R, and vice versa".
//
// Experiment A (--mode matrix):
//   Compute-side *probes* -- persistent kernels, each calibrated to saturate
//   one resource, each self-reporting progress through a per-CTA iteration
//   counter in global memory:
//     mma    tensor-core FLOPs (wmma chain, register-resident)
//     ffma   issue/ALU (FFMA dependency chains, register-resident)
//     hbm    HBM streaming bandwidth (working set >> L2)
//     l2     L2 bandwidth (shared read-only working set ~ L2/2)
//     smem   shared-memory / LSU pressure (dependent smem loads)
//   Communication side reuses the three movers from pk_bw_sweep:
//     ce     host-initiated cudaMemcpyPeerAsync per message
//     tma    device-initiated cp.async.bulk pipeline (gmem->smem->peer gmem)
//     reg    device-initiated uint4 ld/st into the peer VA
//   For each (probe, comm) cell we measure comm alone, probe alone, and the
//   two running concurrently, over a host-timed window using counter
//   snapshots. Reported:
//     S_m = BW_alone / BW_overlap        (communication slowdown)
//     S_c = rate_alone / rate_overlap    (compute slowdown)
//   A probe counter delta > 0 during the overlap window is also the proof
//   that the two really were co-resident (not serialized).
//
//   Placement: the probe runs on --probe-dev (src by default; dst measures
//   receiver-side interference, experiment E). When probe and comm kernels
//   share a device, the probe gets (SMs - comm_sms) CTAs; for ce the probe
//   keeps all SMs -- that asymmetry is precisely CE's selling point and is
//   reported in the probe_sms column.
//
// Experiment C (--mode starve):
//   A full-occupancy ffma probe (SMs x 4 CTAs) squats on the source GPU while
//   each mover tries to push one payload. CE should complete unimpeded; tma
//   and reg kernels cannot get resident and starve until the probe exits.
//   This is the progress-model difference that raw bandwidth curves hide.
//
// Caveats (see tests/README.md): lock clocks before trusting slowdowns
// (nvidia-smi -lgc), and remember CTA placement is best-effort -- without
// green contexts the probe and comm CTAs are not hard-partitioned onto
// disjoint SMs.
//
// Build:  make            (see tests/Makefile)
// Run:    ./interference_matrix
//         ./interference_matrix --probes hbm,mma --comms ce,tma --msg 2M
//         ./interference_matrix --probe-dev dst          # receiver-side (E)
//         ./interference_matrix --mode starve

#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
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

static bool cuda_ok(const char* what) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[FAIL] %s -> %s\n", what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// PTX helpers for cp.async.bulk / mbarrier (same as pk_bw_sweep.cu)
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
// communication kernels (identical mechanics to pk_bw_sweep.cu)
// ---------------------------------------------------------------------------

constexpr uint32_t kTmaThreads = 128;
constexpr uint32_t kRegMaxThreads = 1024;
constexpr uint32_t kProbeThreads = 256;

__global__ __launch_bounds__(kRegMaxThreads) void reg_p2p_kernel(
    const uint4* __restrict__ src, uint4* __restrict__ dst, uint64_t total_vec,
    uint32_t msg_vec) {
  const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t nwarps = (gridDim.x * blockDim.x) >> 5;
  const uint64_t num_msgs = total_vec / msg_vec;
  if (num_msgs >= nwarps) {
    for (uint64_t m = warp; m < num_msgs; m += nwarps) {
      const uint64_t base = m * (uint64_t)msg_vec;
      for (uint32_t i = lane; i < msg_vec; i += 32) dst[base + i] = src[base + i];
    }
  } else {
    const uint32_t wpm = nwarps / (uint32_t)num_msgs;
    const uint64_t m = warp / wpm;
    if (m >= num_msgs) return;
    const uint64_t base = m * (uint64_t)msg_vec;
    const uint64_t step = (uint64_t)wpm * 32;
    for (uint64_t i = (uint64_t)(warp % wpm) * 32 + lane; i < msg_vec; i += step)
      dst[base + i] = src[base + i];
  }
}

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
  tma_store_wait_all<0>();
}

// ---------------------------------------------------------------------------
// probe kernels
//
// All probes share the same contract: run until *stop != 0, and after every
// outer iteration thread 0 publishes the CTA's iteration count to
// counters[blockIdx.x] with a volatile store. The host measures throughput by
// snapshotting the counter array twice across a timed window, so kernel
// launch/teardown never pollutes the number, and a nonzero delta during the
// overlap window proves the probe was genuinely co-resident with the comm.
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool probe_stop(const uint32_t* stop) {
  return *reinterpret_cast<const volatile uint32_t*>(stop) != 0;
}

__device__ __forceinline__ void probe_publish(unsigned long long* counters,
                                              unsigned long long it) {
  if (threadIdx.x == 0)
    reinterpret_cast<volatile unsigned long long*>(counters)[blockIdx.x] = it;
}

// Issue/ALU probe: register-resident FFMA dependency chains, no memory
// traffic beyond the stop flag. Work per CTA iteration:
// kProbeThreads * 512 FMA = that many * 2 FLOP.
constexpr int kFfmaUnroll = 512;

__global__ __launch_bounds__(kProbeThreads) void probe_ffma_kernel(
    const uint32_t* stop, unsigned long long* counters, float* sink) {
  float a = threadIdx.x * 1e-3f + 1.0f;
  const float b = 1.0000001f, c = 1e-7f;
  unsigned long long it = 0;
  while (!probe_stop(stop)) {
#pragma unroll
    for (int u = 0; u < kFfmaUnroll; ++u) a = fmaf(a, b, c);
    ++it;
    probe_publish(counters, it);
  }
  if (a == 12345.678f) *sink = a;  // never true; defeats DCE
}

// Tensor-core probe: register-resident wmma chain. Work per CTA iteration:
// (kProbeThreads/32) warps * kMmaUnroll * 2*16*16*16 FLOP.
constexpr int kMmaUnroll = 64;

__global__ __launch_bounds__(kProbeThreads) void probe_mma_kernel(
    const uint32_t* stop, unsigned long long* counters, __half* sink) {
  using namespace nvcuda::wmma;
  fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
  fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
  fragment<accumulator, 16, 16, 16, __half> fc;
  fill_fragment(fa, __float2half(1.001f));
  fill_fragment(fb, __float2half(0.999f));
  fill_fragment(fc, __float2half(0.0f));
  unsigned long long it = 0;
  while (!probe_stop(stop)) {
#pragma unroll
    for (int u = 0; u < kMmaUnroll; ++u) mma_sync(fc, fa, fb, fc);
    ++it;
    probe_publish(counters, it);
  }
  if (__half2float(fc.x[0]) == 12345.0f) *sink = fc.x[0];
}

// HBM probe: each CTA streams its private slice (read + write, .cg to skip
// L1); the total working set is sized well past L2 so this is DRAM traffic.
// Work per CTA iteration: slice_vec(blockIdx) * 16 bytes read + written.
__global__ __launch_bounds__(kProbeThreads) void probe_hbm_kernel(
    const uint32_t* stop, unsigned long long* counters,
    const uint4* __restrict__ src, uint4* __restrict__ dst, uint64_t nvec) {
  const uint64_t per = (nvec + gridDim.x - 1) / gridDim.x;
  const uint64_t lo = blockIdx.x * per;
  const uint64_t hi = lo + per < nvec ? lo + per : nvec;
  unsigned long long it = 0;
  while (!probe_stop(stop)) {
    for (uint64_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
      dst[i] = __ldcg(&src[i]);
    ++it;
    probe_publish(counters, it);
  }
}

// L2 probe: every CTA scans the same read-only buffer (sized ~L2/2), so the
// working set stays L2-resident and the probe is sensitive to L2 pollution
// and L2 bandwidth contention, not DRAM. Work per CTA iteration: nvec * 16 B.
__global__ __launch_bounds__(kProbeThreads) void probe_l2_kernel(
    const uint32_t* stop, unsigned long long* counters,
    const uint4* __restrict__ buf, uint64_t nvec, uint4* sink) {
  uint4 acc = {0u, 0u, 0u, 0u};
  unsigned long long it = 0;
  while (!probe_stop(stop)) {
    for (uint64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
      const uint4 v = __ldcg(&buf[i]);
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    ++it;
    probe_publish(counters, it);
  }
  if (acc.x == 0xdeadbeefu && acc.y == 0x12345678u) *sink = acc;
}

// Shared-memory/LSU probe: dependent smem loads, one in flight per thread.
// Work per CTA iteration: kProbeThreads * kSmemRounds * 4 B loaded.
constexpr int kSmemWords = 8192;  // 32 KiB
constexpr int kSmemRounds = 256;

__global__ __launch_bounds__(kProbeThreads) void probe_smem_kernel(
    const uint32_t* stop, unsigned long long* counters, uint32_t* sink) {
  __shared__ uint32_t sm[kSmemWords];
  for (int i = threadIdx.x; i < kSmemWords; i += blockDim.x)
    sm[i] = i * 2654435761u;
  __syncthreads();
  uint32_t acc = threadIdx.x;
  unsigned long long it = 0;
  while (!probe_stop(stop)) {
#pragma unroll 4
    for (int r = 0; r < kSmemRounds; ++r)
      acc += sm[(acc ^ (uint32_t)r) & (kSmemWords - 1)];
    ++it;
    probe_publish(counters, it);
  }
  if (acc == 0xdeadbeefu) *sink = acc;
}

// ---------------------------------------------------------------------------
// config / CLI
// ---------------------------------------------------------------------------

struct Config {
  std::string mode = "matrix";  // matrix | starve | both
  std::vector<std::string> probes{"mma", "ffma", "hbm", "l2", "smem"};
  std::vector<std::string> comms{"ce", "tma", "reg"};
  uint64_t total = 64ull << 20;   // comm payload per iteration
  uint64_t msg = 2ull << 20;      // comm message size
  int comm_sms = 8;               // CTAs for tma/reg comm kernels
  int reg_threads = 512;
  int stages = 4;
  int probe_sms = 0;              // 0 = auto (SMs - comm_sms when sharing)
  double window_ms = 300;         // measurement window per cell
  double starve_ms = 1000;        // starvation deadline
  uint64_t hbm_buf = 256ull << 20;
  int src_dev = 0;
  int dst_dev = 1;
  std::string probe_dev = "src";  // src | dst
  std::string csv;                // optional csv output path
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

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --mode M          matrix | starve | both (default matrix)\n"
      "  --probes LIST     subset of mma,ffma,hbm,l2,smem (default all)\n"
      "  --comms LIST      subset of ce,tma,reg (default all)\n"
      "  --total BYTES     comm payload per iteration (default 64M)\n"
      "  --msg BYTES       comm message size (default 2M)\n"
      "  --comm-sms N      CTAs for the tma/reg comm kernels (default 8)\n"
      "  --probe-sms N     probe CTAs, 0 = auto: all SMs, minus comm-sms when\n"
      "                    the probe shares a device with a comm kernel\n"
      "  --reg-threads N   threads/CTA for reg (default 512)\n"
      "  --stages N        TMA pipeline depth 3,4,6,8 (default 4)\n"
      "  --window-ms MS    measurement window per cell (default 300)\n"
      "  --starve-ms MS    starvation deadline (default 1000)\n"
      "  --hbm-buf BYTES   hbm probe working set (default 256M)\n"
      "  --probe-dev D     src | dst -- where the probe runs (default src;\n"
      "                    dst measures receiver-side interference)\n"
      "  --src N / --dst N GPU indices (default 0 / 1)\n"
      "  --csv PATH        also append machine-readable rows to PATH\n",
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
    if (a == "--mode") c.mode = next();
    else if (a == "--probes") c.probes = split_csv(next());
    else if (a == "--comms") c.comms = split_csv(next());
    else if (a == "--total") c.total = parse_bytes(next());
    else if (a == "--msg") c.msg = parse_bytes(next());
    else if (a == "--comm-sms") c.comm_sms = std::atoi(next().c_str());
    else if (a == "--probe-sms") c.probe_sms = std::atoi(next().c_str());
    else if (a == "--reg-threads") c.reg_threads = std::atoi(next().c_str());
    else if (a == "--stages") c.stages = std::atoi(next().c_str());
    else if (a == "--window-ms") c.window_ms = std::atof(next().c_str());
    else if (a == "--starve-ms") c.starve_ms = std::atof(next().c_str());
    else if (a == "--hbm-buf") c.hbm_buf = parse_bytes(next());
    else if (a == "--probe-dev") c.probe_dev = next();
    else if (a == "--src") c.src_dev = std::atoi(next().c_str());
    else if (a == "--dst") c.dst_dev = std::atoi(next().c_str());
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (c.mode != "matrix" && c.mode != "starve" && c.mode != "both") {
    std::fprintf(stderr, "bad --mode %s\n", c.mode.c_str());
    std::exit(1);
  }
  if (c.stages != 3 && c.stages != 4 && c.stages != 6 && c.stages != 8) {
    std::fprintf(stderr, "--stages must be one of 3,4,6,8\n");
    std::exit(1);
  }
  if (c.probe_dev != "src" && c.probe_dev != "dst") {
    std::fprintf(stderr, "--probe-dev must be src or dst\n");
    std::exit(1);
  }
  if (c.msg > c.total) c.msg = c.total;
  if (c.msg < 16 || c.msg % 16 != 0 || c.total % c.msg != 0) {
    std::fprintf(stderr, "--msg must be a 16B multiple dividing --total\n");
    std::exit(1);
  }
  return c;
}

// ---------------------------------------------------------------------------
// TMA planning (same policy as pk_bw_sweep.cu)
// ---------------------------------------------------------------------------

static uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

struct TmaPlan {
  bool ok = false;
  int stages = 0;
  uint32_t stride = 0;
  size_t smem = 0;
};

static TmaPlan plan_tma(uint32_t msg_bytes, int want_stages, size_t budget) {
  TmaPlan p;
  const uint32_t stride = round_up(msg_bytes, 128);
  static const int cand[] = {8, 6, 4, 3};
  for (int s : cand) {
    if (s > want_stages) continue;
    const size_t need = (size_t)stride * s + 8u * s;
    if (need <= budget) {
      p.ok = true; p.stages = s; p.stride = stride; p.smem = need;
      return p;
    }
  }
  return p;
}

using TmaKernel = void (*)(const uint8_t*, uint8_t*, uint64_t, uint32_t,
                           uint32_t, uint32_t);

static TmaKernel pick_tma_kernel(int stages) {
  switch (stages) {
    case 3: return tma_pipe_kernel<3>;
    case 4: return tma_pipe_kernel<4>;
    case 6: return tma_pipe_kernel<6>;
    case 8: return tma_pipe_kernel<8>;
    default: return nullptr;
  }
}

// ---------------------------------------------------------------------------
// probe runner: launch / snapshot / stop
// ---------------------------------------------------------------------------

struct ProbeCtx {
  int dev = -1;
  int blocks = 0;
  // device buffers (allocated once per device, reused across probes)
  uint32_t* d_stop = nullptr;
  unsigned long long* d_counters = nullptr;
  void* d_sink = nullptr;
  uint4* d_hbm = nullptr;     // hbm probe: [src half | dst half]
  uint64_t hbm_nvec = 0;      // vectors per half
  uint4* d_l2 = nullptr;
  uint64_t l2_nvec = 0;
  cudaStream_t probe_stream = nullptr;  // probe kernel
  cudaStream_t ctrl_stream = nullptr;   // stop flag + counter snapshots
  unsigned long long* h_snap = nullptr; // pinned, kMaxBlocks entries
};

constexpr int kMaxProbeBlocks = 4096;

static void probe_ctx_init(ProbeCtx* p, int dev, uint64_t hbm_buf, int l2_size) {
  p->dev = dev;
  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaMalloc(&p->d_stop, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&p->d_counters,
                        kMaxProbeBlocks * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&p->d_sink, 64));
  CUDA_CHECK(cudaMalloc(&p->d_hbm, hbm_buf));
  CUDA_CHECK(cudaMemset(p->d_hbm, 1, hbm_buf));
  p->hbm_nvec = hbm_buf / sizeof(uint4) / 2;
  const uint64_t l2_bytes =
      std::max<uint64_t>(1u << 20, (uint64_t)l2_size / 2) / 16 * 16;
  CUDA_CHECK(cudaMalloc(&p->d_l2, l2_bytes));
  CUDA_CHECK(cudaMemset(p->d_l2, 2, l2_bytes));
  p->l2_nvec = l2_bytes / sizeof(uint4);
  CUDA_CHECK(cudaStreamCreateWithFlags(&p->probe_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&p->ctrl_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMallocHost(&p->h_snap,
                            kMaxProbeBlocks * sizeof(unsigned long long)));
}

// Work done per CTA outer iteration, in the probe's natural unit.
// Units: mma/ffma -> FLOP, hbm -> bytes moved (r+w), l2 -> bytes read,
// smem -> bytes loaded.
static double probe_work_per_iter(const std::string& name, const ProbeCtx& p,
                                  int block, int blocks) {
  if (name == "ffma") return (double)kProbeThreads * kFfmaUnroll * 2.0;
  if (name == "mma")
    return (double)(kProbeThreads / 32) * kMmaUnroll * 2.0 * 16 * 16 * 16;
  if (name == "hbm") {
    const uint64_t per = (p.hbm_nvec + blocks - 1) / blocks;
    const uint64_t lo = (uint64_t)block * per;
    const uint64_t hi = std::min<uint64_t>(lo + per, p.hbm_nvec);
    return hi > lo ? (double)(hi - lo) * 16.0 * 2.0 : 0.0;
  }
  if (name == "l2") return (double)p.l2_nvec * 16.0;
  if (name == "smem") return (double)kProbeThreads * kSmemRounds * 4.0;
  return 0.0;
}

static const char* probe_unit(const std::string& name) {
  if (name == "ffma") return "GFLOP/s";
  if (name == "mma") return "TFLOP/s";
  return "GB/s";
}

static double probe_unit_scale(const std::string& name) {
  if (name == "mma") return 1e12;
  return 1e9;  // GFLOP/s or GB/s
}

static void probe_launch(const std::string& name, ProbeCtx* p, int blocks) {
  CUDA_CHECK(cudaSetDevice(p->dev));
  p->blocks = blocks;
  CUDA_CHECK(cudaMemset(p->d_stop, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(p->d_counters, 0,
                        kMaxProbeBlocks * sizeof(unsigned long long)));
  CUDA_CHECK(cudaDeviceSynchronize());
  if (name == "ffma")
    probe_ffma_kernel<<<blocks, kProbeThreads, 0, p->probe_stream>>>(
        p->d_stop, p->d_counters, (float*)p->d_sink);
  else if (name == "mma")
    probe_mma_kernel<<<blocks, kProbeThreads, 0, p->probe_stream>>>(
        p->d_stop, p->d_counters, (__half*)p->d_sink);
  else if (name == "hbm")
    probe_hbm_kernel<<<blocks, kProbeThreads, 0, p->probe_stream>>>(
        p->d_stop, p->d_counters, p->d_hbm, p->d_hbm + p->hbm_nvec,
        p->hbm_nvec);
  else if (name == "l2")
    probe_l2_kernel<<<blocks, kProbeThreads, 0, p->probe_stream>>>(
        p->d_stop, p->d_counters, p->d_l2, p->l2_nvec, (uint4*)p->d_sink);
  else if (name == "smem")
    probe_smem_kernel<<<blocks, kProbeThreads, 0, p->probe_stream>>>(
        p->d_stop, p->d_counters, (uint32_t*)p->d_sink);
  CUDA_CHECK(cudaGetLastError());
}

// Snapshot the counters while the probe keeps running.
static void probe_snapshot(ProbeCtx* p, std::vector<unsigned long long>* out) {
  CUDA_CHECK(cudaSetDevice(p->dev));
  CUDA_CHECK(cudaMemcpyAsync(p->h_snap, p->d_counters,
                             p->blocks * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, p->ctrl_stream));
  CUDA_CHECK(cudaStreamSynchronize(p->ctrl_stream));
  out->assign(p->h_snap, p->h_snap + p->blocks);
}

static void probe_stop_and_join(ProbeCtx* p) {
  CUDA_CHECK(cudaSetDevice(p->dev));
  static const uint32_t one = 1;
  CUDA_CHECK(cudaMemcpyAsync(p->d_stop, &one, sizeof(one),
                             cudaMemcpyHostToDevice, p->ctrl_stream));
  CUDA_CHECK(cudaStreamSynchronize(p->ctrl_stream));
  CUDA_CHECK(cudaStreamSynchronize(p->probe_stream));
}

// Aggregate rate over a window from two snapshots, in unit/s.
static double probe_rate(const std::string& name, const ProbeCtx& p,
                         const std::vector<unsigned long long>& a,
                         const std::vector<unsigned long long>& b,
                         double window_s, unsigned long long* delta_iters) {
  double work = 0;
  unsigned long long delta = 0;
  for (int i = 0; i < p.blocks; ++i) {
    const unsigned long long d = b[i] - a[i];
    delta += d;
    work += (double)d * probe_work_per_iter(name, p, i, p.blocks);
  }
  if (delta_iters) *delta_iters = delta;
  return work / window_s;
}

static void sleep_ms(double ms) {
  std::this_thread::sleep_for(std::chrono::microseconds((long long)(ms * 1000)));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return 1;
  }

  cudaDeviceProp psrc, pdst;
  CUDA_CHECK(cudaGetDeviceProperties(&psrc, cfg.src_dev));
  CUDA_CHECK(cudaGetDeviceProperties(&pdst, cfg.dst_dev));
  int can_peer = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, cfg.src_dev, cfg.dst_dev));

  std::printf("=== compute/communication interference matrix ===\n");
  std::printf("src GPU %d  %-28s sm_%d%d  %d SMs  clk %d MHz\n", cfg.src_dev,
              psrc.name, psrc.major, psrc.minor, psrc.multiProcessorCount,
              psrc.clockRate / 1000);
  std::printf("dst GPU %d  %-28s sm_%d%d  %d SMs\n", cfg.dst_dev, pdst.name,
              pdst.major, pdst.minor, pdst.multiProcessorCount);
  std::printf("P2P: %s   payload/iter: %llu MiB   msg: %llu KiB   probe on: %s\n",
              can_peer ? "yes" : "NO", (unsigned long long)(cfg.total >> 20),
              (unsigned long long)(cfg.msg >> 10), cfg.probe_dev.c_str());
  std::printf("[note] lock clocks (nvidia-smi -lgc) before trusting slowdowns;\n"
              "       DVFS can masquerade as resource contention.\n\n");

  if (can_peer) {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    cudaError_t e = cudaDeviceEnablePeerAccess(cfg.dst_dev, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    else cudaGetLastError();
    CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
    e = cudaDeviceEnablePeerAccess(cfg.src_dev, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    else cudaGetLastError();
  }

  const int probe_gpu = cfg.probe_dev == "dst" ? cfg.dst_dev : cfg.src_dev;
  const cudaDeviceProp& probe_prop = cfg.probe_dev == "dst" ? pdst : psrc;
  const int probe_all_sms = probe_prop.multiProcessorCount;

  // Comm buffers on src/dst.
  uint8_t *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaMalloc(&d_src, cfg.total));
  CUDA_CHECK(cudaMemset(d_src, 3, cfg.total));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaMalloc(&d_dst, cfg.total));
  CUDA_CHECK(cudaMemset(d_dst, 0, cfg.total));

  // Comm stream lives on the source device (all movers here are push-side).
  cudaStream_t comm_stream;
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));

  size_t smem_budget = 0;
  {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      cfg.src_dev));
    smem_budget = (size_t)v;
  }

  // TMA plan: cap the message at the largest size whose >=3-stage pipeline
  // fits the smem budget. Consumer parts have far less than H100's 227 KB, so
  // the shared default (--msg 2M) would otherwise wipe out the whole column.
  uint64_t tma_msg = cfg.msg;
  TmaPlan tma_plan = plan_tma((uint32_t)tma_msg, cfg.stages, smem_budget);
  while (!tma_plan.ok && tma_msg >= 32 && tma_msg % 32 == 0) {
    tma_msg /= 2;
    tma_plan = plan_tma((uint32_t)tma_msg, cfg.stages, smem_budget);
  }
  TmaKernel tma_k = tma_plan.ok ? pick_tma_kernel(tma_plan.stages) : nullptr;
  if (tma_k) {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    CUDA_CHECK(cudaFuncSetAttribute(
        tma_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tma_plan.smem));
    if (tma_msg != cfg.msg)
      std::printf("[tma] msg capped to %llu KiB to fit shared memory\n\n",
                  (unsigned long long)(tma_msg >> 10));
  }
  const uint32_t num_msgs = (uint32_t)(cfg.total / cfg.msg);
  const uint32_t tma_num_msgs = (uint32_t)(cfg.total / tma_msg);
  const uint64_t tma_bytes = (uint64_t)tma_num_msgs * tma_msg;

  // One launch = one full payload through the given mover.
  auto comm_launch = [&](const std::string& comm, cudaStream_t s) {
    if (comm == "ce") {
      for (uint64_t off = 0; off < cfg.total; off += cfg.msg)
        CUDA_CHECK(cudaMemcpyPeerAsync(d_dst + off, cfg.dst_dev, d_src + off,
                                       cfg.src_dev, cfg.msg, s));
    } else if (comm == "tma") {
      const int blocks = std::min<uint32_t>((uint32_t)cfg.comm_sms, tma_num_msgs);
      tma_k<<<blocks, kTmaThreads, tma_plan.smem, s>>>(
          d_src, d_dst, tma_bytes, (uint32_t)tma_msg, tma_plan.stride,
          tma_num_msgs);
    } else if (comm == "reg") {
      reg_p2p_kernel<<<cfg.comm_sms, cfg.reg_threads, 0, s>>>(
          reinterpret_cast<const uint4*>(d_src),
          reinterpret_cast<uint4*>(d_dst), cfg.total / 16,
          (uint32_t)(cfg.msg / 16));
    }
  };

  auto comm_available = [&](const std::string& comm, std::string* why) {
    if (comm != "ce" && !can_peer) { *why = "needs P2P"; return false; }
    if (comm == "tma" && !tma_k) { *why = "no tma plan fits smem"; return false; }
    return true;
  };

  // tma may move slightly less than --total when the capped message does not
  // divide it; bandwidth must be computed over the bytes actually moved.
  auto comm_bytes = [&](const std::string& comm) -> uint64_t {
    return comm == "tma" ? tma_bytes : cfg.total;
  };

  // Timed run of `iters` payloads; returns average us per payload.
  auto comm_time_us = [&](const std::string& comm, int iters) -> double {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg, comm_stream));
    for (int i = 0; i < iters; ++i) comm_launch(comm, comm_stream);
    CUDA_CHECK(cudaEventRecord(end, comm_stream));
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return (double)ms * 1000.0 / iters;
  };

  ProbeCtx probe_ctx;
  probe_ctx_init(&probe_ctx, probe_gpu, cfg.hbm_buf, probe_prop.l2CacheSize);

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv,
                   "mode,probe,comm,probe_dev,probe_sms,comm_sms,msg_bytes,"
                   "comm_alone_gbps,comm_ovl_gbps,S_m,probe_alone,probe_ovl,"
                   "unit,S_c\n");
  }

  // ---- experiment A: interference matrix ----------------------------------

  if (cfg.mode == "matrix" || cfg.mode == "both") {
    // Probe-alone rates, cached by (probe, blocks).
    std::map<std::pair<std::string, int>, double> alone_cache;
    auto probe_alone_rate = [&](const std::string& probe, int blocks) {
      auto key = std::make_pair(probe, blocks);
      auto it = alone_cache.find(key);
      if (it != alone_cache.end()) return it->second;
      probe_launch(probe, &probe_ctx, blocks);
      sleep_ms(50);  // settle
      std::vector<unsigned long long> a, b;
      probe_snapshot(&probe_ctx, &a);
      const auto t0 = std::chrono::steady_clock::now();
      sleep_ms(cfg.window_ms);
      probe_snapshot(&probe_ctx, &b);
      const auto t1 = std::chrono::steady_clock::now();
      probe_stop_and_join(&probe_ctx);
      const double win_s = std::chrono::duration<double>(t1 - t0).count();
      const double rate = probe_rate(probe, probe_ctx, a, b, win_s, nullptr);
      alone_cache[key] = rate;
      return rate;
    };

    std::printf("--- interference matrix   [comm push src->dst, window ~%.0f ms] ---\n",
                cfg.window_ms);
    std::printf("%-5s %-5s %9s %8s | %10s %10s %6s | %11s %11s %8s %6s\n",
                "probe", "comm", "probe_sms", "comm_sms", "alone(GB/s)",
                "ovl(GB/s)", "S_m", "alone", "overlap", "unit", "S_c");

    for (const auto& comm : cfg.comms) {
      std::string why;
      if (!comm_available(comm, &why)) {
        std::printf("%-5s %-5s : skipped (%s)\n", "*", comm.c_str(), why.c_str());
        continue;
      }
      // comm alone: calibrate iteration count to fill the window.
      const double warm_us = comm_time_us(comm, 3);
      int iters = (int)std::max(3.0, cfg.window_ms * 1000.0 / warm_us);
      iters = std::min(iters, 20000);
      const double alone_us = comm_time_us(comm, iters);
      const double alone_gbps =
          (double)comm_bytes(comm) / (alone_us * 1e-6) / 1e9;

      // Probe CTA count: full device, minus comm CTAs when they share it.
      const bool share =
          (comm != "ce") && (probe_gpu == cfg.src_dev);
      int pblocks = cfg.probe_sms > 0
                        ? cfg.probe_sms
                        : (share ? std::max(1, probe_all_sms - cfg.comm_sms)
                                 : probe_all_sms);
      pblocks = std::min(pblocks, kMaxProbeBlocks);

      for (const auto& probe : cfg.probes) {
        const double rate_alone = probe_alone_rate(probe, pblocks);

        // Overlap: probe running, then a timed comm burst inside its window.
        probe_launch(probe, &probe_ctx, pblocks);
        sleep_ms(50);
        std::vector<unsigned long long> a, b;
        probe_snapshot(&probe_ctx, &a);
        const auto t0 = std::chrono::steady_clock::now();
        const double ovl_us = comm_time_us(comm, iters);
        const auto t1 = std::chrono::steady_clock::now();
        probe_snapshot(&probe_ctx, &b);
        probe_stop_and_join(&probe_ctx);

        const double win_s = std::chrono::duration<double>(t1 - t0).count();
        unsigned long long delta = 0;
        const double rate_ovl = probe_rate(probe, probe_ctx, a, b, win_s, &delta);
        const double ovl_gbps =
            (double)comm_bytes(comm) / (ovl_us * 1e-6) / 1e9;

        const double scale = probe_unit_scale(probe);
        const double s_m = ovl_gbps > 0 ? alone_gbps / ovl_gbps : 0;
        const double s_c = rate_ovl > 0 ? rate_alone / rate_ovl : 0;

        std::printf("%-5s %-5s %9d %8d | %10.2f %10.2f %6.2f |"
                    " %11.1f %11.1f %8s %6.2f%s\n",
                    probe.c_str(), comm.c_str(), pblocks,
                    comm == "ce" ? 0 : cfg.comm_sms, alone_gbps, ovl_gbps, s_m,
                    rate_alone / scale, rate_ovl / scale, probe_unit(probe),
                    s_c, delta == 0 ? "  [!] probe made NO progress" : "");
        std::fflush(stdout);
        if (csv)
          std::fprintf(csv, "matrix,%s,%s,%s,%d,%d,%llu,%.3f,%.3f,%.3f,%.1f,"
                       "%.1f,%s,%.3f\n",
                       probe.c_str(), comm.c_str(), cfg.probe_dev.c_str(),
                       pblocks, comm == "ce" ? 0 : cfg.comm_sms,
                       (unsigned long long)(comm == "tma" ? tma_msg : cfg.msg),
                       alone_gbps, ovl_gbps, s_m,
                       rate_alone / scale, rate_ovl / scale, probe_unit(probe),
                       s_c);
      }
    }
    std::printf(
        "\nS_m = comm slowdown under compute; S_c = compute slowdown under comm.\n"
        "S_c is measured against a probe baseline with the SAME CTA count, so\n"
        "it isolates memory/pipe contention. The opportunity cost of ceding %d\n"
        "CTAs to the tma/reg movers shows up separately, as the gap in the\n"
        "probe 'alone' column between the ce row and the tma/reg rows.\n\n",
        cfg.comm_sms);
  }

  // ---- experiment C: starvation -------------------------------------------

  if (cfg.mode == "starve" || cfg.mode == "both") {
    // The squatter must occupy every CTA slot, not just every SM: any
    // residual slot (threads, registers) lets the comm kernel co-schedule and
    // the starvation test silently measures nothing. Ask the occupancy API
    // for the true CTAs-per-SM limit of the ffma probe.
    int ffma_ctas_per_sm = 0;
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ffma_ctas_per_sm, probe_ffma_kernel, kProbeThreads, 0));
    if (ffma_ctas_per_sm < 1) ffma_ctas_per_sm = 1;
    const int squat_blocks = std::min(
        psrc.multiProcessorCount * ffma_ctas_per_sm, kMaxProbeBlocks);

    std::printf("--- starvation: full-occupancy ffma squatter on GPU %d "
                "(%d CTAs = %d/SM x %d SMs), deadline %.0f ms ---\n",
                cfg.src_dev, squat_blocks, ffma_ctas_per_sm,
                psrc.multiProcessorCount, cfg.starve_ms);
    std::printf("%-5s %14s %16s   %s\n", "comm", "alone(us)", "under-squat(us)",
                "verdict");

    // The squatter must run on the device that executes the comm kernels.
    ProbeCtx squat;
    if (probe_gpu == cfg.src_dev) {
      squat = probe_ctx;  // reuse buffers/streams
    } else {
      probe_ctx_init(&squat, cfg.src_dev, 1 << 20, psrc.l2CacheSize);
    }

    for (const auto& comm : cfg.comms) {
      std::string why;
      if (!comm_available(comm, &why)) {
        std::printf("%-5s : skipped (%s)\n", comm.c_str(), why.c_str());
        continue;
      }
      const double alone_us = comm_time_us(comm, 5);

      probe_launch("ffma", &squat, squat_blocks);
      sleep_ms(50);  // let the squatter own every SM

      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      cudaEvent_t beg, end;
      CUDA_CHECK(cudaEventCreate(&beg));
      CUDA_CHECK(cudaEventCreate(&end));
      CUDA_CHECK(cudaEventRecord(beg, comm_stream));
      comm_launch(comm, comm_stream);
      CUDA_CHECK(cudaEventRecord(end, comm_stream));

      const auto t0 = std::chrono::steady_clock::now();
      bool starved = true;
      while (std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count() < cfg.starve_ms) {
        if (cudaEventQuery(end) == cudaSuccess) { starved = false; break; }
        sleep_ms(1);
      }
      cudaGetLastError();  // clear the cudaErrorNotReady from polling
      probe_stop_and_join(&squat);
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      CUDA_CHECK(cudaStreamSynchronize(comm_stream));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
      CUDA_CHECK(cudaEventDestroy(beg));
      CUDA_CHECK(cudaEventDestroy(end));

      std::printf("%-5s %14.1f %16.1f   %s\n", comm.c_str(), alone_us,
                  (double)ms * 1000.0,
                  starved ? "STARVED until squatter stopped"
                          : "made independent progress");
      std::fflush(stdout);
      if (csv)
        std::fprintf(csv, "starve,ffma,%s,src,%d,%d,%llu,%.1f,%.1f,%s,,,,\n",
                     comm.c_str(), squat_blocks, cfg.comm_sms,
                     (unsigned long long)cfg.msg, alone_us, (double)ms * 1000.0,
                     starved ? "starved" : "progressed");
    }
    std::printf(
        "\n'under-squat' for a starved mover ~= how long the squatter held the\n"
        "SMs, not the transfer itself: the mover could not begin until the\n"
        "compute kernel exited. That is the progress-model gap between CE and\n"
        "SM-resident movers.\n");
  }

  if (csv) std::fclose(csv);
  if (!cuda_ok("final")) return 1;
  return 0;
}
