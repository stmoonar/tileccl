// intra_sm_matrix.cu
//
// Quantifies INTRA-SM interference between communication and compute -- the
// warp-specialized fusion case, where communication warps and compute warps
// live in the SAME CTA and share the warp scheduler, LSU/MIO pipes, register
// file and shared memory. The interference_matrix tool covers the inter-SM
// case (separate kernels contending for DRAM/L2/link); this covers the
// channels that only exist inside a fused kernel.
//
// One kernel, W warps per CTA (default 16), one CTA per SM:
//   warps [0, C)  : communication -- 'ldst' (uint4 push into the peer VA) or
//                   'tma' (lane-0-driven cp.async.bulk pipeline, local gmem
//                   -> smem -> peer gmem)
//   warps [C, W)  : one compute probe -- mma / ffma / smem / hbm, per-warp
//                   variants of the interference_matrix probes
// Everything runs until a stop flag; each warp publishes its iteration count,
// so compute throughput and communication bandwidth are measured over the
// same host-timed window.
//
// Two baselines separate the two costs of donating warps to communication:
//   S_warp  = rate(C=0, all warps compute) / rate(comm warps EXIT at entry)
//             -- the static cost of having fewer compute warps, no contention
//   S_intra = rate(comm warps exit) / rate(comm warps ACTIVE)
//             -- the dynamic cost: issue slots, LSU/MIO, memory pipeline
//                stolen by live communication, at identical warp counts
// The static smem/register footprint of the comm path is visible in the
// launch config (printed); tma staging also caps the message size.
//
// Copy engine has no row here by construction: CE cannot appear inside a
// CTA. Its intra-SM interference is exactly zero -- that is its selling
// point; this tool measures what the SM-resident alternatives pay instead.
//
// Build:  make            (see tests/Makefile)
// Run:    ./intra_sm_matrix
//         ./intra_sm_matrix --probes ffma,smem --comms ldst --comm-warps 1,2,4
//         ./intra_sm_matrix --warps 16 --msg 16K --csv intra.csv

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
// PTX helpers (same as the sibling benchmarks)
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

__device__ __forceinline__ void tma_store_wait_all0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// ---------------------------------------------------------------------------
// fused kernel
// ---------------------------------------------------------------------------

constexpr int kMaxWarps = 32;
constexpr int kFfmaUnroll = 512;
constexpr int kMmaUnroll = 64;
constexpr int kSmemRounds = 256;
constexpr int kSmemWordsPerWarp = 256;              // 1 KiB private region
constexpr int kProbeSmemWords = kMaxWarps * kSmemWordsPerWarp;  // 32 KiB
constexpr int kHbmVecPerIter = 512;                 // 8 KiB per iteration

enum { kProbeMma = 0, kProbeFfma, kProbeHbm, kProbeSmem };
enum { kCommNone = 0, kCommLdst, kCommTma };

__device__ __forceinline__ bool f_stop(const uint32_t* stop) {
  return *reinterpret_cast<const volatile uint32_t*>(stop) != 0;
}

__device__ __forceinline__ void f_publish(unsigned long long* c, int gw,
                                          unsigned long long it) {
  reinterpret_cast<volatile unsigned long long*>(c)[gw] = it;
}

__global__ __launch_bounds__(kMaxWarps * 32, 1) void fused_kernel(
    int probe_id, int comm_id, int comm_warps, const uint32_t* stop,
    unsigned long long* counters,
    // communication buffers (src local, dst on the peer)
    const uint4* __restrict__ csrc, uint4* __restrict__ cdst, uint64_t cbuf_vec,
    uint32_t msg_vec,
    // compute hbm-probe buffers (both local)
    const uint4* __restrict__ psrc, uint4* __restrict__ pdst, uint64_t pbuf_vec,
    // tma staging layout
    uint32_t stages, uint32_t stride_bytes) {
  extern __shared__ __align__(128) uint8_t dsm[];
  __shared__ uint32_t psm[kProbeSmemWords];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const int gw = blockIdx.x * kMaxWarps + warp;
  const bool is_comm = warp < comm_warps;

  for (int i = threadIdx.x; i < kProbeSmemWords; i += blockDim.x)
    psm[i] = i * 2654435761u;

  uint64_t* bars =
      reinterpret_cast<uint64_t*>(dsm + (size_t)comm_warps * stages * stride_bytes);
  if (is_comm && comm_id == kCommTma && lane == 0)
    for (uint32_t s = 0; s < stages; ++s) mbarrier_init(&bars[warp * stages + s], 1);
  if (comm_id == kCommTma) fence_proxy_async_shared();
  __syncthreads();

  unsigned long long it = 0;

  if (is_comm) {
    if (comm_id == kCommNone) return;  // baseline: cede the warp, no activity

    // Each comm warp owns a disjoint slice of the transfer buffers, across
    // all CTAs, and streams messages of msg_vec uint4s through it.
    const uint64_t nslices = (uint64_t)gridDim.x * comm_warps;
    const uint64_t slice = cbuf_vec / nslices;
    const uint64_t base = ((uint64_t)blockIdx.x * comm_warps + warp) * slice;

    if (comm_id == kCommLdst) {
      uint64_t pos = 0;
      while (!f_stop(stop)) {
        const uint64_t off = base + pos;
        for (uint32_t i = lane; i < msg_vec; i += 32)
          cdst[off + i] = csrc[off + i];
        pos += msg_vec;
        if (pos + msg_vec > slice) pos = 0;
        ++it;
        if (lane == 0) f_publish(counters, gw, it);
      }
    } else {  // kCommTma: lane 0 drives a per-warp bulk-copy pipeline
      if (lane != 0) return;
      uint8_t* buf = dsm + (size_t)warp * stages * stride_bytes;
      uint64_t* bar = &bars[warp * stages];
      const uint32_t msg_bytes = msg_vec * 16;
      const uint64_t slice_msgs = slice / msg_vec;
      if (slice_msgs == 0) return;
      auto issue_load = [&](uint64_t k) {
        const uint32_t s = (uint32_t)(k % stages);
        const uint64_t off = base + (k % slice_msgs) * msg_vec;
        mbarrier_arrive_expect_tx(&bar[s], msg_bytes);
        tma_load_1d(buf + (size_t)s * stride_bytes, csrc + off, msg_bytes,
                    &bar[s]);
      };
      uint32_t phase_bits = 0;
      uint64_t issued = 0;
      const uint64_t inflight = stages >= 3 ? stages - 2 : 1;
      for (; issued < inflight; ++issued) issue_load(issued);
      uint64_t done = 0;
      while (!f_stop(stop)) {
        const uint32_t s = (uint32_t)(done % stages);
        while (!mbarrier_try_wait(&bar[s], (phase_bits >> s) & 1u)) {
          if (f_stop(stop)) break;
        }
        phase_bits ^= (1u << s);
        const uint64_t off = base + (done % slice_msgs) * msg_vec;
        tma_store_1d(cdst + off, buf + (size_t)s * stride_bytes, msg_bytes);
        tma_store_commit();
        tma_store_wait_read<1>();
        issue_load(issued++);
        ++done;
        ++it;
        f_publish(counters, gw, it);
      }
      tma_store_wait_all0();
    }
    return;
  }

  // ---- compute warps ------------------------------------------------------

  if (probe_id == kProbeFfma) {
    float a = threadIdx.x * 1e-3f + 1.0f;
    const float b = 1.0000001f, c = 1e-7f;
    while (!f_stop(stop)) {
#pragma unroll
      for (int u = 0; u < kFfmaUnroll; ++u) a = fmaf(a, b, c);
      ++it;
      if (lane == 0) f_publish(counters, gw, it);
    }
    if (a == 12345.678f) counters[gw] = 0;  // defeats DCE, never taken
  } else if (probe_id == kProbeMma) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, __half> fc;
    fill_fragment(fa, __float2half(1.001f));
    fill_fragment(fb, __float2half(0.999f));
    fill_fragment(fc, __float2half(0.0f));
    while (!f_stop(stop)) {
#pragma unroll
      for (int u = 0; u < kMmaUnroll; ++u) mma_sync(fc, fa, fb, fc);
      ++it;
      if (lane == 0) f_publish(counters, gw, it);
    }
    if (__half2float(fc.x[0]) == 12345.0f) counters[gw] = 0;
  } else if (probe_id == kProbeSmem) {
    const uint32_t wbase = (uint32_t)warp * kSmemWordsPerWarp;
    uint32_t acc = threadIdx.x;
    while (!f_stop(stop)) {
#pragma unroll 4
      for (int r = 0; r < kSmemRounds; ++r)
        acc += psm[wbase + ((acc ^ (uint32_t)r) & (kSmemWordsPerWarp - 1))];
      ++it;
      if (lane == 0) f_publish(counters, gw, it);
    }
    if (acc == 0xdeadbeefu) counters[gw] = 0;
  } else {  // kProbeHbm: per-warp local streaming copy
    const int cwarps = nwarps - comm_warps;
    const uint64_t nslices = (uint64_t)gridDim.x * cwarps;
    const uint64_t slice = pbuf_vec / nslices;
    const uint64_t base =
        ((uint64_t)blockIdx.x * cwarps + (warp - comm_warps)) * slice;
    uint64_t pos = 0;
    while (!f_stop(stop)) {
      const uint64_t off = base + pos;
      for (uint32_t i = lane; i < kHbmVecPerIter; i += 32)
        pdst[off + i] = __ldcg(&psrc[off + i]);
      pos += kHbmVecPerIter;
      if (pos + kHbmVecPerIter > slice) pos = 0;
      ++it;
      if (lane == 0) f_publish(counters, gw, it);
    }
  }
}

// ---------------------------------------------------------------------------
// config / CLI
// ---------------------------------------------------------------------------

struct Config {
  std::vector<std::string> probes{"mma", "ffma", "smem", "hbm"};
  std::vector<std::string> comms{"ldst", "tma"};
  std::vector<int> comm_warps{1, 2, 4};
  int warps = 16;
  int blocks = 0;  // 0 = one per SM
  uint64_t cbuf = 64ull << 20;   // communication buffer
  uint64_t pbuf = 256ull << 20;  // hbm probe buffer (split into two halves)
  uint64_t msg = 16 << 10;
  int stages = 4;
  double window_ms = 300;
  int src_dev = 0;
  int dst_dev = 1;
  std::string csv;
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
    if (a == "--probes") c.probes = split_csv(next());
    else if (a == "--comms") c.comms = split_csv(next());
    else if (a == "--comm-warps") {
      c.comm_warps.clear();
      for (auto& t : split_csv(next())) c.comm_warps.push_back(std::atoi(t.c_str()));
    } else if (a == "--warps") c.warps = std::atoi(next().c_str());
    else if (a == "--blocks") c.blocks = std::atoi(next().c_str());
    else if (a == "--cbuf") c.cbuf = parse_bytes(next());
    else if (a == "--pbuf") c.pbuf = parse_bytes(next());
    else if (a == "--msg") c.msg = parse_bytes(next());
    else if (a == "--stages") c.stages = std::atoi(next().c_str());
    else if (a == "--window-ms") c.window_ms = std::atof(next().c_str());
    else if (a == "--src") c.src_dev = std::atoi(next().c_str());
    else if (a == "--dst") c.dst_dev = std::atoi(next().c_str());
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: %s [options]\n"
          "  --probes LIST      subset of mma,ffma,smem,hbm (default all)\n"
          "  --comms LIST       subset of ldst,tma (default both)\n"
          "  --comm-warps LIST  comm warps per CTA to sweep (default 1,2,4)\n"
          "  --warps N          warps per CTA, <= %d (default 16)\n"
          "  --blocks N         CTAs, 0 = one per SM (default 0)\n"
          "  --msg BYTES        per-message size, 16B multiple (default 16K)\n"
          "  --stages N         tma pipeline depth >= 3 (default 4)\n"
          "  --cbuf/--pbuf B    comm / hbm-probe buffer sizes (64M / 256M)\n"
          "  --window-ms MS     measurement window (default 300)\n"
          "  --src N / --dst N  GPU indices (default 0 / 1)\n"
          "  --csv PATH         append machine-readable rows to PATH\n",
          argv[0], kMaxWarps);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      std::exit(1);
    }
  }
  if (c.warps < 2 || c.warps > kMaxWarps) {
    std::fprintf(stderr, "--warps must be in [2,%d]\n", kMaxWarps);
    std::exit(1);
  }
  if (c.msg < 16 || c.msg % 16 != 0) {
    std::fprintf(stderr, "--msg must be a 16B multiple\n");
    std::exit(1);
  }
  if (c.stages < 3) {
    std::fprintf(stderr, "--stages must be >= 3\n");
    std::exit(1);
  }
  return c;
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------

static void sleep_ms(double ms) {
  std::this_thread::sleep_for(std::chrono::microseconds((long long)(ms * 1000)));
}

static int probe_id_of(const std::string& s) {
  if (s == "mma") return kProbeMma;
  if (s == "ffma") return kProbeFfma;
  if (s == "hbm") return kProbeHbm;
  if (s == "smem") return kProbeSmem;
  return -1;
}

static double probe_work_per_iter(int pid) {
  switch (pid) {
    case kProbeFfma: return 32.0 * kFfmaUnroll * 2.0;         // FLOP
    case kProbeMma: return (double)kMmaUnroll * 2 * 16 * 16 * 16;  // FLOP
    case kProbeSmem: return 32.0 * kSmemRounds * 4.0;         // bytes
    case kProbeHbm: return (double)kHbmVecPerIter * 16.0 * 2.0;    // bytes
  }
  return 0;
}

static const char* probe_unit(int pid) {
  if (pid == kProbeMma) return "TFLOP/s";
  if (pid == kProbeFfma) return "GFLOP/s";
  return "GB/s";
}

static double probe_scale(int pid) { return pid == kProbeMma ? 1e12 : 1e9; }

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return 1;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.src_dev));
  int can_peer = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, cfg.src_dev, cfg.dst_dev));
  if (!can_peer) {
    std::fprintf(stderr, "P2P unavailable; this benchmark needs it.\n");
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  cudaError_t e = cudaDeviceEnablePeerAccess(cfg.dst_dev, 0);
  if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
  else cudaGetLastError();
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  e = cudaDeviceEnablePeerAccess(cfg.src_dev, 0);
  if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
  else cudaGetLastError();

  const int blocks = cfg.blocks > 0 ? cfg.blocks : prop.multiProcessorCount;
  const int threads = cfg.warps * 32;

  // Buffers.
  uint4 *csrc = nullptr, *cdst = nullptr, *pbuf = nullptr;
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaMalloc(&cdst, cfg.cbuf));
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaMalloc(&csrc, cfg.cbuf));
  CUDA_CHECK(cudaMemset(csrc, 3, cfg.cbuf));
  CUDA_CHECK(cudaMalloc(&pbuf, cfg.pbuf));
  CUDA_CHECK(cudaMemset(pbuf, 5, cfg.pbuf));
  const uint64_t cbuf_vec = cfg.cbuf / 16;
  const uint64_t pbuf_vec = cfg.pbuf / 16 / 2;
  uint4* psrc = pbuf;
  uint4* pdst = pbuf + pbuf_vec;

  uint32_t* d_stop = nullptr;
  unsigned long long* d_cnt = nullptr;
  unsigned long long* h_snap = nullptr;
  const size_t cnt_n = (size_t)blocks * kMaxWarps;
  CUDA_CHECK(cudaMalloc(&d_stop, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cnt, cnt_n * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMallocHost(&h_snap, cnt_n * sizeof(unsigned long long)));
  cudaStream_t s_kernel, s_ctrl;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_kernel, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_ctrl, cudaStreamNonBlocking));

  // Shared memory budget for tma staging: opt-in max minus the kernel's
  // static usage. The tma message auto-halves until max(comm_warps) pipelines fit.
  cudaFuncAttributes fattr;
  CUDA_CHECK(cudaFuncGetAttributes(&fattr, fused_kernel));
  int optin = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    cfg.src_dev));
  const size_t dyn_budget = (size_t)optin > fattr.sharedSizeBytes
                                ? (size_t)optin - fattr.sharedSizeBytes - 256
                                : 0;
  int max_cw = 1;
  for (int cw : cfg.comm_warps) max_cw = std::max(max_cw, cw);
  uint64_t tma_msg = cfg.msg;
  auto tma_dyn = [&](uint64_t m) {
    const uint64_t stride = (m + 127) / 128 * 128;
    return (size_t)max_cw * cfg.stages * stride + (size_t)max_cw * cfg.stages * 8;
  };
  while (tma_dyn(tma_msg) > dyn_budget && tma_msg >= 32 && tma_msg % 32 == 0)
    tma_msg /= 2;
  const bool tma_ok = tma_dyn(tma_msg) <= dyn_budget;
  const uint32_t tma_stride = (uint32_t)((tma_msg + 127) / 128 * 128);

  std::printf("=== intra-SM interference (warp-specialized fusion) ===\n");
  std::printf("GPU %d %s sm_%d%d  %d SMs  %d CTAs x %d warps  window ~%.0f ms\n",
              cfg.src_dev, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, blocks, cfg.warps, cfg.window_ms);
  std::printf("msg: ldst=%llu KiB  tma=%llu KiB (stages=%d, dyn smem %zu B, "
              "static %zu B)\n\n",
              (unsigned long long)(cfg.msg >> 10),
              (unsigned long long)(tma_msg >> 10), cfg.stages,
              tma_dyn(tma_msg), fattr.sharedSizeBytes);
  std::printf("[note] same clock caveats as interference_matrix; S_intra is\n"
              "       clean by construction (baseline runs seconds apart).\n\n");

  // One measured run; returns aggregate compute rate (unit/s) and comm GB/s.
  auto run = [&](int pid, int cid, int cw, double* comm_gbps) -> double {
    const uint64_t msg_vec = (cid == kCommTma ? tma_msg : cfg.msg) / 16;
    const size_t dyn = cid == kCommTma ? tma_dyn(tma_msg) : 0;
    if (dyn > 0)
      CUDA_CHECK(cudaFuncSetAttribute(
          fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn));
    CUDA_CHECK(cudaMemset(d_stop, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, cnt_n * sizeof(unsigned long long)));
    CUDA_CHECK(cudaDeviceSynchronize());
    fused_kernel<<<blocks, threads, dyn, s_kernel>>>(
        pid, cid, cw, d_stop, d_cnt, csrc, cdst, cbuf_vec, (uint32_t)msg_vec,
        psrc, pdst, pbuf_vec, (uint32_t)cfg.stages, tma_stride);
    CUDA_CHECK(cudaGetLastError());
    sleep_ms(50);
    std::vector<unsigned long long> a(cnt_n), b(cnt_n);
    auto snap = [&](std::vector<unsigned long long>* v) {
      CUDA_CHECK(cudaMemcpyAsync(h_snap, d_cnt,
                                 cnt_n * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, s_ctrl));
      CUDA_CHECK(cudaStreamSynchronize(s_ctrl));
      v->assign(h_snap, h_snap + cnt_n);
    };
    snap(&a);
    const auto t0 = std::chrono::steady_clock::now();
    sleep_ms(cfg.window_ms);
    snap(&b);
    const auto t1 = std::chrono::steady_clock::now();
    static const uint32_t one = 1;
    CUDA_CHECK(cudaMemcpyAsync(d_stop, &one, sizeof(one),
                               cudaMemcpyHostToDevice, s_ctrl));
    CUDA_CHECK(cudaStreamSynchronize(s_ctrl));
    CUDA_CHECK(cudaStreamSynchronize(s_kernel));
    const double win = std::chrono::duration<double>(t1 - t0).count();

    double compute = 0, comm_bytes = 0;
    for (int blk = 0; blk < blocks; ++blk)
      for (int w = 0; w < cfg.warps; ++w) {
        const double d = (double)(b[blk * kMaxWarps + w] - a[blk * kMaxWarps + w]);
        if (w < cw)
          comm_bytes += d * (double)(msg_vec * 16);
        else
          compute += d * probe_work_per_iter(pid);
      }
    *comm_gbps = comm_bytes / win / 1e9;
    return compute / win;
  };

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv, "probe,comm,warps,comm_warps,msg_bytes,comm_gbps,"
                   "full,base,ovl,unit,S_warp,S_intra\n");
  }

  std::printf("%-5s %-5s %3s | %10s | %11s %11s %11s %8s | %7s %7s\n", "probe",
              "comm", "cw", "comm(GB/s)", "full", "base", "overlap", "unit",
              "S_warp", "S_intra");

  // Full-strength reference (C=0) per probe.
  std::map<int, double> full_rate;

  for (const auto& pname : cfg.probes) {
    const int pid = probe_id_of(pname);
    if (pid < 0) { std::printf("unknown probe %s\n", pname.c_str()); continue; }
    double dummy = 0;
    full_rate[pid] = run(pid, kCommNone, 0, &dummy);

    for (const auto& cname : cfg.comms) {
      const int cid = cname == "tma" ? kCommTma : kCommLdst;
      if (cid == kCommTma && !tma_ok) {
        std::printf("%-5s %-5s : skipped (no smem for tma staging)\n",
                    pname.c_str(), cname.c_str());
        continue;
      }
      for (int cw : cfg.comm_warps) {
        if (cw < 1 || cw >= cfg.warps) continue;
        double base_comm = 0, ovl_comm = 0;
        const double base = run(pid, kCommNone, cw, &base_comm);
        const double ovl = run(pid, cid, cw, &ovl_comm);
        const double s_warp = base > 0 ? full_rate[pid] / base : 0;
        const double s_intra = ovl > 0 ? base / ovl : 0;
        const double sc = probe_scale(pid);
        std::printf("%-5s %-5s %3d | %10.2f | %11.1f %11.1f %11.1f %8s |"
                    " %7.3f %7.3f\n",
                    pname.c_str(), cname.c_str(), cw, ovl_comm,
                    full_rate[pid] / sc, base / sc, ovl / sc, probe_unit(pid),
                    s_warp, s_intra);
        std::fflush(stdout);
        if (csv)
          std::fprintf(csv, "%s,%s,%d,%d,%llu,%.3f,%.1f,%.1f,%.1f,%s,%.4f,%.4f\n",
                       pname.c_str(), cname.c_str(), cfg.warps, cw,
                       (unsigned long long)(cid == kCommTma ? tma_msg : cfg.msg),
                       ovl_comm, full_rate[pid] / sc, base / sc, ovl / sc,
                       probe_unit(pid), s_warp, s_intra);
      }
    }
  }

  std::printf(
      "\nS_warp  = full / base : static cost of ceding cw warps (no comm\n"
      "                        activity; pure loss of parallelism)\n"
      "S_intra = base / ovl  : dynamic cost of LIVE communication at the same\n"
      "                        warp count -- issue slots, LSU/MIO, memory\n"
      "                        pipeline shared inside the SM\n"
      "comm(GB/s) is what those warps bought you in return.\n");

  if (csv) std::fclose(csv);
  return 0;
}
