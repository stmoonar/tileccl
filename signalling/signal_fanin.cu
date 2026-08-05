// signal_fanin.cu
//
// Push-vs-pull completion-signalling microbenchmark, after the MoK
// (Mixture of Kittens) observation that push-based dispatch signalling costs
// ~103 us against ~18 us for pull-based dispatch on NVL72: the difference is
// not the payload transfer but the *completion protocol* -- push needs a
// cross-GPU "data is ready" notification (fence + remote flag + P-way fan-in
// on the receiver), pull turns data arrival itself into the completion event
// (requester-local, single entity).
//
// This is the single-process plain-CUDA translation of that experiment (the
// NVSHMEM version needs a launcher and a library the bench machines may not
// have; every mechanism it uses maps 1:1 onto peer-access primitives):
//
//   nvshmemx_signal_op            -> st.release.sys into a peer flag
//   nvshmem_putmem_signal_nbi     -> peer uint4 stores + st.release.sys flag
//   put + nvshmem_fence + signal  -> peer stores + __threadfence_system + st
//   nvshmem_getmem_nbi + quiet    -> __ldcv peer loads + local completion
//   nvshmem_uint64_wait_until     -> ld.acquire.sys polling
//
// Roles: one *target* GPU (timer lives there), the remaining GPUs are
// *sources*. Fan-in beyond the physical GPU count is emulated with logical
// sources: each source GPU runs M independent CTAs, each with its own payload
// slot and its own flag, and the target waits for all F = sum(M) of them.
// This keeps the protocol shape (F independent producers, F-way fan-in wait)
// even on a 2-GPU box.
//
// Modes (what the timed window on the target contains):
//   signal      go -> sources fire st.release.sys flag -> F-way wait.
//               Pure fan-in floor; no payload anywhere.
//   post        payload delivered and confirmed *before* t0; timed window is
//               go -> flag -> wait, identical protocol to `signal`. Any
//               excess over `signal` is residual drain/congestion from the
//               just-delivered payload -- the "data already sitting at the
//               destination" dead window measured honestly.
//   push        go -> payload stores into target slot -> st.release.sys flag
//               -> F-way wait. Full push completion protocol.
//   push_fence  same, but flag = __threadfence_system + plain store, the
//               hand-written put+fence+signal sequence. Compare with `push`
//               to price the fused release-store against an explicit fence.
//   pull        target CTAs read their slot from the source GPU via
//               ld.global.cv (L2-bypassing) loads, store locally, count local
//               completions. No source-side kernel at all; completion is
//               requester-local, matching pull-based dispatch.
//
// Timing: the target times each round with clock64() in-kernel (cross-GPU
// clocks are never compared); cycles are converted to us with a spin-kernel
// calibration against CUDA events. Per-round samples give min/p50/p95/p99 --
// fan-in cost is a tail phenomenon, means hide it.
//
// The go handshake: rounds are paced by the target releasing a `go` flag to
// every logical source (otherwise sources would free-run ahead of the
// receiver). For signal/post/push/push_fence the timed window therefore
// includes one target->source one-way hop. Deltas between those modes are
// clean (same adder); to compare their absolute values against `pull` (which
// has no handshake), subtract the one-way latency from sync_cost's ping-pong.
//
// Build:  make            (see signalling/Makefile)
// Run:    ./signal_fanin
//         ./signal_fanin --modes push,pull --fanin 1,2,4,8 --sizes 4K,32K
//         ./signal_fanin --modes signal,post --fanin 1,8,32 --csv out.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

enum Mode {
  kModeSignal = 0,
  kModePost = 1,
  kModePush = 2,
  kModePushFence = 3,
  kModePull = 4,
};

static const char* kModeNames[] = {"signal", "post", "push", "push_fence",
                                   "pull"};

// ---------------------------------------------------------------------------
// system-scope primitives (same as tests/sync_cost.cu)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ void st_plain(uint32_t* p, uint32_t v) {
  asm volatile("st.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// One CTA per logical source. Waits for the target's go, produces the
// mode-specific payload/flag sequence, epoch-stamps word 0 of its slot so the
// host can verify the *last round's* write really landed (content freshness,
// not just pattern equality).
__global__ void source_kernel(int mode, uint32_t nvec, uint64_t stride_vec,
                              uint32_t rounds, uint32_t* goA, uint32_t* goB,
                              const uint4* __restrict__ pat_base,
                              uint4* __restrict__ tgt_base, uint32_t* sig,
                              uint32_t* pre, int logical_base) {
  const int cta = blockIdx.x;
  const int lg = logical_base + cta;
  const uint4* src = pat_base + (uint64_t)cta * stride_vec;
  uint4* dst = tgt_base + (uint64_t)lg * stride_vec;

  for (uint32_t r = 1; r <= rounds; ++r) {
    if (threadIdx.x == 0) {
      while (ld_acquire_sys(&goA[cta]) < r) {
      }
    }
    __syncthreads();

    if (mode == kModeSignal) {
      if (threadIdx.x == 0) st_release_sys(&sig[lg], r);
      continue;
    }

    for (uint32_t v = threadIdx.x; v < nvec; v += blockDim.x) dst[v] = src[v];
    __syncthreads();

    if (threadIdx.x == 0) {
      st_plain(reinterpret_cast<uint32_t*>(dst), r);  // freshness stamp
      if (mode == kModePush) {
        st_release_sys(&sig[lg], r);
      } else if (mode == kModePushFence) {
        __threadfence_system();
        st_plain(&sig[lg], r);
      } else {  // kModePost: confirm delivery, wait for goB, then signal
        st_release_sys(&pre[lg], r);
        while (ld_acquire_sys(&goB[cta]) < r) {
        }
        st_release_sys(&sig[lg], r);
      }
    }
  }
}

// Single CTA on the target; thread i owns logical source i (release its go,
// poll its flag). t0..t1 brackets go-fanout + [payload] + signal + F-way wait.
__global__ void target_ctrl_kernel(int mode, int fanin, uint32_t rounds,
                                   uint32_t warmup,
                                   uint32_t* const* goA_ptrs,
                                   uint32_t* const* goB_ptrs, uint32_t* sig,
                                   uint32_t* pre,
                                   unsigned long long* times) {
  const int tid = threadIdx.x;
  unsigned long long t0 = 0;

  for (uint32_t r = 1; r <= rounds; ++r) {
    if (mode == kModePost && tid < fanin) {
      // Untimed: trigger payload delivery and wait until it is confirmed.
      st_release_sys(goA_ptrs[tid], r);
      while (ld_acquire_sys(&pre[tid]) < r) {
      }
    }
    __syncthreads();
    if (tid == 0) t0 = clock64();
    __syncthreads();

    if (tid < fanin) {
      st_release_sys(mode == kModePost ? goB_ptrs[tid] : goA_ptrs[tid], r);
      while (ld_acquire_sys(&sig[tid]) < r) {
      }
    }
    __syncthreads();
    if (tid == 0 && r > warmup) times[r - warmup - 1] = clock64() - t0;
  }
}

// Software grid barrier; safe because the kernel is launched cooperatively
// (all CTAs co-resident).
__device__ void grid_barrier(unsigned* count, volatile unsigned* gen,
                             unsigned nctas) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned g = *gen;
    __threadfence();
    if (atomicAdd(count, 1u) == nctas - 1) {
      *count = 0;
      __threadfence();
      *gen = g + 1;
    } else {
      while (*gen == g) {
      }
    }
  }
  __syncthreads();
}

// Pull protocol, entirely on the target: CTA i reads its slot from a source
// GPU with L2-bypassing loads (each round re-fetches over the link instead of
// hitting a stale local L2 line), stores it locally, and reports local
// completion into a counter. CTA 0 waits for all F completions -- that wait
// is the whole "signalling": requester-local, no cross-GPU flag anywhere.
__global__ void pull_kernel(int fanin, uint32_t nvec, uint64_t stride_vec,
                            uint32_t rounds, uint32_t warmup,
                            const uint4* const* src_ptrs,
                            uint4* __restrict__ dst_base, unsigned* bar_count,
                            unsigned* bar_gen, unsigned* done,
                            unsigned long long* times) {
  const int bid = blockIdx.x;
  const uint4* src = src_ptrs[bid];
  uint4* dst = dst_base + (uint64_t)bid * stride_vec;
  unsigned long long t0 = 0;

  for (uint32_t r = 1; r <= rounds; ++r) {
    grid_barrier(bar_count, bar_gen, gridDim.x);
    if (bid == 0 && threadIdx.x == 0) t0 = clock64();

    for (uint32_t v = threadIdx.x; v < nvec; v += blockDim.x)
      dst[v] = __ldcv(&src[v]);
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(done, 1u);
    }

    if (bid == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<volatile unsigned*>(done) < (unsigned)fanin) {
      }
      const unsigned long long t1 = clock64();
      *reinterpret_cast<volatile unsigned*>(done) = 0;
      __threadfence();
      if (r > warmup) times[r - warmup - 1] = t1 - t0;
    }
  }
}

// Spins for a fixed cycle count; event-timed on the host to calibrate
// clock64 cycles -> us on the target device.
__global__ void calib_kernel(unsigned long long spin_cycles,
                             unsigned long long* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const unsigned long long t0 = clock64();
  unsigned long long t;
  do {
    t = clock64();
  } while (t - t0 < spin_cycles);
  *out = t - t0;
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------

struct Config {
  std::vector<int> modes{kModeSignal, kModePost, kModePush, kModePushFence,
                         kModePull};
  std::vector<uint64_t> sizes{4096, 8192, 16384, 32768, 65536, 131072};
  std::vector<int> fanins{1, 2, 4, 8, 16, 32};
  std::vector<int> sources;  // empty = every peer-capable GPU
  int target = 0;
  int threads = 256;
  uint32_t rounds = 300;  // measured rounds
  uint32_t warmup = 50;
  bool check = true;
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

static std::vector<std::string> split_list(const std::string& list) {
  std::vector<std::string> out;
  size_t beg = 0;
  while (beg <= list.size()) {
    size_t end = list.find(',', beg);
    if (end == std::string::npos) end = list.size();
    if (end > beg) out.push_back(list.substr(beg, end - beg));
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
    if (a == "--modes") {
      c.modes.clear();
      for (const auto& m : split_list(next())) {
        int id = -1;
        for (int k = 0; k < 5; ++k)
          if (m == kModeNames[k]) id = k;
        if (id < 0) {
          std::fprintf(stderr, "unknown mode '%s'\n", m.c_str());
          std::exit(1);
        }
        c.modes.push_back(id);
      }
    } else if (a == "--sizes") {
      c.sizes.clear();
      for (const auto& s : split_list(next()))
        c.sizes.push_back(parse_bytes(s));
    } else if (a == "--fanin") {
      c.fanins.clear();
      for (const auto& s : split_list(next()))
        c.fanins.push_back(std::atoi(s.c_str()));
    } else if (a == "--sources") {
      c.sources.clear();
      for (const auto& s : split_list(next()))
        c.sources.push_back(std::atoi(s.c_str()));
    } else if (a == "--target") c.target = std::atoi(next().c_str());
    else if (a == "--threads") c.threads = std::atoi(next().c_str());
    else if (a == "--rounds") c.rounds = (uint32_t)std::atoi(next().c_str());
    else if (a == "--warmup") c.warmup = (uint32_t)std::atoi(next().c_str());
    else if (a == "--no-check") c.check = false;
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: %s [--modes LIST] [--sizes LIST] [--fanin LIST]\n"
          "          [--sources LIST] [--target N] [--threads N]\n"
          "          [--rounds N] [--warmup N] [--no-check] [--csv FILE]\n"
          "modes: signal post push push_fence pull\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      std::exit(1);
    }
  }
  if (c.threads < 32 || c.threads > 1024 || c.threads % 32) {
    std::fprintf(stderr, "--threads must be a multiple of 32 in [32,1024]\n");
    std::exit(1);
  }
  for (uint64_t s : c.sizes) {
    if (s == 0 || s % 16) {
      std::fprintf(stderr, "sizes must be nonzero multiples of 16 B\n");
      std::exit(1);
    }
  }
  for (int f : c.fanins) {
    if (f < 1 || f > 1024) {
      std::fprintf(stderr, "fanin must be in [1,1024]\n");
      std::exit(1);
    }
  }
  return c;
}

static uint32_t pat_word(int lg, uint64_t j) {
  return (uint32_t)((uint32_t)lg * 2654435761u ^ (uint32_t)j * 2246822519u ^
                    0x9E3779B9u);
}

struct Stats {
  double min_us, p50_us, p95_us, p99_us;
};

static Stats make_stats(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double p) {
    size_t i = (size_t)(p * (double)(v.size() - 1));
    return v[i];
  };
  return Stats{v.front(), q(0.50), q(0.95), q(0.99)};
}

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return 1;
  }

  // Pick source GPUs: every device that has bidirectional peer access with
  // the target (or the user-given subset, validated).
  std::vector<int> srcs;
  {
    std::vector<int> cand = cfg.sources;
    if (cand.empty())
      for (int d = 0; d < ndev; ++d)
        if (d != cfg.target) cand.push_back(d);
    for (int d : cand) {
      if (d == cfg.target || d < 0 || d >= ndev) {
        std::fprintf(stderr, "bad source device %d\n", d);
        return 1;
      }
      int fwd = 0, rev = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&fwd, d, cfg.target));
      CUDA_CHECK(cudaDeviceCanAccessPeer(&rev, cfg.target, d));
      if (fwd && rev) {
        srcs.push_back(d);
      } else {
        std::fprintf(stderr, "GPU %d: no P2P with target GPU %d, skipped\n",
                     d, cfg.target);
      }
    }
  }
  if (srcs.empty()) {
    std::fprintf(stderr, "no peer-capable source GPUs\n");
    return 1;
  }
  const int G = (int)srcs.size();

  // Enable peer access both ways for every (source, target) pair.
  for (int d : srcs) {
    CUDA_CHECK(cudaSetDevice(d));
    cudaError_t e = cudaDeviceEnablePeerAccess(cfg.target, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    else cudaGetLastError();
    CUDA_CHECK(cudaSetDevice(cfg.target));
    e = cudaDeviceEnablePeerAccess(d, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    else cudaGetLastError();
  }

  cudaDeviceProp tprop;
  CUDA_CHECK(cudaGetDeviceProperties(&tprop, cfg.target));

  const int F_max = *std::max_element(cfg.fanins.begin(), cfg.fanins.end());
  const int maxM = (F_max + G - 1) / G;
  const uint64_t max_size =
      *std::max_element(cfg.sizes.begin(), cfg.sizes.end());
  const uint64_t stride = (max_size + 255) & ~255ull;
  const uint64_t stride_vec = stride / 16;
  const uint32_t rounds_total = cfg.warmup + cfg.rounds;

  std::printf("=== push-vs-pull completion signalling, %d-way logical fan-in"
              " max ===\n", F_max);
  std::printf("target GPU %d (%s), %d source GPU(s):", cfg.target, tprop.name,
              G);
  for (int d : srcs) std::printf(" %d", d);
  std::printf("\nrounds %u (+%u warmup)   threads/CTA %d   slot stride %llu B\n",
              cfg.rounds, cfg.warmup, cfg.threads,
              (unsigned long long)stride);

  // ---- calibrate clock64 on the target -------------------------------------
  double cyc_per_us = 0;
  {
    CUDA_CHECK(cudaSetDevice(cfg.target));
    int khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, cfg.target));
    const unsigned long long spin = (unsigned long long)khz * 20;  // ~20 ms
    unsigned long long* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(*d_out)));
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    std::vector<double> ratio;
    for (int it = 0; it < 3; ++it) {
      CUDA_CHECK(cudaEventRecord(beg));
      calib_kernel<<<1, 32>>>(spin, d_out);
      CUDA_CHECK(cudaEventRecord(end));
      CUDA_CHECK(cudaDeviceSynchronize());
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
      unsigned long long cyc = 0;
      CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost));
      ratio.push_back((double)cyc / ((double)ms * 1000.0));
    }
    std::sort(ratio.begin(), ratio.end());
    cyc_per_us = ratio[1];
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("clock64 calibration: %.1f cycles/us (lock clocks or this "
                "drifts with DVFS)\n\n", cyc_per_us);
  }

  // ---- allocations ---------------------------------------------------------
  // Target-resident: payload slots, sig/pre flags, pointer tables, times,
  // pull-barrier scratch.
  CUDA_CHECK(cudaSetDevice(cfg.target));
  uint4* payload_t = nullptr;
  uint32_t *sig = nullptr, *pre = nullptr;
  uint32_t **goA_ptrs = nullptr, **goB_ptrs = nullptr;
  const uint4** src_ptrs = nullptr;
  unsigned* pull_scratch = nullptr;  // [bar_count, bar_gen, done]
  unsigned long long* d_times = nullptr;
  CUDA_CHECK(cudaMalloc(&payload_t, (uint64_t)F_max * stride));
  CUDA_CHECK(cudaMalloc(&sig, F_max * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&pre, F_max * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&goA_ptrs, F_max * sizeof(uint32_t*)));
  CUDA_CHECK(cudaMalloc(&goB_ptrs, F_max * sizeof(uint32_t*)));
  CUDA_CHECK(cudaMalloc(&src_ptrs, F_max * sizeof(uint4*)));
  CUDA_CHECK(cudaMalloc(&pull_scratch, 3 * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_times, cfg.rounds * sizeof(unsigned long long)));
  cudaStream_t s_tgt;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_tgt, cudaStreamNonBlocking));

  // Source-resident (per GPU): go flags polled locally, pattern buffer.
  std::vector<uint32_t*> goA(G), goB(G);
  std::vector<uint4*> pat(G);
  std::vector<cudaStream_t> s_src(G);
  for (int g = 0; g < G; ++g) {
    CUDA_CHECK(cudaSetDevice(srcs[g]));
    CUDA_CHECK(cudaMalloc(&goA[g], maxM * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&goB[g], maxM * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&pat[g], (uint64_t)maxM * stride));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_src[g], cudaStreamNonBlocking));
  }

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "w");
    if (!csv) {
      std::fprintf(stderr, "cannot open %s\n", cfg.csv.c_str());
      return 1;
    }
    std::fprintf(csv, "mode,fanin,bytes,min_us,p50_us,p95_us,p99_us,verify\n");
  }

  std::vector<unsigned long long> h_times(cfg.rounds);
  std::vector<uint32_t> h_slot(stride / 4);
  int last_filled_F = -1;

  for (int mode : cfg.modes) {
    std::printf("=== %s ===\n", kModeNames[mode]);
    std::printf("%6s %10s %11s %11s %11s %11s   %s\n", "fanin", "size",
                "min(us)", "p50(us)", "p95(us)", "p99(us)", "verify");

    for (int F : cfg.fanins) {
      // Logical sources round-robin over source GPUs, contiguous ids per GPU.
      std::vector<int> cnt(G), base(G);
      for (int g = 0; g < G; ++g) cnt[g] = F / G + (g < F % G ? 1 : 0);
      for (int g = 1; g < G; ++g) base[g] = base[g - 1] + cnt[g - 1];

      // Pattern fill depends only on the logical-id assignment, i.e. on F.
      if (F != last_filled_F) {
        for (int g = 0; g < G; ++g) {
          if (!cnt[g]) continue;
          std::vector<uint32_t> h((uint64_t)cnt[g] * stride / 4);
          for (int m = 0; m < cnt[g]; ++m)
            for (uint64_t j = 0; j < stride / 4; ++j)
              h[(uint64_t)m * (stride / 4) + j] = pat_word(base[g] + m, j);
          CUDA_CHECK(cudaSetDevice(srcs[g]));
          CUDA_CHECK(cudaMemcpy(pat[g], h.data(), (uint64_t)cnt[g] * stride,
                                cudaMemcpyHostToDevice));
        }
        // Pointer tables: go flags per logical source, pull source slots.
        std::vector<uint32_t*> hA(F), hB(F);
        std::vector<const uint4*> hS(F);
        for (int g = 0; g < G; ++g)
          for (int m = 0; m < cnt[g]; ++m) {
            hA[base[g] + m] = goA[g] + m;
            hB[base[g] + m] = goB[g] + m;
            hS[base[g] + m] = pat[g] + (uint64_t)m * stride_vec;
          }
        CUDA_CHECK(cudaSetDevice(cfg.target));
        CUDA_CHECK(cudaMemcpy(goA_ptrs, hA.data(), F * sizeof(uint32_t*),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(goB_ptrs, hB.data(), F * sizeof(uint32_t*),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(src_ptrs, hS.data(), F * sizeof(uint4*),
                              cudaMemcpyHostToDevice));
        last_filled_F = F;
      }

      for (size_t si = 0; si < cfg.sizes.size(); ++si) {
        const uint64_t S = cfg.sizes[si];
        if (mode == kModeSignal && si > 0) break;  // size-independent
        const uint32_t nvec = (uint32_t)(S / 16);

        // Reset flags and payload for this config.
        CUDA_CHECK(cudaSetDevice(cfg.target));
        CUDA_CHECK(cudaMemset(sig, 0, F_max * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(pre, 0, F_max * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(pull_scratch, 0, 3 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(payload_t, 0, (uint64_t)F_max * stride));
        for (int g = 0; g < G; ++g) {
          CUDA_CHECK(cudaSetDevice(srcs[g]));
          CUDA_CHECK(cudaMemset(goA[g], 0, maxM * sizeof(uint32_t)));
          CUDA_CHECK(cudaMemset(goB[g], 0, maxM * sizeof(uint32_t)));
          CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(cfg.target));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (mode == kModePull) {
          // Cooperative launch guarantees the F CTAs are co-resident, which
          // the software grid barrier requires.
          int per_sm = 0;
          CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &per_sm, pull_kernel, cfg.threads, 0));
          if (F > per_sm * tprop.multiProcessorCount) {
            std::printf("%6d %10llu   skipped: fanin exceeds cooperative "
                        "capacity (%d CTAs)\n", F, (unsigned long long)S,
                        per_sm * tprop.multiProcessorCount);
            continue;
          }
          int fanin = F;
          uint32_t nv = nvec, rt = rounds_total, wu = cfg.warmup;
          uint64_t sv = stride_vec;
          const uint4* const* sp = src_ptrs;
          uint4* db = payload_t;
          unsigned* bc = pull_scratch;
          unsigned* bg = pull_scratch + 1;
          unsigned* dn = pull_scratch + 2;
          unsigned long long* tm = d_times;
          void* args[] = {&fanin, &nv, &sv, &rt, &wu, &sp, &db,
                          &bc, &bg, &dn, &tm};
          CUDA_CHECK(cudaLaunchCooperativeKernel(
              (void*)pull_kernel, dim3(F), dim3(cfg.threads), args, 0, s_tgt));
          CUDA_CHECK(cudaStreamSynchronize(s_tgt));
        } else {
          for (int g = 0; g < G; ++g) {
            if (!cnt[g]) continue;
            CUDA_CHECK(cudaSetDevice(srcs[g]));
            source_kernel<<<cnt[g], cfg.threads, 0, s_src[g]>>>(
                mode, nvec, stride_vec, rounds_total, goA[g], goB[g], pat[g],
                payload_t, sig, pre, base[g]);
            CUDA_CHECK(cudaGetLastError());
          }
          CUDA_CHECK(cudaSetDevice(cfg.target));
          const int ctrl_threads = std::max(64, (F + 31) & ~31);
          target_ctrl_kernel<<<1, ctrl_threads, 0, s_tgt>>>(
              mode, F, rounds_total, cfg.warmup, goA_ptrs, goB_ptrs, sig, pre,
              d_times);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaStreamSynchronize(s_tgt));
          for (int g = 0; g < G; ++g) {
            if (!cnt[g]) continue;
            CUDA_CHECK(cudaSetDevice(srcs[g]));
            CUDA_CHECK(cudaStreamSynchronize(s_src[g]));
          }
        }

        // Collect and convert.
        CUDA_CHECK(cudaSetDevice(cfg.target));
        CUDA_CHECK(cudaMemcpy(h_times.data(), d_times,
                              cfg.rounds * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        std::vector<double> us(cfg.rounds);
        for (uint32_t i = 0; i < cfg.rounds; ++i)
          us[i] = (double)h_times[i] / cyc_per_us;
        const Stats st = make_stats(us);

        // Verify: pattern equality for the payload, and for push modes the
        // round stamp in word 0 (proves the last round's write landed).
        const char* verify = "skip";
        if (cfg.check && mode != kModeSignal) {
          bool ok = true;
          for (int lg = 0; lg < F && ok; ++lg) {
            CUDA_CHECK(cudaMemcpy(h_slot.data(),
                                  (const char*)payload_t + (uint64_t)lg * stride,
                                  S, cudaMemcpyDeviceToHost));
            uint64_t j0 = 0;
            if (mode != kModePull) {
              ok = h_slot[0] == rounds_total;
              j0 = 1;
            }
            for (uint64_t j = j0; j < S / 4 && ok; ++j)
              ok = h_slot[j] == pat_word(lg, j);
          }
          verify = ok ? "ok" : "FAIL";
        }

        if (mode == kModeSignal)
          std::printf("%6d %10s %11.2f %11.2f %11.2f %11.2f   %s\n", F, "-",
                      st.min_us, st.p50_us, st.p95_us, st.p99_us, verify);
        else
          std::printf("%6d %10llu %11.2f %11.2f %11.2f %11.2f   %s\n", F,
                      (unsigned long long)S, st.min_us, st.p50_us, st.p95_us,
                      st.p99_us, verify);
        if (csv)
          std::fprintf(csv, "%s,%d,%llu,%.3f,%.3f,%.3f,%.3f,%s\n",
                       kModeNames[mode], F,
                       (unsigned long long)(mode == kModeSignal ? 0 : S),
                       st.min_us, st.p50_us, st.p95_us, st.p99_us, verify);
      }
    }
    std::printf("\n");
  }

  if (csv) std::fclose(csv);

  CUDA_CHECK(cudaSetDevice(cfg.target));
  CUDA_CHECK(cudaStreamDestroy(s_tgt));
  CUDA_CHECK(cudaFree(payload_t));
  CUDA_CHECK(cudaFree(sig));
  CUDA_CHECK(cudaFree(pre));
  CUDA_CHECK(cudaFree(goA_ptrs));
  CUDA_CHECK(cudaFree(goB_ptrs));
  CUDA_CHECK(cudaFree(src_ptrs));
  CUDA_CHECK(cudaFree(pull_scratch));
  CUDA_CHECK(cudaFree(d_times));
  for (int g = 0; g < G; ++g) {
    CUDA_CHECK(cudaSetDevice(srcs[g]));
    CUDA_CHECK(cudaStreamDestroy(s_src[g]));
    CUDA_CHECK(cudaFree(goA[g]));
    CUDA_CHECK(cudaFree(goB[g]));
    CUDA_CHECK(cudaFree(pat[g]));
  }
  return 0;
}
