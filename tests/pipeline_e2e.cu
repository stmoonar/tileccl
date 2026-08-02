// pipeline_e2e.cu
//
// Experiment D: end-to-end validation of the fusion cost model built from
// the other benchmarks. A synthetic producer -> transfer -> consumer
// pipeline runs across two GPUs; for each (method, arithmetic intensity)
// point we measure four times:
//
//   Tc_src  produce only  (same compute, tiles written to LOCAL memory)
//   Tm      transfer only (intensity 0: seed + store, no FMA chains)
//   Tc_dst  consume only  (flags preset, buffer prefilled)
//   To      the full overlapped pipeline, host wall clock across both GPUs
//
// and report the steady-state model prediction
//
//   T_pred = max(Tc_src, Tm, Tc_dst)
//
// plus its error against To. If the parameters measured by
// interference_matrix / intra_sm_matrix / sync_cost describe reality, the
// error stays small and its sign/size shows where the model leaks
// (interference, pipeline fill, sync overhead).
//
// Transfer methods -- one per "where the produced tile lives" path:
//   ce    produce into LOCAL staging (register -> gmem), then chunked
//         cudaMemcpyPeerAsync with stream/event dependencies, consumer
//         kernels chained per chunk. The kernel-boundary pipeline.
//   ldst  fused: producer computes in registers and stores STRAIGHT into
//         the peer VA, one release-flag per tile. No local staging at all.
//   tma   fused: producer computes into SHARED memory, lane 0 pushes the
//         tile with cp.async.bulk (double-buffered), release-flag follows
//         the bulk group's completion.
// The consumer is identical for all methods: persistent kernel on the dst
// GPU, acquire-polls per-tile flags (skipped for ce, which uses stream
// order), FMA-chains over the received tile, accumulates into a sink.
//
// Verification: element 0 of every tile is recomputed by the consumer from
// the producer's seed formula and compared; mismatches are counted.
//
// Sweep --intensity (FMAs per float element; FLOP/byte = intensity / 2).
// Low intensity => comm-bound (To ~ Tm), high => compute-bound (To ~ Tc).
// The crossover and the model error across it are the deliverables.
//
// Build:  make            (see tests/Makefile)
// Run:    ./pipeline_e2e
//         ./pipeline_e2e --methods ldst,tma --intensity 0,256,1024,4096
//         ./pipeline_e2e --total 128M --tile 16K --csv e2e.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

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

// ---------------------------------------------------------------------------
// workload definition
// ---------------------------------------------------------------------------

constexpr int kThreads = 256;
constexpr float kFmaB = 1.0000001f;
constexpr float kFmaC = 1e-7f;

// Deterministic per-element seed, cheap and stable across kernels.
__device__ __forceinline__ float elem_seed(uint32_t tile, uint32_t idx) {
  const uint32_t h = (tile * 40503u + idx) * 2654435761u;
  return 1.0f + (float)(h & 0xffffu) * 1e-6f;
}

// The producer's per-element compute: `intensity` dependent FMAs.
__device__ __forceinline__ float produce_elem(uint32_t tile, uint32_t idx,
                                              uint32_t intensity) {
  float a = elem_seed(tile, idx);
  for (uint32_t k = 0; k < intensity; ++k) a = fmaf(a, kFmaB, kFmaC);
  return a;
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// ldst producer: compute in registers, store straight to `dst` (peer VA in
// transfer mode, local staging in compute-only mode), optional release flag
// per tile. Also serves the ce path's produce-chunk kernel (signal=0, local
// dst, restricted tile range).
__global__ __launch_bounds__(kThreads) void produce_ldst_kernel(
    float* __restrict__ dst, uint32_t* flags, uint32_t tile_begin,
    uint32_t tile_end, uint32_t tile_floats, uint32_t intensity, int signal) {
  for (uint32_t t = tile_begin + blockIdx.x; t < tile_end; t += gridDim.x) {
    float* out = dst + (uint64_t)t * tile_floats;
    for (uint32_t i = threadIdx.x; i < tile_floats; i += blockDim.x)
      out[i] = produce_elem(t, i, intensity);
    __syncthreads();  // all stores issued before the flag
    if (signal && threadIdx.x == 0) st_release_sys(&flags[t], 1u);
  }
}

// tma producer: compute into a double-buffered smem stage, lane 0 pushes the
// tile with a bulk store; the release flag for tile k is written once the
// bulk group of tile k has fully completed (wait_group <= 1 after commit of
// k+1 guarantees k is done -- flags lag one tile, last one flushed at exit).
__global__ __launch_bounds__(kThreads) void produce_tma_kernel(
    float* __restrict__ dst, uint32_t* flags, uint32_t tile_begin,
    uint32_t tile_end, uint32_t tile_floats, uint32_t intensity, int signal) {
  extern __shared__ __align__(128) float smem[];
  const uint32_t tile_bytes = tile_floats * 4;
  uint32_t prev = 0;
  bool have_prev = false;
  uint32_t stage = 0;
  for (uint32_t t = tile_begin + blockIdx.x; t < tile_end; t += gridDim.x) {
    float* buf = smem + (uint64_t)stage * tile_floats;
    for (uint32_t i = threadIdx.x; i < tile_floats; i += blockDim.x)
      buf[i] = produce_elem(t, i, intensity);
    __syncthreads();
    if (threadIdx.x == 0) {
      fence_proxy_async_shared();
      tma_store_1d(dst + (uint64_t)t * tile_floats, buf, tile_bytes);
      tma_store_commit();
      // <=1 outstanding group: the previous tile's store has landed, its
      // buffer is free for the next produce, and its flag may be released.
      tma_store_wait_all<1>();
      if (have_prev && signal) st_release_sys(&flags[prev], 1u);
    }
    __syncthreads();  // nobody touches the freed buffer before the wait
    prev = t;
    have_prev = true;
    stage ^= 1;
  }
  if (threadIdx.x == 0) {
    tma_store_wait_all<0>();
    if (have_prev && signal) st_release_sys(&flags[prev], 1u);
  }
}

// Consumer: identical for every method. Polls per-tile flags (unless
// preset), FMA-chains over the tile, accumulates into a sink, verifies
// element 0 against the producer formula.
__global__ __launch_bounds__(kThreads) void consume_kernel(
    const float* __restrict__ buf, const uint32_t* flags, uint32_t tile_begin,
    uint32_t tile_end, uint32_t tile_floats, uint32_t intensity,
    uint32_t prod_intensity, int preset, int check, float* sink,
    unsigned int* errors) {
  float acc = 0.0f;
  for (uint32_t t = tile_begin + blockIdx.x; t < tile_end; t += gridDim.x) {
    if (!preset) {
      if (threadIdx.x == 0)
        while (ld_acquire_sys(&flags[t]) == 0u) __nanosleep(128);
      __syncthreads();
    }
    const float* in = buf + (uint64_t)t * tile_floats;
    if (check && threadIdx.x == 0) {
      const float expect = produce_elem(t, 0, prod_intensity);
      if (in[0] != expect) atomicAdd(errors, 1u);
    }
    for (uint32_t i = threadIdx.x; i < tile_floats; i += blockDim.x) {
      float a = in[i];
      for (uint32_t k = 0; k < intensity; ++k) a = fmaf(a, kFmaB, kFmaC);
      acc += a;
    }
  }
  atomicAdd(sink, acc);  // defeats DCE; negligible cost, once per thread
}

// ---------------------------------------------------------------------------
// config / CLI
// ---------------------------------------------------------------------------

struct Config {
  std::vector<std::string> methods{"ce", "ldst", "tma"};
  std::vector<uint32_t> intensity{0, 64, 256, 1024, 4096};
  uint64_t total = 256ull << 20;
  uint64_t tile = 16 << 10;
  int blocks = 0;      // producer/consumer CTAs; 0 = one per SM
  int chunks = 16;     // ce pipeline chunks
  int reps = 3;        // repetitions; min reported
  int src_dev = 0;
  int dst_dev = 1;
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
    if (a == "--methods") c.methods = split_csv(next());
    else if (a == "--intensity") {
      c.intensity.clear();
      for (auto& t : split_csv(next()))
        c.intensity.push_back((uint32_t)std::strtoul(t.c_str(), nullptr, 10));
    } else if (a == "--total") c.total = parse_bytes(next());
    else if (a == "--tile") c.tile = parse_bytes(next());
    else if (a == "--blocks") c.blocks = std::atoi(next().c_str());
    else if (a == "--chunks") c.chunks = std::atoi(next().c_str());
    else if (a == "--reps") c.reps = std::atoi(next().c_str());
    else if (a == "--src") c.src_dev = std::atoi(next().c_str());
    else if (a == "--dst") c.dst_dev = std::atoi(next().c_str());
    else if (a == "--no-check") c.check = false;
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: %s [options]\n"
          "  --methods LIST    subset of ce,ldst,tma (default all)\n"
          "  --intensity LIST  FMAs per float element (default 0,64,256,1024,4096)\n"
          "  --total BYTES     payload (default 256M)\n"
          "  --tile BYTES      tile size, 16B multiple (default 16K)\n"
          "  --blocks N        producer/consumer CTAs, 0 = one per SM\n"
          "  --chunks N        ce pipeline chunks (default 16)\n"
          "  --reps N          repetitions, min reported (default 3)\n"
          "  --src N / --dst N GPU indices (default 0 / 1)\n"
          "  --no-check        skip element-0 verification\n"
          "  --csv PATH        append machine-readable rows to PATH\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      std::exit(1);
    }
  }
  if (c.tile < 16 || c.tile % 16 != 0 || c.total % c.tile != 0) {
    std::fprintf(stderr, "--tile must be a 16B multiple dividing --total\n");
    std::exit(1);
  }
  return c;
}

// ---------------------------------------------------------------------------
// host
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

  const uint32_t tiles = (uint32_t)(cfg.total / cfg.tile);
  const uint32_t tile_floats = (uint32_t)(cfg.tile / 4);
  const int src_blocks =
      cfg.blocks > 0 ? cfg.blocks
                     : (int)std::min<uint32_t>(psrc.multiProcessorCount, tiles);
  const int dst_blocks =
      cfg.blocks > 0 ? cfg.blocks
                     : (int)std::min<uint32_t>(pdst.multiProcessorCount, tiles);

  // tma smem: two stages of one tile each.
  const size_t tma_smem = 2 * (size_t)cfg.tile;
  size_t smem_budget = 0;
  {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      cfg.src_dev));
    smem_budget = (size_t)v;
  }
  const bool tma_ok = tma_smem + 1024 <= smem_budget;
  if (tma_ok) {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    CUDA_CHECK(cudaFuncSetAttribute(produce_tma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)tma_smem));
  }

  // Buffers: local staging + remote destination + flags + sink.
  float *stage = nullptr, *remote = nullptr, *sinkd = nullptr;
  uint32_t *flags = nullptr, *flags_local = nullptr;
  unsigned int* errs = nullptr;
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaMalloc(&remote, cfg.total));
  CUDA_CHECK(cudaMalloc(&flags, tiles * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&sinkd, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&errs, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(errs, 0, sizeof(unsigned int)));
  cudaStream_t s_cons;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_cons, cudaStreamNonBlocking));
  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaMalloc(&stage, cfg.total));
  CUDA_CHECK(cudaMalloc(&flags_local, tiles * sizeof(uint32_t)));
  cudaStream_t s_prod, s_copy;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_prod, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking));

  std::printf("=== experiment D: end-to-end fused pipeline vs cost model ===\n");
  std::printf("src GPU %d %s  ->  dst GPU %d %s\n", cfg.src_dev, psrc.name,
              cfg.dst_dev, pdst.name);
  std::printf("payload %llu MiB  tile %llu KiB  tiles %u  blocks %d/%d  "
              "ce chunks %d  tma %s\n\n",
              (unsigned long long)(cfg.total >> 20),
              (unsigned long long)(cfg.tile >> 10), tiles, src_blocks,
              dst_blocks, cfg.chunks, tma_ok ? "ok" : "tile > smem, skipped");

  auto reset = [&]() {
    CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
    CUDA_CHECK(cudaMemset(flags, 0, tiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(sinkd, 0, sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  auto sync_both = [&]() {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  };

  // Wall-clock one enqueue+sync round of `body`, in ms.
  auto timed = [&](auto&& body) -> double {
    double best = 1e30;
    for (int r = 0; r < cfg.reps; ++r) {
      reset();
      const auto t0 = std::chrono::steady_clock::now();
      body();
      sync_both();
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min(best,
                      std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
  };

  auto launch_producer = [&](const std::string& m, uint32_t I, bool to_peer,
                             bool signal) {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    float* dst = to_peer ? remote : stage;
    uint32_t* fl = to_peer ? flags : flags_local;
    if (m == "tma")
      produce_tma_kernel<<<src_blocks, kThreads, tma_smem, s_prod>>>(
          dst, fl, 0, tiles, tile_floats, I, signal ? 1 : 0);
    else
      produce_ldst_kernel<<<src_blocks, kThreads, 0, s_prod>>>(
          dst, fl, 0, tiles, tile_floats, I, signal ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
  };

  auto launch_consumer = [&](uint32_t I, uint32_t prodI, bool preset) {
    CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
    consume_kernel<<<dst_blocks, kThreads, 0, s_cons>>>(
        remote, flags, 0, tiles, tile_floats, I, prodI, preset ? 1 : 0,
        cfg.check ? 1 : 0, sinkd, errs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  };

  // ce pipeline: produce chunk -> copy chunk -> consume chunk, all async.
  auto launch_ce_pipeline = [&](uint32_t I, bool produce, bool copy,
                                bool consume) {
    const uint32_t per = (tiles + cfg.chunks - 1) / cfg.chunks;
    std::vector<cudaEvent_t> evP(cfg.chunks), evC(cfg.chunks);
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    for (int k = 0; k < cfg.chunks; ++k) {
      CUDA_CHECK(cudaEventCreateWithFlags(&evP[k], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&evC[k], cudaEventDisableTiming));
    }
    for (int k = 0; k < cfg.chunks; ++k) {
      const uint32_t t0 = k * per, t1 = std::min(tiles, t0 + per);
      if (t0 >= t1) break;
      const uint64_t off = (uint64_t)t0 * tile_floats;
      const uint64_t bytes = (uint64_t)(t1 - t0) * cfg.tile;
      if (produce) {
        CUDA_CHECK(cudaSetDevice(cfg.src_dev));
        produce_ldst_kernel<<<src_blocks, kThreads, 0, s_prod>>>(
            stage, flags_local, t0, t1, tile_floats, I, 0);
        CUDA_CHECK(cudaEventRecord(evP[k], s_prod));
      }
      if (copy) {
        if (produce) CUDA_CHECK(cudaStreamWaitEvent(s_copy, evP[k], 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(remote + off, cfg.dst_dev, stage + off,
                                       cfg.src_dev, bytes, s_copy));
        CUDA_CHECK(cudaEventRecord(evC[k], s_copy));
      }
      if (consume) {
        CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
        if (copy) CUDA_CHECK(cudaStreamWaitEvent(s_cons, evC[k], 0));
        consume_kernel<<<dst_blocks, kThreads, 0, s_cons>>>(
            remote, flags, t0, t1, tile_floats, I, I, 1, cfg.check ? 1 : 0,
            sinkd, errs);
        CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      }
    }
    sync_both();
    for (int k = 0; k < cfg.chunks; ++k) {
      CUDA_CHECK(cudaEventDestroy(evP[k]));
      CUDA_CHECK(cudaEventDestroy(evC[k]));
    }
  };

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv, "method,intensity,flop_per_byte,Tc_src_ms,Tm_ms,"
                   "Tc_dst_ms,To_ms,pred_ms,err_pct,errors\n");
  }

  std::printf("%-5s %9s %8s | %9s %9s %9s %9s | %9s %7s %6s\n", "meth",
              "intensity", "FLOP/B", "Tc_src", "Tm", "Tc_dst", "To", "pred",
              "err%", "verify");

  for (const auto& m : cfg.methods) {
    if (m == "tma" && !tma_ok) {
      std::printf("%-5s : skipped (tile > smem budget)\n", m.c_str());
      continue;
    }
    for (uint32_t I : cfg.intensity) {
      double tc_src, tm, tc_dst, to;
      unsigned int nerr = 0;

      if (m == "ce") {
        tc_src = timed([&] { launch_ce_pipeline(I, true, false, false); });
        tm = timed([&] { launch_ce_pipeline(0, false, true, false); });
        // Prefill dst once so consume-only reads valid data.
        reset();
        launch_ce_pipeline(I, true, true, false);
        tc_dst = timed([&] { launch_ce_pipeline(I, false, false, true); });
        to = timed([&] { launch_ce_pipeline(I, true, true, true); });
      } else {
        tc_src = timed([&] { launch_producer(m, I, false, false); });
        tm = timed([&] { launch_producer(m, 0, true, true); });
        reset();
        launch_producer(m, I, true, true);
        sync_both();
        tc_dst = timed([&] { launch_consumer(I, I, true); });
        to = timed([&] {
          launch_consumer(I, I, false);
          launch_producer(m, I, true, true);
        });
      }

      if (cfg.check) {
        CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
        CUDA_CHECK(cudaMemcpy(&nerr, errs, sizeof(nerr), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(errs, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      }

      const double pred = std::max(tc_src, std::max(tm, tc_dst));
      const double err = (to - pred) / pred * 100.0;
      std::printf("%-5s %9u %8.1f | %9.2f %9.2f %9.2f %9.2f | %9.2f %6.1f%%"
                  " %6s\n",
                  m.c_str(), I, I / 2.0, tc_src, tm, tc_dst, to, pred, err,
                  !cfg.check ? "skip" : (nerr == 0 ? "ok" : "FAIL"));
      std::fflush(stdout);
      if (csv)
        std::fprintf(csv, "%s,%u,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%u\n",
                     m.c_str(), I, I / 2.0, tc_src, tm, tc_dst, to, pred, err,
                     nerr);
    }
  }

  std::printf(
      "\npred = max(Tc_src, Tm, Tc_dst): the perfect-overlap steady-state\n"
      "model. err%% > 0 is time the model does not explain -- interference,\n"
      "pipeline fill/drain, sync overhead. Small err at both ends and a\n"
      "bump near the crossover (Tc ~ Tm) is the expected signature.\n");

  if (csv) std::fclose(csv);
  return 0;
}
