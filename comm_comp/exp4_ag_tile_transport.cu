// exp4_ag_tile_transport.cu
//
// Experiment 4: in tile-granularity AG+GEMM fusion, how does the TRANSPORT
// CHANNEL -- copy engine vs TMA/SM-driven -- affect the compute timeline as
// the per-transfer data volume changes?
//
// Setup: 4 GPUs, full AllGather, all ranks symmetric, single process + UVA
// (enable_peer_all). Each rank owns an A shard [M/4, K] fp16; compute is a
// flag-gated fixed workload per (row-block, K-slice) unit that spin-waits on
// a per-chunk arrival flag, then streams the unit's data with real vectorized
// loads plus a fixed FMA chain (compute PERFORMANCE is out of scope; the
// dependency structure is the subject). The consumer instruction stream is
// IDENTICAL across variants (tests/pipeline_e2e.cu principle) -- only the
// flag producer and the A source pointer differ:
//
//   ce  : host issues cudaMemcpyPeerAsync pulls of G row-blocks per copy on a
//         high-priority comm stream (flux AG structure), each followed by a
//         cuStreamWriteValue32 flag publish (what flux really does). The CE
//         cannot produce the tile-blocked layout the consumer wants, so this
//         variant is a SIMULATION: the copied bytes are the real rows (volume
//         and content faithful), but the consumer loads from a pre-transformed
//         local shadow buffer. All ranks submit their waiting consumers before
//         the host starts this iteration's copies, so active time cannot be
//         shortened accidentally by copy-call submission latency.
//   tma : blockIdx.x < n_comm blocks of the SAME kernel are dedicated comm
//         blocks that pull the peers' panels with 2D tensor-map TMA loads
//         (descriptors host-encoded over PEER pointers), store them
//         tile-blocked into local HBM, and st.release the flag. The consumer
//         reads the actually-transferred data. n_comm SMs are the price.
//
// Sweep axes: G (row-blocks aggregated per copy+flag; per-copy bytes at fixed
// total volume), K (bytes per row-block, 0.25--2 MiB; per-tile bytes and
// FLOPs both scale with K so the comm/compute ratio is K-invariant -- K
// isolates absolute-copy-size effects, i.e. CE per-copy overhead
// amortization), and n_comm (TMA only).
//
// Modes per config: fused, compute-only (paired baseline, re-checked for
// drift), comm-only, bystander (comm runs, gates pass -> pure interference),
// local (same-GPU transport + real gating), memop-cost (CE flag chain alone),
// arrival (comm + an observer kernel
// recording per-chunk arrival times).
//
// Usage:
//   ./exp4_ag_tile_transport [--m 8192] [--n 8192] [--k LIST] [--g LIST]
//       [--n-comm LIST] [--variants ce,tma] [--modes LIST]
//       [--comm-streams 1|3] [--intensity N] [--slices N]
//       [--window-ms MS] [--warmup-ms MS] [--iters N] [--ndev N]
//       [--verify] [--csv PATH] [--dump-tiles PRE] [--flag-kernel]
//       [--ce-cycle-dst]
//
// See comm_comp/README.md for the full experiment description.

#include "common.cuh"
#include "exp4_transport.cuh"

#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CU_CHECK(expr)                                                       \
  do {                                                                       \
    CUresult _r = (expr);                                                    \
    if (_r != CUDA_SUCCESS) {                                                \
      std::fprintf(stderr, "[CU] %s:%d: %s -> %d\n", __FILE__, __LINE__,     \
                   #expr, (int)_r);                                          \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

enum Mode {
  M_FUSED = 0,
  M_COMPUTE,
  M_COMMONLY,
  M_BYSTANDER,
  M_LOCAL,
  M_MEMOP,
  M_ARRIVAL
};

enum Variant { V_CE = 0, V_TMA };

// K=8192 compute-only is ~0.5--0.7 ms on H800; 1024 permits the requested
// 500 ms warmup and 200 ms timing window instead of silently truncating them.
constexpr int kMaxIters = 1024;
constexpr int kCycleSlots = 8;  // --ce-cycle-dst rotation depth

struct Config {
  int m = 8192, n = 8192;  // n is bookkeeping only (no N loop in the synthetic
                           // consumer; --slices plays the many-CTAs-per-rb role)
  std::vector<int> ks = {1024, 2048, 4096, 8192};
  std::vector<int> gs = {1, 2, 4, 8, 16};
  std::vector<int> panel_hs;
  std::vector<int> ncomms = {8};
  std::vector<std::string> variants = {"ce", "tma"};
  std::vector<std::string> modes = {"fused",     "compute-only", "comm-only",
                                    "bystander", "local",        "memop-cost",
                                    "arrival"};
  int comm_streams = 1;  // 1 = flux-faithful single stream, 3 = one per peer
  int ce_reserve_sm = 0; // iso-SM control: leave this many SM slots unused
  int intensity = 1024;  // FMA ops per loaded 16 B vector
  int slices = 16;       // K-slices per row-block (compute units per rb)
  double window_ms = 200, warmup_ms = 0;
  int iters = 0;  // 0 = auto from window
  int ndev = 0;
  bool verify = false, flag_kernel = false, ce_cycle_dst = false;
  bool parallel_host = true;
  bool panel_mode = false;
  std::string csv, dump;
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --m N            gathered rows (default 8192; M %% (128*world) == 0)\n"
      "  --n N            recorded in the CSV only (default 8192)\n"
      "  --k LIST         K sweep (default 1024,2048,4096,8192; K %% 64 == 0)\n"
      "  --g LIST         row-blocks per copy+flag (default 1,2,4,8,16)\n"
      "  --panel-h LIST   supplement: 128x64 panels per flag; enables\n"
      "                   ce-aggregate,tma variants\n"
      "  --n-comm LIST    TMA comm blocks sweep (default 8)\n"
      "  --variants LIST  ce,tma; panel mode also accepts ce-aggregate,\n"
      "                   ce-panelized (diagnostic opt-in; not default)\n"
      "  --modes LIST     fused,compute-only,comm-only,bystander,local,\n"
      "                   memop-cost,arrival (default all)\n"
      "  --comm-streams N CE streams: 1 = serial ring (flux), 3 = per peer\n"
      "  --ce-reserve-sm N leave N compute blocks unused for iso-SM control\n"
      "  --intensity N    FMA ops per 16 B vector (default 1024)\n"
      "  --slices N       K-slices per row-block (default 16)\n"
      "  --window-ms MS   timing window per mode (default 200)\n"
      "  --warmup-ms MS   wall-clock warmup floor (default 0)\n"
      "  --iters N        fixed iteration count, 0 = auto\n"
      "  --ndev N         use only the first N GPUs\n"
      "  --verify         layout/transport/checksum/protocol checks\n"
      "  --csv PATH       append machine-readable rows\n"
      "  --dump-tiles PRE write PRE.tiles.rank<r>.csv + PRE.arrival.rank<r>.csv\n"
      "  --flag-kernel    force the flag-set-kernel fallback (CE publish)\n"
      "  --ce-cycle-dst   cycle the CE destination through %d slots\n"
      "  --serial-host    diagnostic: submit ranks serially (default parallel)\n",
      prog, kCycleSlots);
}

static Config parse_args(int argc, char** argv) {
  Config c;
  bool k_set = false, g_set = false, nc_set = false, var_set = false,
       mode_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    auto ints = [&](std::vector<int>& v) {
      v.clear();
      for (const auto& t : split_csv(next())) v.push_back(std::atoi(t.c_str()));
    };
    if (a == "--m") c.m = std::atoi(next().c_str());
    else if (a == "--n") c.n = std::atoi(next().c_str());
    else if (a == "--k") { ints(c.ks); k_set = true; }
    else if (a == "--g") { ints(c.gs); g_set = true; }
    else if (a == "--panel-h") {
      ints(c.panel_hs);
      c.panel_mode = true;
    }
    else if (a == "--n-comm") { ints(c.ncomms); nc_set = true; }
    else if (a == "--variants") { c.variants = split_csv(next()); var_set = true; }
    else if (a == "--modes") { c.modes = split_csv(next()); mode_set = true; }
    else if (a == "--comm-streams") c.comm_streams = std::atoi(next().c_str());
    else if (a == "--ce-reserve-sm") c.ce_reserve_sm = std::atoi(next().c_str());
    else if (a == "--intensity") c.intensity = std::atoi(next().c_str());
    else if (a == "--slices") c.slices = std::atoi(next().c_str());
    else if (a == "--window-ms") c.window_ms = std::atof(next().c_str());
    else if (a == "--warmup-ms") c.warmup_ms = std::atof(next().c_str());
    else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--ndev") c.ndev = std::atoi(next().c_str());
    else if (a == "--verify") c.verify = true;
    else if (a == "--csv") c.csv = next();
    else if (a == "--dump-tiles") c.dump = next();
    else if (a == "--flag-kernel") c.flag_kernel = true;
    else if (a == "--ce-cycle-dst") c.ce_cycle_dst = true;
    else if (a == "--serial-host") c.parallel_host = false;
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  (void)k_set; (void)g_set; (void)nc_set; (void)var_set; (void)mode_set;
  if (c.comm_streams != 1 && c.comm_streams != 3) {
    std::fprintf(stderr, "--comm-streams must be 1 or 3\n");
    std::exit(1);
  }
  if (c.ce_reserve_sm < 0) {
    std::fprintf(stderr, "--ce-reserve-sm must be non-negative\n");
    std::exit(1);
  }
  if (c.panel_mode) {
    if (c.panel_hs.empty()) {
      std::fprintf(stderr, "--panel-h requires at least one value\n");
      std::exit(1);
    }
    if (g_set) {
      std::fprintf(stderr, "--panel-h and --g are separate axes; omit --g in panel mode\n");
      std::exit(1);
    }
    if (!var_set) c.variants = {"ce-aggregate", "tma"};
    if (c.comm_streams != 1) {
      std::fprintf(stderr, "panel mode currently requires --comm-streams 1\n");
      std::exit(1);
    }
    for (int h : c.panel_hs) {
      if (h < 1) {
        std::fprintf(stderr, "--panel-h values must be positive\n");
        std::exit(1);
      }
    }
  }
  for (const auto& v : c.variants) {
    const bool ok = v == "ce" || v == "tma" ||
                    (c.panel_mode &&
                     (v == "ce-aggregate" || v == "ce-panelized"));
    if (!ok) {
      std::fprintf(stderr, "unsupported variant '%s'%s\n", v.c_str(),
                   c.panel_mode ? " in panel mode" : "");
      std::exit(1);
    }
  }
  return c;
}

// Bitwise compare on the receiving GPU.  At the 128-MiB endpoint the older
// host-vector comparison allocated and transferred multiple GiB per verify.
static bool device_equal(int device, const void* a, const void* b,
                         size_t bytes) {
  if (bytes % sizeof(uint4)) {
    std::fprintf(stderr, "device_equal requires uint4-aligned byte count\n");
    std::exit(1);
  }
  CUDA_CHECK(cudaSetDevice(device));
  unsigned* d_bad = nullptr;
  unsigned h_bad = 0;
  CUDA_CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(unsigned)));
  const size_t n = bytes / sizeof(uint4);
  const int blocks = (int)std::max<size_t>(
      1, std::min<size_t>(4096, (n + 255) / 256));
  e4::compare_u4_kernel<<<blocks, 256>>>(
      static_cast<const uint4*>(a), static_cast<const uint4*>(b), n, d_bad);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(
      cudaMemcpy(&h_bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_bad));
  return h_bad == 0;
}

// ---------------------------------------------------------------------------
// per-rank state
// ---------------------------------------------------------------------------

struct RankBuf {
  __half* A_src = nullptr;     // [m_shard, K] row-major
  __half* A_shadow = nullptr;  // [n_rb][P][128][64] tile-blocked, full M
  __half* A_staged = nullptr;  // same layout; TMA comm blocks rewrite remote rb
  __half* ce_dst = nullptr;    // [slot][peer][m_shard][K] CE landing area
  uint32_t* flags = nullptr;
  uint32_t* grid_arrivals = nullptr;
  uint32_t* grid_ready = nullptr;
  uint32_t* chunk_done = nullptr;
  uint64_t* ts_log = nullptr;
  uint64_t* arr_log = nullptr;
  e4::BlockTiming* block_timing = nullptr;
  float* sink = nullptr;
  unsigned long long* bitsum = nullptr;
  unsigned* err = nullptr;
  cudaStream_t compute = nullptr, obs = nullptr;
  std::vector<cudaStream_t> comm;              // 3, priority hi
  std::vector<cudaEvent_t> started;            // kernel-active begin
  std::vector<cudaEvent_t> done;               // kMaxIters
  std::vector<cudaEvent_t> done_o;             // kLogIters (arrival)
  cudaEvent_t e_beg = nullptr;
  e4::TmaMaps maps{};                          // passed by value at launch
};

struct ModeStats {
  Stats st[e4::kMaxWorld];
  Stats kernel[e4::kMaxWorld];
  Stats start_skew[e4::kMaxWorld];
  Stats e2e[e4::kMaxWorld];
  double host_ms_per_iter = 0;
  unsigned err_count = 0;
};

// ---------------------------------------------------------------------------
// the bench
// ---------------------------------------------------------------------------

struct Bench {
  Config cfg;
  int world = 0, n_sm = 0;
  e4::DriverApi drv;
  bool use_memop = false;
  std::vector<RankBuf> R;
  std::vector<std::thread> workers;
  std::mutex worker_mu;
  std::condition_variable worker_cv, worker_done_cv;
  std::function<void(int)> worker_task;
  int worker_generation = 0, worker_done = 0;
  bool worker_stop = false;

  // per-K geometry
  int M = 0, K = 0, P = 0, m_shard = 0, rps = 0, n_rb = 0;
  int slices_eff = 1, n_units = 0;

  // per-config
  Variant var = V_CE;
  int G = 1, H = 0, cps = 0, n_chunks = 0;
  int n_comm = 0, n_compute = 0;
  int ce_reserve_eff = 0;
  bool ce_panelized = false;
  int chunk_major = 0;

  uint32_t epoch = 0;

  size_t shard_elems() const { return (size_t)m_shard * K; }
  size_t full_elems() const { return (size_t)M * K; }
  size_t ce_slot_elems() const { return (size_t)(world - 1) * shard_elems(); }
  size_t ce_cycle_off(uint32_t e) const {
    return cfg.ce_cycle_dst ? (size_t)(e % kCycleSlots) * ce_slot_elems() : 0;
  }
  int n_jobs() const { return (world - 1) * cps; }
  const char* var_name() const {
    if (var == V_TMA) return "tma";
    if (!cfg.panel_mode) return "ce";
    return ce_panelized ? "ce-panelized" : "ce-aggregate";
  }

  void start_workers() {
    if (!cfg.parallel_host) return;
    for (int r = 0; r < world; ++r) {
      workers.emplace_back([this, r]() {
        int seen = 0;
        for (;;) {
          std::function<void(int)> task;
          {
            std::unique_lock<std::mutex> lock(worker_mu);
            worker_cv.wait(lock, [&] {
              return worker_stop || worker_generation != seen;
            });
            if (worker_stop) return;
            seen = worker_generation;
            task = worker_task;
          }
          task(r);
          {
            std::lock_guard<std::mutex> lock(worker_mu);
            if (++worker_done == world) worker_done_cv.notify_one();
          }
        }
      });
    }
  }

  void parallel_ranks(const std::function<void(int)>& task) {
    if (!cfg.parallel_host) {
      for (int r = 0; r < world; ++r) task(r);
      return;
    }
    std::unique_lock<std::mutex> lock(worker_mu);
    worker_task = task;
    worker_done = 0;
    ++worker_generation;
    worker_cv.notify_all();
    worker_done_cv.wait(lock, [&] { return worker_done == world; });
  }

  void stop_workers() {
    if (workers.empty()) return;
    {
      std::lock_guard<std::mutex> lock(worker_mu);
      worker_stop = true;
      ++worker_generation;
    }
    worker_cv.notify_all();
    for (auto& w : workers) w.join();
    workers.clear();
  }

  // ------------------------------------------------------------------ setup
  void init_devices() {
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    world = cfg.ndev > 0 ? std::min(n, cfg.ndev) : n;
    if (world < 2) {
      std::fprintf(stderr, "exp4 needs at least 2 GPUs\n");
      std::exit(1);
    }
    if (world > e4::kMaxWorld) world = e4::kMaxWorld;
    std::vector<int> devs;
    for (int r = 0; r < world; ++r) devs.push_back(r);
    enable_peer_all(devs);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    n_sm = prop.multiProcessorCount;
    drv.init();
    use_memop = drv.write32 && !cfg.flag_kernel;
    R.resize(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaFuncSetAttribute(
          e4::ag_consume_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          (int)e4::smem_bytes()));
      int cooperative = 0, active_blocks = 0;
      CUDA_CHECK(cudaDeviceGetAttribute(&cooperative,
                                        cudaDevAttrCooperativeLaunch, r));
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, e4::ag_consume_kernel, e4::kThreads,
          e4::smem_bytes()));
      if (!cooperative || active_blocks != 1) {
        std::fprintf(stderr,
                     "GPU%d persistent launch preflight failed: cooperative=%d "
                     "active_blocks_per_sm=%d (expected 1)\n",
                     r, cooperative, active_blocks);
        std::exit(1);
      }
      RankBuf& b = R[r];
      CUDA_CHECK(cudaStreamCreateWithFlags(&b.compute, cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&b.obs, cudaStreamNonBlocking));
      int lo = 0, hi = 0;
      CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
      b.comm.resize(3);
      for (auto& s : b.comm)
        CUDA_CHECK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, hi));
      b.done.resize(kMaxIters);
      for (auto& e : b.done) CUDA_CHECK(cudaEventCreate(&e));
      b.started.resize(kMaxIters);
      for (auto& e : b.started) CUDA_CHECK(cudaEventCreate(&e));
      b.done_o.resize(e4::kLogIters);
      for (auto& e : b.done_o) CUDA_CHECK(cudaEventCreate(&e));
      CUDA_CHECK(cudaEventCreate(&b.e_beg));
      CUDA_CHECK(cudaMalloc(&b.flags,
                            (size_t)e4::kMaxChunks * e4::kFlagStride * 4));
      CUDA_CHECK(cudaMalloc(&b.grid_arrivals, sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&b.grid_ready, sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&b.chunk_done,
                            (size_t)e4::kMaxChunks * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemset(b.grid_arrivals, 0, sizeof(uint32_t)));
      CUDA_CHECK(cudaMemset(b.grid_ready, 0, sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&b.ts_log, (size_t)e4::kLogIters * e4::kMaxUnits *
                                           3 * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&b.arr_log, (size_t)e4::kLogIters *
                                            (e4::kMaxChunks + 1) *
                                            sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&b.block_timing,
                            (size_t)kMaxIters * n_sm *
                                sizeof(e4::BlockTiming)));
      CUDA_CHECK(cudaMalloc(&b.sink, sizeof(float)));
      CUDA_CHECK(cudaMalloc(&b.bitsum, sizeof(unsigned long long)));
      CUDA_CHECK(cudaMalloc(&b.err, sizeof(unsigned)));
    }
    start_workers();
  }

  // %globaltimer sanity: back-to-back one-thread probes on every device.
  // Cross-GPU offsets (~launch skew) are printed as evidence; headline stats
  // only ever subtract same-GPU timestamps.
  void timer_probe() {
    std::vector<uint64_t*> d(world);
    std::vector<uint64_t> h(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMalloc(&d[r], sizeof(uint64_t)));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      e4::timer_probe_kernel<<<1, 1>>>(d[r]);
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(&h[r], d[r], 8, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d[r]));
    }
    std::printf("globaltimer cross-GPU offsets vs GPU0 (us, includes launch "
                "skew):");
    for (int r = 1; r < world; ++r)
      std::printf(" %+.1f", ((double)h[r] - (double)h[0]) / 1e3);
    std::printf("\n");
  }

  void alloc_for_k(int k) {
    K = k;
    P = K / e4::kPanelCols;
    M = cfg.m;
    m_shard = M / world;
    rps = m_shard / e4::kRbRows;
    n_rb = M / e4::kRbRows;
    slices_eff = cfg.panel_mode
                     ? P
                     : ((cfg.slices > 0 && P % cfg.slices == 0) ? cfg.slices
                                                                 : 1);
    if (!cfg.panel_mode && slices_eff != cfg.slices)
      std::fprintf(stderr, "note: K=%d -> P=%d not divisible by --slices %d, "
                           "using slices=1\n", K, P, cfg.slices);
    n_units = n_rb * slices_eff;
    if (n_units > e4::kMaxUnits) {
      std::fprintf(stderr, "n_units %d > %d: reduce --slices or M\n", n_units,
                   e4::kMaxUnits);
      std::exit(1);
    }
    const int cyc = cfg.ce_cycle_dst ? kCycleSlots : 1;
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      RankBuf& b = R[r];
      CUDA_CHECK(cudaMalloc(&b.A_src, shard_elems() * 2));
      CUDA_CHECK(cudaMalloc(&b.A_shadow, full_elems() * 2));
      CUDA_CHECK(cudaMalloc(&b.A_staged, full_elems() * 2));
      CUDA_CHECK(cudaMalloc(&b.ce_dst, ce_slot_elems() * cyc * 2));
      fill_half(b.A_src, shard_elems(), 1000 + r);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    // tile-blocked transforms: shadow AND staged get the full correct
    // content at init (TMA rewrites the remote panels of staged every fused
    // iteration with the same bytes)
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      for (int o = 0; o < world; ++o) {
        e4::transform_blocked_kernel<<<256, 256>>>(R[o].A_src, R[r].A_shadow,
                                                   o, rps, P);
        e4::transform_blocked_kernel<<<256, 256>>>(R[o].A_src, R[r].A_staged,
                                                   o, rps, P);
      }
      CUDA_CHECK(cudaGetLastError());
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
      // TMA descriptors over peer shards; last slot = self for local mode.
      for (int i = 0; i < world - 1; ++i) {
        const int owner = (r + 1 + i) % world;
        e4::encode_tmap_2d(drv, &R[r].maps.m[i], R[owner].A_src, K, m_shard);
      }
      e4::encode_tmap_2d(drv, &R[r].maps.m[e4::kMaxWorld - 1], R[r].A_src,
                         K, m_shard);
    }
  }

  void free_for_k() {
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaFree(R[r].A_src);
      cudaFree(R[r].A_shadow);
      cudaFree(R[r].A_staged);
      cudaFree(R[r].ce_dst);
    }
  }

  // config change: chunk numbering (and which slots are "local") moves with
  // G, so stale values in re-purposed slots must be cleared. Monotonic epochs
  // make plain zeros safe; local chunks get the always-passing kArmed.
  void reset_flags() {
    std::vector<uint32_t> h((size_t)e4::kMaxChunks * e4::kFlagStride, 0);
    for (int r = 0; r < world; ++r) {
      std::fill(h.begin(), h.end(), 0u);
      for (int c = r * cps; c < (r + 1) * cps; ++c)
        h[e4::flag_idx(c)] = e4::kArmed;
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(R[r].flags, h.data(), h.size() * 4,
                            cudaMemcpyHostToDevice));
    }
  }

  // -------------------------------------------------------------- iteration
  // CE comm production for rank r's iteration `epoch`: pull mode, copies on
  // this rank's high-priority stream(s) in ring order (exp2 pattern), each
  // chunk's copy followed by its flag publish on the same stream (flux
  // all_gather_op structure).
  void enqueue_ce_comm(int r, Mode mode, bool wait_consumer = true) {
    RankBuf& b = R[r];
    CUDA_CHECK(cudaSetDevice(r));
    // The cooperative consumer grid publishes grid_ready only after every
    // persistent block is resident and waiting.  This closes the cross-stream
    // race where CE traffic could otherwise start before the consumer kernel.
    if (wait_consumer)
      for (int si = 0; si < cfg.comm_streams; ++si)
        CU_CHECK(drv.wait32((CUstream)b.comm[si],
                            (CUdeviceptr)b.grid_ready, epoch,
                            CU_STREAM_WAIT_VALUE_GEQ));
    auto do_chunk = [&](int peer_i, int c_in) {
      const int owner = (r + 1 + peer_i) % world;
      cudaStream_t s = b.comm[cfg.comm_streams == 1 ? 0 : peer_i];
      const size_t elems = cfg.panel_mode
                               ? (size_t)H * e4::kPanelElems
                               : (size_t)G * e4::kRbRows * K;
      const size_t off = (size_t)c_in * elems;
      if (mode != M_MEMOP) {
        __half* dst =
            b.ce_dst + ce_cycle_off(epoch) + (size_t)peer_i * shard_elems() + off;
        const __half* src =
            mode == M_LOCAL ? b.A_src + off : R[owner].A_src + off;
        auto copy_one = [&](size_t elem_off, size_t n_elem) {
          if (mode == M_LOCAL) {
            CUDA_CHECK(cudaMemcpyAsync(dst + elem_off, src + elem_off,
                                       n_elem * 2, cudaMemcpyDeviceToDevice, s));
          } else {
            CUDA_CHECK(cudaMemcpyPeerAsync(dst + elem_off, r, src + elem_off,
                                           owner, n_elem * 2, s));
          }
        };
        if (cfg.panel_mode && ce_panelized) {
          for (int p = 0; p < H; ++p)
            copy_one((size_t)p * e4::kPanelElems, e4::kPanelElems);
        } else {
          copy_one(0, elems);
        }
      }
      uint32_t* flag = b.flags + e4::flag_idx(owner * cps + c_in);
      if (use_memop) {
        CU_CHECK(drv.write32((CUstream)s, (CUdeviceptr)flag, epoch,
                             CU_STREAM_WRITE_VALUE_DEFAULT));
      } else {
        e4::flag_set_kernel<<<1, 1, 0, s>>>(flag, epoch);
        CUDA_CHECK(cudaGetLastError());
      }
    };
    if (cfg.comm_streams == 1) {  // rank-major on the single serial stream
      for (int p = 0; p < world - 1; ++p)
        for (int c = 0; c < cps; ++c) do_chunk(p, c);
    } else {  // chunk-major across the per-peer streams
      for (int c = 0; c < cps; ++c)
        for (int p = 0; p < world - 1; ++p) do_chunk(p, c);
    }
  }

  void launch_consume(int r, Mode mode, int log_slot, int timing_slot) {
    e4::KCfg kc;
    kc.rank = r;
    kc.world = world;
    kc.P = P;
    kc.rb_per_shard = rps;
    kc.G = G;
    kc.cps = cps;
    kc.panel_mode = cfg.panel_mode ? 1 : 0;
    kc.H = H;
    kc.n_comm = (var == V_TMA) ? n_comm : 0;
    kc.n_compute = (mode == M_COMMONLY || mode == M_ARRIVAL) ? 0 : n_compute;
    kc.n_units = n_units;
    kc.slices = slices_eff;
    kc.intensity = cfg.intensity;
    kc.epoch = epoch;
    kc.epoch_cmp =
        (mode == M_FUSED || mode == M_LOCAL || mode == M_MEMOP) ? epoch : 0;
    kc.comm_enabled = (var == V_TMA && mode != M_COMPUTE) ? 1 : 0;
    kc.local_mode = (var == V_TMA && mode == M_LOCAL) ? 1 : 0;
    kc.flag_scope_sys = var == V_CE ? 1 : 0;
    kc.chunk_major = (var == V_CE && cfg.comm_streams > 1) ? 1 : 0;
    kc.log_slot = log_slot;
    kc.timing_slot = timing_slot;
    kc.timing_stride = n_sm;
    const int grid = kc.n_comm + kc.n_compute;
    if (grid == 0) return;
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemsetAsync(R[r].grid_arrivals, 0, sizeof(uint32_t),
                               R[r].compute));
    CUDA_CHECK(cudaMemsetAsync(R[r].chunk_done, 0,
                               (size_t)n_chunks * sizeof(uint32_t),
                               R[r].compute));
    const __half* src = var == V_CE ? R[r].A_shadow : R[r].A_staged;
    __half* staged = R[r].A_staged;
    uint32_t* flags = R[r].flags;
    uint64_t* ts_log = R[r].ts_log;
    float* sink = R[r].sink;
    unsigned long long* bitsum = R[r].bitsum;
    unsigned* err = R[r].err;
    e4::BlockTiming* block_timing = R[r].block_timing;
    uint32_t* grid_arrivals = R[r].grid_arrivals;
    uint32_t* grid_ready = R[r].grid_ready;
    uint32_t* chunk_done = R[r].chunk_done;
    void* args[] = {&R[r].maps, &kc,       &src,    &staged, &flags,
                    &ts_log,     &sink,     &bitsum, &err,    &block_timing,
                    &grid_arrivals, &grid_ready, &chunk_done};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)e4::ag_consume_kernel, dim3(grid), dim3(e4::kThreads), args,
        e4::smem_bytes(), R[r].compute));
  }

  // one gated iteration across all ranks (generic path; CE comm-only and
  // arrival have their own loops below)
  void enqueue_iter(Mode mode, std::vector<cudaEvent_t>& prev, int slot,
                    int log_slot, int timing_slot) {
    ++epoch;
    const bool ce_comm =
        var == V_CE && (mode == M_FUSED || mode == M_BYSTANDER ||
                        mode == M_LOCAL || mode == M_MEMOP);
    // Phase 1: submit every consumer before any CE producer.  The consumer
    // must be resident and waiting when the first tile can arrive; otherwise
    // host time spent enqueueing many small copies lets early flags become
    // ready before the kernel starts and makes active-time comparisons depend
    // on copy-call count rather than the intended dependency timeline.
    parallel_ranks([&](int r) {
      CUDA_CHECK(cudaSetDevice(r));
      if (ce_comm)
        for (int si = 0; si < cfg.comm_streams; ++si)
          for (int q = 0; q < world; ++q)
            CUDA_CHECK(cudaStreamWaitEvent(R[r].comm[si], prev[q], 0));
      for (int q = 0; q < world; ++q)
        CUDA_CHECK(cudaStreamWaitEvent(R[r].compute, prev[q], 0));
      CUDA_CHECK(cudaEventRecord(R[r].started[slot], R[r].compute));
      launch_consume(r, mode, log_slot, timing_slot);
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(R[r].done[slot], R[r].compute));
    });
    // Phase 2: only after all ranks have submitted their waiting consumers do
    // the host workers enqueue this iteration's CE copies and flag publishes.
    if (ce_comm)
      parallel_ranks([&](int r) { enqueue_ce_comm(r, mode); });
    for (int r = 0; r < world; ++r) prev[r] = R[r].done[slot];
  }

  ModeStats time_mode(Mode mode, int warm, int iters, bool log_tiles) {
    iters = std::min(iters, kMaxIters);
    ModeStats out;
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMemset(R[r].err, 0, sizeof(unsigned)));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<cudaEvent_t> prev(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(R[r].e_beg, R[r].compute));
      prev[r] = R[r].e_beg;
    }
    // Warmup reuses the started[0]/done[0] event slot every pass, and
    // cudaStreamWaitEvent binds to whichever record is most recent AT ENQUEUE
    // time.  Phase 1 of enqueue_iter runs on concurrent per-rank workers, so
    // even with each pass fully retired, worker r's wait on done[0] of rank q
    // could bind to the record worker q was concurrently making for the SAME
    // pass.  Bound that way, rank r launches its consumer only after rank q's
    // finishes -- and a CE-fused consumer finishes only after its own comm
    // stream's waits clear, which may symmetrically be bound to rank r.  Two
    // ranks then spin in their kernels while the other two never launch (the
    // ce-aggregate H=64 hang; odds grow with the pass count, warmup_ms/est).
    // Since every pass is retired below, cross-pass sequencing through done[0]
    // is unnecessary: re-anchor prev to e_beg each pass, recorded HERE, where
    // no concurrent worker can re-record it.
    for (int w = 0; w < warm; ++w) {
      for (int r = 0; r < world; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaEventRecord(R[r].e_beg, R[r].compute));
        prev[r] = R[r].e_beg;
      }
      enqueue_iter(mode, prev, 0, -1, -1);
      for (int r = 0; r < world; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaEventSynchronize(R[r].done[0]));
        for (auto& s : R[r].comm) CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaStreamSynchronize(R[r].compute));
      }
    }
    // Re-anchor both timing and cross-rank sequencing after warmup.  In
    // particular, do not leave prev pointing at done[0], which the first
    // measured iteration is about to record again.
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(R[r].e_beg, R[r].compute));
      prev[r] = R[r].e_beg;
    }
    const auto t_enq0 = std::chrono::steady_clock::now();
    const int log_base = std::max(0, iters - e4::kLogIters);
    for (int i = 0; i < iters; ++i) {
      const int ls = (log_tiles && i >= log_base) ? i - log_base : -1;
      enqueue_iter(mode, prev, i, ls, i);
    }
    out.host_ms_per_iter = wall_ms_since(t_enq0) / iters;
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventSynchronize(R[r].done[iters - 1]));
      // bystander comm may lag the kernel; drain everything before moving on
      for (auto& s : R[r].comm) CUDA_CHECK(cudaStreamSynchronize(s));
      CUDA_CHECK(cudaStreamSynchronize(R[r].compute));
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      std::vector<double> kernel_active(iters), e2e(iters);
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, R[r].e_beg, R[r].done[0]));
      e2e[0] = ms * 1000.0;
      CUDA_CHECK(
          cudaEventElapsedTime(&ms, R[r].started[0], R[r].done[0]));
      kernel_active[0] = ms * 1000.0;
      for (int i = 1; i < iters; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, R[r].done[i - 1], R[r].done[i]));
        e2e[i] = ms * 1000.0;
        CUDA_CHECK(
            cudaEventElapsedTime(&ms, R[r].started[i], R[r].done[i]));
        kernel_active[i] = ms * 1000.0;
      }
      out.kernel[r] = make_stats(kernel_active);
      out.e2e[r] = make_stats(e2e);

      const int compute_blocks = mode == M_COMMONLY ? 0 : n_compute;
      const int grid = n_comm + compute_blocks;
      std::vector<e4::BlockTiming> bt((size_t)iters * n_sm);
      CUDA_CHECK(cudaMemcpy(bt.data(), R[r].block_timing,
                            bt.size() * sizeof(e4::BlockTiming),
                            cudaMemcpyDeviceToHost));
      std::vector<double> compute_done_us, start_skew_us;
      compute_done_us.reserve(iters);
      start_skew_us.reserve(iters);
      for (int it = 0; it < iters; ++it) {
        uint64_t first_start = ~0ull, last_start = 0, last_compute_done = 0;
        std::vector<uint32_t> sm_ids;
        sm_ids.reserve(grid);
        bool timing_ok = grid > 0;
        for (int block = 0; block < grid; ++block) {
          const e4::BlockTiming& t = bt[(size_t)it * n_sm + block];
          const uint32_t expected_role = block < n_comm ? 0u : 1u;
          timing_ok &= t.start_ns != 0 && t.done_ns >= t.start_ns;
          timing_ok &= t.role == expected_role && t.sm_id < (uint32_t)n_sm;
          first_start = std::min(first_start, t.start_ns);
          last_start = std::max(last_start, t.start_ns);
          if (expected_role == 1u)
            last_compute_done = std::max(last_compute_done, t.done_ns);
          sm_ids.push_back(t.sm_id);
        }
        std::sort(sm_ids.begin(), sm_ids.end());
        timing_ok &= std::unique(sm_ids.begin(), sm_ids.end()) == sm_ids.end();
        if (!timing_ok || (compute_blocks > 0 && last_compute_done == 0)) {
          std::fprintf(stderr,
                       "GPU%d persistent timing integrity failure: mode=%d "
                       "iter=%d grid=%d compute=%d\n",
                       r, (int)mode, it, grid, compute_blocks);
          std::exit(1);
        }
        start_skew_us.push_back((double)(last_start - first_start) / 1e3);
        if (compute_blocks > 0)
          compute_done_us.push_back(
              (double)(last_compute_done - first_start) / 1e3);
      }
      // t_us_* is now the requested dependent-compute completion time.  For a
      // communication-only mode there is no compute role, so retain the whole
      // kernel event duration for backward-compatible transport reporting.
      out.st[r] = compute_blocks > 0 ? make_stats(compute_done_us)
                                     : out.kernel[r];
      out.start_skew[r] = make_stats(start_skew_us);
      unsigned e = 0;
      CUDA_CHECK(cudaMemcpy(&e, R[r].err, sizeof(e), cudaMemcpyDeviceToHost));
      out.err_count += e;
    }
    return out;
  }

  // CE copies+flags alone, wall clock over reps (exp2 run_comm_alone_us):
  // one aggregate number for the concurrent 4-rank AG.
  double run_ce_comm_only_us(double window_ms, int* reps_out) {
    auto pass = [&]() {
      ++epoch;
      parallel_ranks(
          [&](int r) { enqueue_ce_comm(r, M_FUSED, false); });
      parallel_ranks([&](int r) {
        CUDA_CHECK(cudaSetDevice(r));
        for (int si = 0; si < cfg.comm_streams; ++si)
          CUDA_CHECK(cudaStreamSynchronize(R[r].comm[si]));
      });
    };
    for (int w = 0; w < 2; ++w) pass();

    const auto probe0 = std::chrono::steady_clock::now();
    pass();
    const double probe_us = wall_ms_since(probe0) * 1000.0;
    const double target_us = std::max(1.0, window_ms * 1000.0);
    const int reps = std::max(
        10, std::min(10000, (int)(target_us / std::max(1.0, probe_us) + 0.999)));
    if (reps_out) *reps_out = reps;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) pass();
    return wall_ms_since(t0) * 1000.0 / reps;
  }

  // arrival mode: comm production + 1-block observer per rank recording each
  // chunk's flag-arrival timestamp; kLogIters gated iterations.
  ModeStats run_arrival(std::vector<std::vector<uint64_t>>& arr_out) {
    ModeStats out;
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    // prev holds one event per rank per producing stream (obs + compute/comm)
    std::vector<cudaEvent_t> prev(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(R[r].e_beg, R[r].obs));
      prev[r] = R[r].e_beg;
    }
    const int iters = e4::kLogIters;
    for (int it = 0; it < iters; ++it) {
      ++epoch;
      parallel_ranks([&](int r) {
        CUDA_CHECK(cudaSetDevice(r));
        for (int q = 0; q < world; ++q) {
          CUDA_CHECK(cudaStreamWaitEvent(R[r].obs, prev[q], 0));
          if (var == V_CE) {
            for (int si = 0; si < cfg.comm_streams; ++si)
              CUDA_CHECK(cudaStreamWaitEvent(R[r].comm[si], prev[q], 0));
          } else {
            CUDA_CHECK(cudaStreamWaitEvent(R[r].compute, prev[q], 0));
          }
        }
      });
      if (var == V_CE)
        parallel_ranks(
            [&](int r) { enqueue_ce_comm(r, M_FUSED, false); });
      else
        parallel_ranks(
            [&](int r) { launch_consume(r, M_ARRIVAL, -1, -1); });
      parallel_ranks([&](int r) {
        CUDA_CHECK(cudaSetDevice(r));
        const int obs_blocks = std::max(1, std::min(64, (n_chunks + 255) / 256));
        e4::observer_kernel<<<obs_blocks, 256, 0, R[r].obs>>>(
            R[r].flags, n_chunks, epoch,
            R[r].arr_log + (size_t)it * (e4::kMaxChunks + 1));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(R[r].done_o[it], R[r].obs));
      });
      // the observer only finishes once every flag has arrived, so its event
      // alone is a sufficient gate for the next iteration
      for (int r = 0; r < world; ++r) prev[r] = R[r].done_o[it];
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventSynchronize(R[r].done_o[iters - 1]));
      for (auto& s : R[r].comm) CUDA_CHECK(cudaStreamSynchronize(s));
      CUDA_CHECK(cudaStreamSynchronize(R[r].compute));
      CUDA_CHECK(cudaStreamSynchronize(R[r].obs));
    }
    arr_out.assign(world, {});
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      std::vector<double> per(iters);
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, R[r].e_beg, R[r].done_o[0]));
      per[0] = ms * 1000.0;
      for (int i = 1; i < iters; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, R[r].done_o[i - 1],
                                        R[r].done_o[i]));
        per[i] = ms * 1000.0;
      }
      out.st[r] = make_stats(per);
      arr_out[r].resize((size_t)iters * (e4::kMaxChunks + 1));
      CUDA_CHECK(cudaMemcpy(arr_out[r].data(), R[r].arr_log,
                            arr_out[r].size() * 8, cudaMemcpyDeviceToHost));
    }
    return out;
  }

  // rebuild A_staged's content from the shards (used after TMA local mode,
  // whose comm blocks deliberately stage own-shard rows into remote panel
  // slots -- correct traffic, wrong content for any later checksum)
  void restore_staged() {
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      for (int o = 0; o < world; ++o)
        e4::transform_blocked_kernel<<<256, 256>>>(R[o].A_src, R[r].A_staged,
                                                   o, rps, P);
      CUDA_CHECK(cudaGetLastError());
    }
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  // helper for the verify paths: exactly one fused iteration, fully synced
  void one_fused_iter() {
    std::vector<cudaEvent_t> prev(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(R[r].e_beg, R[r].compute));
      prev[r] = R[r].e_beg;
    }
    enqueue_iter(M_FUSED, prev, 0, -1, -1);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  void one_compute_iter() {
    std::vector<cudaEvent_t> prev(world);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaEventRecord(R[r].e_beg, R[r].compute));
      prev[r] = R[r].e_beg;
    }
    enqueue_iter(M_COMPUTE, prev, 0, -1, -1);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  // ------------------------------------------------------------------ verify
  // (a) the blocked layout: host-recomputed reference panels vs the transform
  // kernel's output for probe row-blocks. blocked_off is shared by the
  // transform, the comm blocks and the consumer, so it needs an independent
  // check once per K.
  bool verify_transform() {
    const int probes[3] = {0, n_rb / 2, n_rb - 1};
    for (int r = 0; r < world; ++r) {
      for (int rb : probes) {
        const int owner = rb / rps, rb_l = rb % rps;
        std::vector<__half> rows((size_t)e4::kRbRows * K);
        CUDA_CHECK(cudaMemcpy(rows.data(),
                              R[owner].A_src + (size_t)rb_l * e4::kRbRows * K,
                              rows.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<__half> got((size_t)P * e4::kPanelElems);
        CUDA_CHECK(cudaMemcpy(got.data(),
                              R[r].A_shadow + e4::blocked_off(rb, 0, P),
                              got.size() * 2, cudaMemcpyDeviceToHost));
        for (int p = 0; p < P; ++p)
          for (int row = 0; row < e4::kRbRows; ++row)
            for (int col = 0; col < e4::kPanelCols; ++col) {
              const size_t want = (size_t)row * K + p * e4::kPanelCols + col;
              const size_t have =
                  ((size_t)p * e4::kRbRows + row) * e4::kPanelCols + col;
              if (std::memcmp(&rows[want], &got[have], 2)) return false;
            }
      }
    }
    return true;
  }

  // (b) TMA transport: scrub the remote panels of A_staged, rebuild them with
  // one fused-TMA iteration over NVLink, compare bitwise against the shadow,
  // and retain the checksum consumed by the dependent compute blocks.  The
  // caller compares that checksum with the compute-only reference, proving
  // not merely that HBM is correct after kernel exit, but that the release
  // flag did not become visible before the TMA data did.
  bool verify_tma_staged(unsigned long long* fused_bitsum) {
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      __half* s = R[r].A_staged;
      const size_t lo = e4::blocked_off(r * rps, 0, P);
      const size_t hi = e4::blocked_off((r + 1) * rps, 0, P);
      if (lo) CUDA_CHECK(cudaMemset(s, 0x55, lo * 2));
      if (full_elems() - hi)
        CUDA_CHECK(cudaMemset(s + hi, 0x55, (full_elems() - hi) * 2));
      CUDA_CHECK(cudaMemset(R[r].bitsum, 0, sizeof(unsigned long long)));
      // the memsets run on the legacy default stream, which does NOT order
      // against the non-blocking pipeline streams (exp2's documented footgun)
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    one_fused_iter();
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMemcpy(&fused_bitsum[r], R[r].bitsum,
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
      const size_t lo = e4::blocked_off(r * rps, 0, P);
      const size_t hi = e4::blocked_off((r + 1) * rps, 0, P);
      if (lo && !device_equal(r, R[r].A_staged, R[r].A_shadow, lo * 2))
        return false;
      if (full_elems() - hi &&
          !device_equal(r, R[r].A_staged + hi, R[r].A_shadow + hi,
                        (full_elems() - hi) * 2))
        return false;
    }
    return true;
  }

  // (c) CE transport: scrub ce_dst, one fused-CE iteration, then each peer
  // slot must be a bitwise copy of that peer's shard rows.
  bool verify_ce_blob() {
    const int cyc = cfg.ce_cycle_dst ? kCycleSlots : 1;
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMemset(R[r].ce_dst, 0x55, ce_slot_elems() * cyc * 2));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    one_fused_iter();
    for (int r = 0; r < world; ++r) {
      const size_t base = ce_cycle_off(epoch);  // slot that iteration used
      for (int p = 0; p < world - 1; ++p) {
        const int owner = (r + 1 + p) % world;
        if (!device_equal(r,
                          R[r].ce_dst + base + (size_t)p * shard_elems(),
                          R[owner].A_src, shard_elems() * 2))
          return false;
      }
    }
    return true;
  }

  // (d) cross-variant checksum: shadow (CE source) and staged (TMA source)
  // hold the same content, and bitsum is a commutative wrapping add, so one
  // compute-only pass per variant must agree bitwise despite the grid split.
  bool verify_bitsum(unsigned long long* bs_ce, unsigned long long* bs_tma) {
    const Variant saved = var;
    for (int v = 0; v < 2; ++v) {
      var = (v == 0) ? V_CE : V_TMA;
      for (int r = 0; r < world; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemset(R[r].bitsum, 0, 8));
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      one_compute_iter();
      for (int r = 0; r < world; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(v == 0 ? &bs_ce[r] : &bs_tma[r], R[r].bitsum, 8,
                              cudaMemcpyDeviceToHost));
      }
    }
    var = saved;
    for (int r = 0; r < world; ++r)
      if (bs_ce[r] != bs_tma[r]) return false;
    return true;
  }

  // ------------------------------------------------------------- statistics
  int unit_rb(int i, int rank) const {
    const int units_per_rb = cfg.panel_mode ? P : slices_eff;
    return e4::rb_order_at(i / units_per_rb, rank, world, rps, G,
                           chunk_major);
  }

  int unit_chunk(int i, int rank) const {
    const int rb = unit_rb(i, rank);
    return cfg.panel_mode ? (rb * P + (i % P)) / H : rb / G;
  }

  // remote-unit wait times from the per-tile timestamp ring (local units
  // pass their pre-armed flag in one load and would only dilute the stats)
  Stats wait_stats(int r, int n_logged) {
    std::vector<uint64_t> h((size_t)n_logged * e4::kMaxUnits * 3);
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemcpy(h.data(), R[r].ts_log, h.size() * 8,
                          cudaMemcpyDeviceToHost));
    std::vector<double> waits;
    waits.reserve((size_t)n_logged * n_units);
    for (int s = 0; s < n_logged; ++s)
      for (int i = 0; i < n_units; ++i) {
        const int rb = unit_rb(i, r);
        if (rb / rps == r) continue;
        const uint64_t* e = &h[((size_t)s * e4::kMaxUnits + i) * 3];
        waits.push_back((double)(e[1] - e[0]) / 1e3);
      }
    return make_stats(waits);
  }

  void dump_tiles(FILE* f, int r, int n_logged) {
    std::vector<uint64_t> h((size_t)n_logged * e4::kMaxUnits * 3);
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemcpy(h.data(), R[r].ts_log, h.size() * 8,
                          cudaMemcpyDeviceToHost));
    for (int s = 0; s < n_logged; ++s)
      for (int i = 0; i < n_units; ++i) {
        const int rb = unit_rb(i, r);
        const int chunk = unit_chunk(i, r);
        const uint64_t* e = &h[((size_t)s * e4::kMaxUnits + i) * 3];
        std::fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",
                     var_name(), K, cfg.panel_mode ? H : G, n_comm, s, i, rb,
                     chunk, rb / rps,
                     (unsigned long long)e[0], (unsigned long long)e[1],
                     (unsigned long long)e[2]);
      }
  }
};

// ---------------------------------------------------------------------------
// main sweep
// ---------------------------------------------------------------------------

static bool has_mode(const Config& c, const char* name) {
  for (const auto& m : c.modes)
    if (m == name) return true;
  return false;
}

int main(int argc, char** argv) {
  // The runner pipes output through tee.  Keep each completed line visible so
  // a timeout identifies the exact configuration instead of the last flush.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  Bench b;
  b.cfg = parse_args(argc, argv);
  const Config& cfg = b.cfg;
  b.init_devices();

  std::printf("=== exp4: tile-granularity AG fusion, CE vs TMA transport%s ===\n",
              cfg.panel_mode ? " [panel supplement]" : "");
  std::printf("world %d, %d SMs, flag mech %s, comm streams %d, intensity %d, "
              "slices %d%s\n",
              b.world, b.n_sm, b.use_memop ? "memop" : "kernel",
              cfg.comm_streams, cfg.intensity, cfg.slices,
              cfg.ce_cycle_dst ? ", ce-cycle-dst" : "");
  if (cfg.panel_mode)
    std::printf("panel mode: one unit = 128x64 fp16 = 16 KiB; --panel-h "
                "controls panels per ready flag\n");
  std::printf("host rank submission: %s\n",
              cfg.parallel_host ? "parallel workers" : "serial diagnostic");
  std::printf("timing: cooperative persistent grid, 1 block/SM; t_us_* = "
              "earliest block start -> last compute block done (%%globaltimer)\n");
  if (!b.drv.write32 && !cfg.flag_kernel)
    std::printf("!! cuStreamWriteValue32 unavailable -> flag-kernel fallback\n");
  for (int r = 0; r < b.world; ++r) print_device_line(r);
  b.timer_probe();
  std::printf("\n");

  if (cfg.m % (e4::kRbRows * b.world)) {
    std::fprintf(stderr, "M=%d must be a multiple of %d\n", cfg.m,
                 e4::kRbRows * b.world);
    b.stop_workers();
    return 1;
  }

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv,
                   "variant,mode,flag_mech,ce_dst,rank,world,m,n,k,g_rb,"
                   "n_chunks,chunk_bytes,n_comm,n_comm_eff,comm_streams,"
                   "slices,intensity,grid,block,iters,epoch_last,t_us_mean,"
                   "t_us_p50,t_us_p95,t_us_min,t_us_max,compute_only_us,"
                   "drift_pct,slowdown,interference_sd,stall_sd,comm_gbps,"
                   "wait_p50_us,wait_p95_us,wait_max_us,bitsum_hex,err_count,"
                   "verify,axis,panel_h,e2e_us_mean,host_enqueue_ms,"
                   "ce_reserve_sm\n");
  }
  std::vector<FILE*> dt(b.world, nullptr), da(b.world, nullptr);
  if (!cfg.dump.empty()) {
    for (int r = 0; r < b.world; ++r) {
      char path[512];
      std::snprintf(path, sizeof path, "%s.tiles.rank%d.csv",
                    cfg.dump.c_str(), r);
      dt[r] = std::fopen(path, "a");
      if (dt[r])
        std::fprintf(dt[r], "variant,k,granularity,n_comm,iter,unit,rb,chunk,peer,"
                            "t_wait_begin_ns,t_flag_seen_ns,t_done_ns\n");
      std::snprintf(path, sizeof path, "%s.arrival.rank%d.csv",
                    cfg.dump.c_str(), r);
      da[r] = std::fopen(path, "a");
      if (da[r])
        std::fprintf(da[r], "variant,k,granularity,n_comm,iter,chunk,peer,t_arm_ns,"
                            "t_arrival_ns\n");
    }
  }

  const bool w_fused = has_mode(cfg, "fused");
  const bool w_byst = has_mode(cfg, "bystander");
  const bool w_comm = has_mode(cfg, "comm-only");
  const bool w_local = has_mode(cfg, "local");
  const bool w_memop = has_mode(cfg, "memop-cost");
  const bool w_arr = has_mode(cfg, "arrival");

  for (int k : cfg.ks) {
    if (k % e4::kPanelCols) {
      std::printf("skip K=%d: not a multiple of %d\n", k, e4::kPanelCols);
      continue;
    }
    b.alloc_for_k(k);
    bool a_ok = true;
    if (cfg.verify) {
      a_ok = b.verify_transform();
      std::printf("K=%d layout transform verify: %s\n", k,
                  a_ok ? "ok" : "FAIL");
    }
    const double remote_bytes =
        (double)(b.world - 1) * b.m_shard * b.K * 2;  // per rank per iter

    const std::vector<int>& granularities =
        cfg.panel_mode ? cfg.panel_hs : cfg.gs;
    for (int gran : granularities) {
      const int panels_per_shard = b.rps * b.P;
      if ((!cfg.panel_mode && (gran < 1 || b.rps % gran)) ||
          (cfg.panel_mode &&
           (gran < 1 || panels_per_shard % gran))) {
        if (cfg.panel_mode)
          std::fprintf(stderr,
                       "skip H=%d: %d panels/shard not divisible\n", gran,
                       panels_per_shard);
        else
          std::fprintf(stderr,
                       "skip G=%d: %d row-blocks/shard not divisible\n",
                       gran, b.rps);
        continue;
      }
      for (const auto& vn : cfg.variants) {
        const Variant v = (vn == "tma") ? V_TMA : V_CE;
        const std::vector<int> ncl =
            (v == V_TMA) ? cfg.ncomms : std::vector<int>{0};
        for (int nc : ncl) {
          // -------------------------------------------------- one config
          if (v == V_TMA && nc < 1) {
            std::fprintf(stderr, "skip tma n_comm=%d: fused would have no "
                                 "flag producer\n", nc);
            continue;
          }
          b.var = v;
          b.ce_panelized = cfg.panel_mode && vn == "ce-panelized";
          b.G = cfg.panel_mode ? 1 : gran;
          b.H = cfg.panel_mode ? gran : 0;
          b.cps = cfg.panel_mode ? panels_per_shard / b.H : b.rps / b.G;
          b.n_chunks = b.world * b.cps;
          if (b.n_chunks > e4::kMaxChunks) {
            std::fprintf(stderr, "skip %c=%d: %d chunks > %d flag slots\n",
                         cfg.panel_mode ? 'H' : 'G', gran, b.n_chunks,
                         e4::kMaxChunks);
            continue;
          }
          b.n_comm =
              (v == V_TMA)
                  ? std::min(
                        std::min(nc, b.n_sm - 1),
                        cfg.panel_mode ? b.n_jobs() * b.H : b.n_jobs())
                  : 0;
          // A flag micro-kernel cannot run if waiting consumers occupy every
          // SM.  The normal H800 path uses stream memops; keep one SM free for
          // the explicitly requested/unavailable-memop fallback.
          b.ce_reserve_eff =
              v == V_CE
                  ? std::min(std::max(cfg.ce_reserve_sm, b.use_memop ? 0 : 1),
                             b.n_sm - 1)
                  : 0;
          b.n_compute =
              std::min(b.n_sm - b.n_comm - b.ce_reserve_eff, b.n_units);
          b.chunk_major =
              (!cfg.panel_mode && v == V_CE && cfg.comm_streams > 1) ? 1 : 0;
          b.reset_flags();
          const size_t chunk_bytes =
              cfg.panel_mode ? (size_t)b.H * e4::kPanelBytes
                             : (size_t)b.G * e4::kRbRows * b.K * 2;
          const int n_comm_eff = (v == V_TMA) ? b.n_comm : 0;
          const int grid = b.n_comm + b.n_compute;

          std::printf("--- %s K=%d %c=%d chunk=%.3fMiB jobs/rank=%d",
                      b.var_name(), b.K, cfg.panel_mode ? 'H' : 'G', gran,
                      chunk_bytes / (double)(1 << 20), b.n_jobs());
          if (v == V_TMA)
            std::printf(" panels/rank=%d n_comm=%d(eff %d)",
                        cfg.panel_mode ? b.n_jobs() * b.H
                                       : b.n_jobs() * b.G * b.P,
                        b.n_comm, n_comm_eff);
          else
            std::printf(" copy-calls/rank=%d streams=%d ce-reserve-sm=%d",
                        b.n_jobs() *
                            ((cfg.panel_mode && b.ce_panelized) ? b.H : 1),
                        cfg.comm_streams, b.ce_reserve_eff);
          std::printf(" grid=%d ---\n", grid);

          // verify
          bool ver_ok = a_ok;
          unsigned long long bs_ce[e4::kMaxWorld] = {0},
                             bs_tma[e4::kMaxWorld] = {0},
                             bs_tma_fused[e4::kMaxWorld] = {0};
          if (cfg.verify) {
            if (v == V_TMA && !b.verify_tma_staged(bs_tma_fused))
              ver_ok = false;
            if (v == V_CE && !b.verify_ce_blob()) ver_ok = false;
            if (!b.verify_bitsum(bs_ce, bs_tma)) ver_ok = false;
            if (v == V_TMA)
              for (int r = 0; r < b.world; ++r)
                if (bs_tma_fused[r] != bs_tma[r]) ver_ok = false;
            if (!ver_ok) std::printf("  !! verify FAIL\n");
          }
          const char* ver = !cfg.verify ? "-" : (ver_ok ? "ok" : "FAIL");

          auto iters_of = [&](double est) {
            return cfg.iters > 0
                       ? std::min(cfg.iters, kMaxIters)
                       : (int)std::max(
                             5.0, std::min((double)kMaxIters,
                                           cfg.window_ms * 1000.0 /
                                               std::max(1.0, est)));
          };
          auto max_mean = [&](const ModeStats& s) {
            double m = 0;
            for (int r = 0; r < b.world; ++r)
              m = std::max(m, std::max(s.st[r].mean, s.e2e[r].mean));
            return m;
          };

          // baseline (paired, rechecked after fused)
          ModeStats probe = b.time_mode(M_COMPUTE, 0, 2, false);
          const double est_b = max_mean(probe);
          const int warm_b =
              iters_for_ms(cfg.warmup_ms, est_b, 2, kMaxIters);
          const int it_b = iters_of(est_b);
          ModeStats base = b.time_mode(M_COMPUTE, warm_b, it_b, false);

          ModeStats fused{};
          Stats ws[e4::kMaxWorld];
          int it_f = 0, n_logged = 0;
          if (w_fused) {
            ModeStats fprobe = b.time_mode(M_FUSED, 0, 2, false);
            const double est_f = max_mean(fprobe);
            it_f = iters_of(est_f);
            fused = b.time_mode(
                M_FUSED, iters_for_ms(cfg.warmup_ms, est_f, 2, kMaxIters),
                it_f, true);
            n_logged = std::min(it_f, e4::kLogIters);
            for (int r = 0; r < b.world; ++r) ws[r] = b.wait_stats(r, n_logged);
            if (!cfg.dump.empty())
              for (int r = 0; r < b.world; ++r)
                if (dt[r]) b.dump_tiles(dt[r], r, n_logged);
          }

          // drift recheck on the baseline
          ModeStats base2 =
              b.time_mode(M_COMPUTE, 2, std::max(5, it_b / 4), false);

          ModeStats byst{};
          if (w_byst) byst = b.time_mode(M_BYSTANDER, 2, iters_of(est_b), false);

          double ce_comm_us = 0;
          int ce_comm_reps = 0;
          ModeStats tcomm{};
          if (w_comm) {
            if (v == V_CE) {
              ce_comm_us =
                  b.run_ce_comm_only_us(cfg.window_ms, &ce_comm_reps);
            } else {
              ModeStats cprobe = b.time_mode(M_COMMONLY, 0, 2, false);
              tcomm = b.time_mode(M_COMMONLY, 2, iters_of(max_mean(cprobe)),
                                  false);
            }
          }

          ModeStats local{};
          if (w_local) {
            local = b.time_mode(M_LOCAL, 2, iters_of(est_b), false);
            if (v == V_TMA) b.restore_staged();
          }
          ModeStats memop{};
          const bool run_memop = w_memop && v == V_CE;
          if (run_memop) memop = b.time_mode(M_MEMOP, 2, iters_of(est_b), false);

          ModeStats arr{};
          std::vector<std::vector<uint64_t>> arr_log;
          if (w_arr) {
            arr = b.run_arrival(arr_log);
            if (!cfg.dump.empty())
              for (int r = 0; r < b.world; ++r)
                if (da[r])
                  for (int it = 0; it < e4::kLogIters; ++it) {
                    const uint64_t arm =
                        arr_log[r][(size_t)it * (e4::kMaxChunks + 1) +
                                   e4::kMaxChunks];
                    for (int c = 0; c < b.n_chunks; ++c)
                      std::fprintf(da[r],
                                   "%s,%d,%d,%d,%d,%d,%d,%llu,%llu\n",
                                   b.var_name(), b.K, gran, b.n_comm, it, c,
                                   c / b.cps, (unsigned long long)arm,
                                   (unsigned long long)
                                       arr_log[r][(size_t)it *
                                                      (e4::kMaxChunks + 1) +
                                                  c]);
                  }
          }

          // ------------------------------------------------ report
          const double gbps_ce =
              ce_comm_us > 0 ? remote_bytes / (ce_comm_us * 1e-6) / 1e9 : 0;
          for (int r = 0; r < b.world; ++r) {
            const double bus = base.st[r].mean;
            const double drift =
                bus > 0 ? (base2.st[r].mean - bus) / bus * 100.0 : 0;
            const double abs_drift = drift < 0 ? -drift : drift;
            if (abs_drift > 2.0) {
              std::fprintf(stderr,
                           "rank%d paired compute baseline drift %.2f%% exceeds "
                           "2%%; refusing timing result\n",
                           r, drift);
              std::exit(1);
            }
            const double sd =
                (w_fused && bus > 0) ? fused.st[r].mean / bus : 0;
            const double isd =
                (w_byst && bus > 0) ? byst.st[r].mean / bus : 0;
            const double ssd = (sd > 0 && isd > 0) ? sd / isd : 0;
            std::printf("  rank%d: base-compute %.1fus (kernel %.1fus)", r, bus,
                        base.kernel[r].mean);
            if (w_fused)
              std::printf(" fused-compute %.1fus (kernel %.1fus) e2e %.1fus "
                          "sd %.3f",
                          fused.st[r].mean, fused.kernel[r].mean,
                          fused.e2e[r].mean, sd);
            if (w_byst) std::printf(" interf %.3f", isd);
            if (ssd > 0) std::printf(" stall %.3f", ssd);
            if (w_fused)
              std::printf(" wait p50/p95 %.1f/%.1fus", ws[r].p50, ws[r].p95);
            std::printf(" drift %+.1f%%\n", drift);

            auto row = [&](const char* mode, const Stats& st, int iters,
                           double slowdown, double interf, double stall,
                           double gbps, const Stats* w, unsigned errs,
                           double e2e_mean, double host_ms) {
              if (!csv) return;
              const unsigned long long bsum =
                  cfg.verify ? (v == V_CE ? bs_ce[r] : bs_tma[r]) : 0;
              std::fprintf(
                  csv,
                  "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%d,%d,"
                  "%d,%d,%u,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,"
                  "%.4f,%.2f,%.2f,%.2f,%.2f,%016llx,%u,%s,%s,%d,%.2f,%.4f,%d\n",
                  b.var_name(), mode, b.use_memop ? "memop" : "kernel",
                  cfg.ce_cycle_dst ? "cycle" : "fixed", r, b.world, b.M,
                  cfg.n, b.K, b.G, b.n_chunks, chunk_bytes, b.n_comm,
                  n_comm_eff, cfg.comm_streams, b.slices_eff, cfg.intensity,
                  grid, e4::kThreads, iters, b.epoch, st.mean, st.p50, st.p95,
                  st.mn, st.mx, bus, drift, slowdown, interf, stall, gbps,
                  w ? w->p50 : 0, w ? w->p95 : 0, w ? w->mx : 0, bsum, errs,
                  ver, cfg.panel_mode ? "panel" : "row", b.H, e2e_mean,
                  host_ms, v == V_CE ? b.ce_reserve_eff : 0);
            };
            row("compute-only", base.st[r], it_b, 0, 0, 0, 0, nullptr,
                base.err_count, base.e2e[r].mean, base.host_ms_per_iter);
            if (w_fused)
              row("fused", fused.st[r], it_f, sd, isd, ssd, 0, &ws[r],
                  fused.err_count, fused.e2e[r].mean,
                  fused.host_ms_per_iter);
            if (w_byst)
              row("bystander", byst.st[r], 0, 0, isd, 0, 0, nullptr,
                  byst.err_count, byst.e2e[r].mean,
                  byst.host_ms_per_iter);
            if (w_comm && v == V_TMA) {
              const double g2 = tcomm.st[r].mean > 0
                                    ? remote_bytes /
                                          (tcomm.st[r].mean * 1e-6) / 1e9
                                    : 0;
              row("comm-only", tcomm.st[r], 0, 0, 0, 0, g2, nullptr, 0,
                  tcomm.e2e[r].mean, tcomm.host_ms_per_iter);
            }
            if (w_local)
              row("local", local.st[r], 0,
                  bus > 0 ? local.st[r].mean / bus : 0, 0, 0, 0, nullptr,
                  local.err_count, local.e2e[r].mean,
                  local.host_ms_per_iter);
            if (run_memop)
              row("memop-cost", memop.st[r], 0,
                  bus > 0 ? memop.st[r].mean / bus : 0, 0, 0, 0, nullptr,
                  memop.err_count, memop.e2e[r].mean,
                  memop.host_ms_per_iter);
            if (w_arr) row("arrival", arr.st[r], e4::kLogIters, 0, 0, 0, 0,
                           nullptr, 0, arr.st[r].mean, 0);
          }
          if (w_comm && v == V_CE) {
            std::printf("  comm: %.1fus %.1fGB/s/rank (%d reps)\n", ce_comm_us,
                        gbps_ce, ce_comm_reps);
            if (csv)  // aggregate wall-clock row, rank = -1
              std::fprintf(
                  csv,
                  "%s,comm-only,%s,%s,-1,%d,%d,%d,%d,%d,%d,%zu,0,0,%d,%d,%d,"
                  "%d,%d,%d,%u,%.2f,%.2f,%.2f,%.2f,%.2f,0,0,0,0,0,%.2f,0,0,0,"
                  "%016llx,0,%s,%s,%d,%.2f,%.4f,%d\n",
                  b.var_name(), b.use_memop ? "memop" : "kernel",
                  cfg.ce_cycle_dst ? "cycle" : "fixed", b.world, b.M, cfg.n,
                  b.K, b.G, b.n_chunks, chunk_bytes, cfg.comm_streams,
                  b.slices_eff, cfg.intensity, grid, e4::kThreads, ce_comm_reps,
                  b.epoch,
                  ce_comm_us, ce_comm_us, ce_comm_us, ce_comm_us, ce_comm_us,
                  gbps_ce, 0ull, ver, cfg.panel_mode ? "panel" : "row", b.H,
                  ce_comm_us, 0.0, b.ce_reserve_eff);
          }
          if (w_comm && v == V_TMA) {
            double m = 0;
            for (int r = 0; r < b.world; ++r)
              m = std::max(m, tcomm.st[r].mean);
            std::printf("  comm: %.1fus %.1fGB/s/rank (slowest rank)\n", m,
                        m > 0 ? remote_bytes / (m * 1e-6) / 1e9 : 0);
          }
          if (w_local || run_memop) {
            double lmax = 0, mmax = 0;
            for (int r = 0; r < b.world; ++r) {
              if (base.st[r].mean <= 0) continue;
              lmax = std::max(lmax, local.st[r].mean / base.st[r].mean);
              mmax = std::max(mmax, memop.st[r].mean / base.st[r].mean);
            }
            std::printf("  controls:");
            if (w_local) std::printf(" local sd %.3f", lmax);
            if (run_memop) std::printf(" memop sd %.3f", mmax);
            std::printf("\n");
          }
          std::printf("  host enqueue: base %.2f fused %.2f ms/iter | "
                      "verify %s\n",
                      base.host_ms_per_iter, fused.host_ms_per_iter, ver);
          std::fflush(stdout);
        }
      }
    }
    b.free_for_k();
  }

  for (int r = 0; r < b.world; ++r) {
    if (dt[r]) std::fclose(dt[r]);
    if (da[r]) std::fclose(da[r]);
  }
  if (csv) std::fclose(csv);
  std::printf(
      "\nt_us_* = earliest persistent-block start to last compute-block done "
      "(same-GPU %%globaltimer); e2e_us_mean = whole-kernel CUDA-event "
      "interval.\nslowdown = fused/compute-only (same variant & grid). "
      "interference_sd"
      " = bystander/compute-only\n(comm running, gates pre-armed -> pure "
      "bandwidth/engine interference). stall_sd = slowdown /\ninterference_sd"
      " is a descriptive fused/bystander ratio, not an additive causal "
      "decomposition.\nlocal & memop-cost are diagnostic controls. CE rows: "
      "full grid computes,"
      " copies ride the copy engine.\nTMA rows: n_comm blocks of the same "
      "kernel pull panels via tensor-map TMA instead.\n");
  b.stop_workers();
  return 0;
}
