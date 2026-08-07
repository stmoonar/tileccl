// exp2_ag_gemm_granularity.cu
//
// Experiment 2: how does the CE transfer granularity trade off against GEMM
// efficiency in an AG+GEMM overlap?
//
// This is a real (dependent) all-gather + GEMM pipeline in the style of
// flux's AG+GEMM, built the "segmented launch" way: the gathered A buffer
// on the GEMM rank is split into world*S row-chunks (S = chunks per remote
// shard, the sweep axis). CE copies pull each remote chunk over NVLink into
// its slot of A_full; a per-chunk CUDA event is recorded on the comm stream
// and the compute stream waits on it before launching the CUTLASS sm90 GEMM
// segment that consumes those rows. The local shard needs no copy and its
// segments are launched first, hiding the first chunk's flight time.
//
// (flux's sm90 kernel instead keeps ONE persistent GEMM whose tiles spin on
// arrival flags inside the kernel; the segmented-launch form measures the
// same granularity trade-off while staying on stock CUTLASS. Its extra cost
// per chunk -- one event record + one stream-wait + one launch -- is the
// same order as flux's per-chunk signalling, and is itself part of the
// trade-off being measured.)
//
// The sweep exposes both sides of the trade-off:
//   fine chunks   -> GEMM segments start earlier (less exposed comm) but
//                    small-M segments waste waves (tile quantization) and
//                    small messages waste CE bandwidth
//   coarse chunks -> full-efficiency GEMM and copies, but the first/last
//                    chunk's flight time cannot be hidden
//
// Per S we report the three isolated components and the pipeline:
//   t_full : unsegmented GEMM, no comm        (compute roofline)
//   t_seg  : segmented GEMM, no comm          (pure quantization cost)
//   t_comm : copies only                      (pure CE cost at that msg size)
//   t_ovl  : the actual pipeline (single-shot latency, iterations gated so
//            iteration i+1's copies start only after iteration i finished)
//   overlap_eff = (t_seg + t_comm - t_ovl) / min(t_seg, t_comm)
//                 (1.0 = comm fully hidden behind compute or vice versa)
//
// --verify runs the pipeline once against a pre-gathered full GEMM and
// compares D bitwise -- if the event dependencies were wrong, remote
// segments would consume stale A and this catches it.
//
// See comm_comp/README.md for build & run lines.

#include "common.cuh"
#include "gemm_sm90.cuh"

#include <cstdio>
#include <string>
#include <vector>

using cc::GemmCfgTma;
using cc::GemmOp;

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct Config {
  std::vector<Shape3> sizes;      // M is the TOTAL (gathered) M
  std::vector<int> chunks;        // S values: chunks per shard
  int comm_streams = 0;           // 0 = one per peer, 1 = single serial stream
  bool push = false;              // copies issued by the peers' CEs instead
  double window_ms = 200;
  int iters = 0;                  // pipeline iterations, 0 = auto
  int gemm_dev = 0;
  int ndev = 0;
  bool verify = false;
  std::string csv;
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --sizes LIST    gathered GEMM shapes, N or MxNxK (default 8192)\n"
      "                  M must be divisible by world * S\n"
      "  --chunks LIST   chunks per remote shard (default 1,2,4,8,16,32)\n"
      "  --comm-streams N  1 = serial ring order, 0 = one stream per peer\n"
      "                  (default 0)\n"
      "  --push          peers' CEs push chunks into the GEMM rank\n"
      "                  (default: GEMM rank pulls)\n"
      "  --window-ms MS  target timing window (default 200)\n"
      "  --iters N       pipeline iterations, 0 = auto\n"
      "  --gemm-dev N    device running the GEMM (default 0)\n"
      "  --ndev N        use only the first N GPUs (default: all)\n"
      "  --verify        check pipeline D against a pre-gathered full GEMM\n"
      "  --csv PATH      append machine-readable rows\n",
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
    if (a == "--sizes") {
      for (const auto& t : split_csv(next())) c.sizes.push_back(parse_shape(t));
    } else if (a == "--chunks") {
      for (const auto& t : split_csv(next()))
        c.chunks.push_back(std::atoi(t.c_str()));
    } else if (a == "--comm-streams") c.comm_streams = std::atoi(next().c_str());
    else if (a == "--push") c.push = true;
    else if (a == "--window-ms") c.window_ms = std::atof(next().c_str());
    else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--gemm-dev") c.gemm_dev = std::atoi(next().c_str());
    else if (a == "--ndev") c.ndev = std::atoi(next().c_str());
    else if (a == "--verify") c.verify = true;
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (c.sizes.empty()) c.sizes.push_back({8192, 8192, 8192});
  if (c.chunks.empty()) c.chunks = {1, 2, 4, 8, 16, 32};
  return c;
}

// ---------------------------------------------------------------------------
// the pipeline for one (shape, S)
// ---------------------------------------------------------------------------

constexpr int kMaxPipeIters = 256;

struct Segment {
  int rank = 0, chunk = 0;
  bool local = false;
  GemmOp<GemmCfgTma> op;
};

struct Pipeline {
  // problem
  int world = 0, g = 0, S = 1;
  Shape3 sh;                 // sh.m = gathered M
  int m_shard = 0, chunk_rows = 0;
  size_t chunk_bytes = 0;
  bool push = false;
  int n_comm_streams = 0;

  // device memory
  cc::ElementAB* A_full = nullptr;             // [M, K] on g
  std::vector<cc::ElementAB*> A_src;           // [M_shard, K] on each rank
  cc::ElementAB *B = nullptr, *C = nullptr, *D = nullptr;

  // streams / events / ops
  cudaStream_t compute = nullptr;
  std::vector<cudaStream_t> comm;              // comm streams (see cfg)
  std::vector<int> comm_dev;                   // device owning comm[i]
  std::vector<Segment> segs;                   // in compute launch order
  std::vector<cudaEvent_t> ev_chunk;           // [world][S], remote only used
  cudaEvent_t e_beg = nullptr;
  std::vector<cudaEvent_t> iter_done;

  // ring order of the remote ranks, nearest first: g+1, g+2, ...
  std::vector<int> ring;

  cudaEvent_t& chunk_ev(int r, int c) { return ev_chunk[r * S + c]; }

  void init(int world_, int g_, Shape3 sh_, int S_, bool push_,
            int comm_streams_cfg) {
    world = world_;
    g = g_;
    sh = sh_;
    S = S_;
    push = push_;
    m_shard = sh.m / world;
    chunk_rows = m_shard / S;
    chunk_bytes = (size_t)chunk_rows * sh.k * sizeof(cc::ElementAB);
    for (int i = 1; i < world; ++i) ring.push_back((g + i) % world);
    n_comm_streams = comm_streams_cfg == 1 ? 1 : (world - 1);

    // buffers
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMalloc(&A_full, (size_t)sh.m * sh.k * 2));
    CUDA_CHECK(cudaMalloc(&B, (size_t)sh.k * sh.n * 2));
    CUDA_CHECK(cudaMalloc(&C, (size_t)sh.m * sh.n * 2));
    CUDA_CHECK(cudaMalloc(&D, (size_t)sh.m * sh.n * 2));
    fill_half(B, (size_t)sh.k * sh.n, 12345);
    fill_half(C, (size_t)sh.m * sh.n, 777);
    CUDA_CHECK(cudaMemset(D, 0, (size_t)sh.m * sh.n * 2));
    A_src.assign(world, nullptr);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMalloc(&A_src[r], (size_t)m_shard * sh.k * 2));
      fill_half(A_src[r], (size_t)m_shard * sh.k, 1000 + r);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    // the local shard sits in place from the start (as in flux)
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaMemcpy(A_full + (size_t)g * m_shard * sh.k, A_src[g],
                          (size_t)m_shard * sh.k * 2,
                          cudaMemcpyDeviceToDevice));

    // streams: compute on g; comm streams on g (pull) or on the peers (push)
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    comm.resize(n_comm_streams);
    comm_dev.resize(n_comm_streams);
    for (int i = 0; i < n_comm_streams; ++i) {
      const int owner = push ? ring[i % ring.size()] : g;
      comm_dev[i] = owner;
      CUDA_CHECK(cudaSetDevice(owner));
      // flux runs its AG copies on a highest-priority stream; match that
      int lo = 0, hi = 0;
      CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
      CUDA_CHECK(cudaStreamCreateWithPriority(&comm[i], cudaStreamNonBlocking,
                                              hi));
    }

    // events (chunk events must live on the device that records them)
    ev_chunk.resize((size_t)world * S);
    for (int i = 0; i < (int)ring.size(); ++i) {
      const int r = ring[i];
      const int owner = push ? r : g;
      CUDA_CHECK(cudaSetDevice(owner));
      for (int c = 0; c < S; ++c)
        CUDA_CHECK(cudaEventCreateWithFlags(&chunk_ev(r, c),
                                            cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaEventCreate(&e_beg));
    iter_done.resize(kMaxPipeIters);
    for (auto& e : iter_done) CUDA_CHECK(cudaEventCreate(&e));

    // segments, in compute launch order: local shard first, then remote in
    // expected arrival order (chunk-major across peers when the pulls run in
    // parallel, rank-major when a single stream serializes the ring)
    segs.reserve((size_t)world * S);
    auto add_seg = [&](int r, int c) {
      segs.emplace_back();
      Segment& s = segs.back();
      s.rank = r;
      s.chunk = c;
      s.local = (r == g);
      const size_t row0 = (size_t)r * m_shard + (size_t)c * chunk_rows;
      s.op.init(chunk_rows, sh.n, sh.k, A_full + row0 * sh.k, B,
                C + row0 * sh.n, D + row0 * sh.n, 1.0f, 0.0f, g);
    };
    for (int c = 0; c < S; ++c) add_seg(g, c);
    if (n_comm_streams == 1) {
      for (int r : ring)
        for (int c = 0; c < S; ++c) add_seg(r, c);
    } else {
      for (int c = 0; c < S; ++c)
        for (int r : ring) add_seg(r, c);
    }
  }

  // stream that carries (remote) rank r's chunks
  int stream_of(int r) const {
    if (n_comm_streams == 1) return 0;
    for (size_t i = 0; i < ring.size(); ++i)
      if (ring[i] == r) return (int)(i % n_comm_streams);
    return 0;
  }

  void enqueue_chunk_copy(int r, int c) {
    const int si = stream_of(r);
    CUDA_CHECK(cudaSetDevice(comm_dev[si]));
    cc::ElementAB* dst =
        A_full + ((size_t)r * m_shard + (size_t)c * chunk_rows) * sh.k;
    const cc::ElementAB* src = A_src[r] + (size_t)c * chunk_rows * sh.k;
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, g, src, r, chunk_bytes, comm[si]));
    CUDA_CHECK(cudaEventRecord(chunk_ev(r, c), comm[si]));
  }

  // copy enqueue order, matching the segment order above
  void enqueue_all_copies() {
    if (n_comm_streams == 1) {
      for (int r : ring)
        for (int c = 0; c < S; ++c) enqueue_chunk_copy(r, c);
    } else {
      for (int c = 0; c < S; ++c)
        for (int r : ring) enqueue_chunk_copy(r, c);
    }
  }

  // One pipeline iteration, gated on `gate` (copies may not start earlier).
  void enqueue_iter(cudaEvent_t gate, cudaEvent_t done) {
    for (int i = 0; i < n_comm_streams; ++i) {
      CUDA_CHECK(cudaSetDevice(comm_dev[i]));
      CUDA_CHECK(cudaStreamWaitEvent(comm[i], gate, 0));
    }
    enqueue_all_copies();
    CUDA_CHECK(cudaSetDevice(g));
    for (auto& s : segs) {
      if (!s.local)
        CUDA_CHECK(cudaStreamWaitEvent(compute, chunk_ev(s.rank, s.chunk), 0));
      s.op.run(compute);
    }
    CUDA_CHECK(cudaEventRecord(done, compute));
  }

  // timed pipeline: returns per-iter stats (us)
  Stats run_pipeline(int warm, int iters) {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaEventRecord(e_beg, compute));
    cudaEvent_t gate = e_beg;
    for (int i = 0; i < warm; ++i) {
      enqueue_iter(gate, iter_done[0]);
      gate = iter_done[0];
    }
    // re-anchor timing after warmup
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaEventRecord(e_beg, compute));
    gate = e_beg;
    for (int i = 0; i < iters; ++i) {
      enqueue_iter(gate, iter_done[i]);
      gate = iter_done[i];
    }
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaEventSynchronize(iter_done[iters - 1]));
    std::vector<double> per(iters);
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_beg, iter_done[0]));
    per[0] = ms * 1000.0;
    for (int i = 1; i < iters; ++i) {
      CUDA_CHECK(cudaEventElapsedTime(&ms, iter_done[i - 1], iter_done[i]));
      per[i] = ms * 1000.0;
    }
    return make_stats(per);
  }

  // segmented GEMM alone (no copies, no waits)
  Stats run_seg_alone(int warm, int iters) {
    CUDA_CHECK(cudaSetDevice(g));
    for (int i = 0; i < warm; ++i)
      for (auto& s : segs) s.op.run(compute);
    CUDA_CHECK(cudaEventRecord(e_beg, compute));
    for (int i = 0; i < iters; ++i) {
      for (auto& s : segs) s.op.run(compute);
      CUDA_CHECK(cudaEventRecord(iter_done[i], compute));
    }
    CUDA_CHECK(cudaEventSynchronize(iter_done[iters - 1]));
    std::vector<double> per(iters);
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_beg, iter_done[0]));
    per[0] = ms * 1000.0;
    for (int i = 1; i < iters; ++i) {
      CUDA_CHECK(cudaEventElapsedTime(&ms, iter_done[i - 1], iter_done[i]));
      per[i] = ms * 1000.0;
    }
    return make_stats(per);
  }

  // copies alone: wall time over reps of {enqueue everything, sync}
  double run_comm_alone_us(int reps) {
    for (int w = 0; w < 2; ++w) {  // warm
      enqueue_all_copies();
      sync_comm();
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
      enqueue_all_copies();
      sync_comm();
    }
    return wall_ms_since(t0) * 1000.0 / reps;
  }

  void sync_comm() {
    for (int i = 0; i < n_comm_streams; ++i) {
      CUDA_CHECK(cudaSetDevice(comm_dev[i]));
      CUDA_CHECK(cudaStreamSynchronize(comm[i]));
    }
  }

  void destroy() {
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaStreamSynchronize(compute));
    sync_comm();
    for (auto& s : segs) s.op.destroy();
    segs.clear();
    for (int i = 0; i < (int)ring.size(); ++i)
      for (int c = 0; c < S; ++c) cudaEventDestroy(chunk_ev(ring[i], c));
    ev_chunk.clear();
    cudaEventDestroy(e_beg);
    for (auto& e : iter_done) cudaEventDestroy(e);
    iter_done.clear();
    for (int i = 0; i < n_comm_streams; ++i) {
      cudaSetDevice(comm_dev[i]);
      cudaStreamDestroy(comm[i]);
    }
    comm.clear();
    cudaSetDevice(g);
    cudaStreamDestroy(compute);
    cudaFree(A_full); cudaFree(B); cudaFree(C); cudaFree(D);
    for (int r = 0; r < world; ++r) {
      cudaSetDevice(r);
      cudaFree(A_src[r]);
    }
  }
};

// bitwise compare of two device fp16 buffers via host
static bool device_equal(const void* a, const void* b, size_t bytes) {
  std::vector<uint8_t> ha(bytes), hb(bytes);
  CUDA_CHECK(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost));
  return std::memcmp(ha.data(), hb.data(), bytes) == 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int world = 0;
  CUDA_CHECK(cudaGetDeviceCount(&world));
  if (cfg.ndev > 0) world = std::min(world, cfg.ndev);
  if (world < 2) {
    std::fprintf(stderr, "exp2 needs at least 2 GPUs\n");
    return 1;
  }
  if (cfg.gemm_dev >= world) {
    std::fprintf(stderr, "--gemm-dev %d out of range (world %d)\n",
                 cfg.gemm_dev, world);
    return 1;
  }
  std::vector<int> devs;
  for (int r = 0; r < world; ++r) devs.push_back(r);
  enable_peer_all(devs);

  std::printf("=== exp2: AG+GEMM overlap granularity sweep ===\n");
  std::printf("world %d, GEMM on GPU %d, %s mode, %s\n", world, cfg.gemm_dev,
              cfg.push ? "push" : "pull",
              cfg.comm_streams == 1 ? "single serial comm stream"
                                    : "one comm stream per peer");
  for (int r = 0; r < world; ++r) print_device_line(r);
  std::printf("\n");

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv,
                   "m,n,k,world,S,chunk_rows,chunk_mib,mode,streams,"
                   "t_full_us,t_seg_us,t_comm_us,t_ovl_us,seg_eff,"
                   "comm_gbps,overlap_eff,e2e_tflops,verify\n");
  }

  for (const Shape3& sh : cfg.sizes) {
    if (sh.m % world) {
      std::printf("skip %dx%dx%d: M not divisible by world %d\n", sh.m, sh.n,
                  sh.k, world);
      continue;
    }
    if ((sh.k % cc::kAlign) || (sh.n % cc::kAlign)) {
      std::printf("skip %dx%dx%d: N,K must be multiples of %d\n", sh.m, sh.n,
                  sh.k, cc::kAlign);
      continue;
    }
    const int m_shard = sh.m / world;
    const double flop = 2.0 * sh.m * sh.n * sh.k;
    const double ag_bytes = (double)(world - 1) * m_shard * sh.k * 2;

    // full-GEMM baseline (its own buffers, freed before the sweep)
    double t_full_us = 0;
    {
      Pipeline p;
      p.init(world, cfg.gemm_dev, sh, 1, cfg.push, cfg.comm_streams);
      // pre-gather so the full GEMM reads valid data
      p.enqueue_all_copies();
      p.sync_comm();
      GemmOp<GemmCfgTma> full;
      full.init(sh.m, sh.n, sh.k, p.A_full, p.B, p.C, p.D, 1.0f, 0.0f,
                cfg.gemm_dev);
      CUDA_CHECK(cudaSetDevice(cfg.gemm_dev));
      for (int i = 0; i < 3; ++i) full.run(p.compute);
      CUDA_CHECK(cudaEventRecord(p.e_beg, p.compute));
      const int reps = 20;
      for (int i = 0; i < reps; ++i) {
        full.run(p.compute);
        CUDA_CHECK(cudaEventRecord(p.iter_done[i], p.compute));
      }
      CUDA_CHECK(cudaEventSynchronize(p.iter_done[reps - 1]));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, p.e_beg, p.iter_done[reps - 1]));
      t_full_us = ms * 1000.0 / reps;
      full.destroy();
      p.destroy();
    }
    const double tflops_full = flop / (t_full_us * 1e-6) / 1e12;

    std::printf("--- %dx%dx%d fp16 (shard M=%d, AG volume %.1f MiB): "
                "full GEMM %.1f us, %.1f TFLOP/s ---\n",
                sh.m, sh.n, sh.k, m_shard, ag_bytes / (1 << 20), t_full_us,
                tflops_full);
    std::printf("%4s %6s %9s | %9s %7s | %9s %7s | %9s %7s %8s %6s | %s\n",
                "S", "rows", "chunkMiB", "t_seg us", "segeff", "t_comm",
                "GB/s", "t_ovl us", "TFLOPs", "ovl_eff", "bub%", "ver");

    for (int S : cfg.chunks) {
      if (S < 1 || m_shard % S) {
        std::printf("%4d : skipped (shard M %d not divisible)\n", S, m_shard);
        continue;
      }
      Pipeline p;
      p.init(world, cfg.gemm_dev, sh, S, cfg.push, cfg.comm_streams);

      // verify (before timing, so a broken dependency fails fast)
      bool ver_ok = true;
      if (cfg.verify) {
        // reference: pre-gather + segmented compute with no dependencies
        p.enqueue_all_copies();
        p.sync_comm();
        (void)p.run_seg_alone(0, 1);
        CUDA_CHECK(cudaStreamSynchronize(p.compute));
        void* D_ref = nullptr;
        const size_t d_bytes = (size_t)sh.m * sh.n * 2;
        CUDA_CHECK(cudaMalloc(&D_ref, d_bytes));
        CUDA_CHECK(cudaMemcpy(D_ref, p.D, d_bytes, cudaMemcpyDeviceToDevice));
        // scrub the remote regions of A_full, then run the real pipeline
        for (int r : p.ring)
          CUDA_CHECK(cudaMemset(p.A_full + (size_t)r * m_shard * sh.k, 0x55,
                                (size_t)m_shard * sh.k * 2));
        CUDA_CHECK(cudaMemset(p.D, 0, d_bytes));
        // the memsets run on the legacy default stream, which does NOT
        // implicitly order against the non-blocking pipeline streams
        CUDA_CHECK(cudaDeviceSynchronize());
        (void)p.run_pipeline(0, 1);
        ver_ok = device_equal(p.D, D_ref, d_bytes);
        cudaFree(D_ref);
      }

      // components
      const Stats seg = p.run_seg_alone(2, 10);
      const double comm_us = p.run_comm_alone_us(10);
      const double comm_gbps = ag_bytes / (comm_us * 1e-6) / 1e9;

      // pipeline
      const double est_us = std::max(seg.mean, comm_us);
      int iters = cfg.iters > 0
                      ? std::min(cfg.iters, kMaxPipeIters)
                      : (int)std::max(5.0, std::min((double)kMaxPipeIters,
                                                    cfg.window_ms * 1000.0 /
                                                        std::max(1.0, est_us)));
      const Stats ovl = p.run_pipeline(2, iters);

      const double seg_eff = seg.mean > 0 ? t_full_us / seg.mean : 0;
      const double ovl_eff =
          (seg.mean + comm_us - ovl.mean) / std::min(seg.mean, comm_us);
      const double e2e_tflops = flop / (ovl.mean * 1e-6) / 1e12;
      const double ideal = std::max(seg.mean, comm_us);
      const double bubble = ideal > 0 ? (ovl.mean - ideal) / ideal * 100 : 0;
      const double chunk_mib = (double)p.chunk_bytes / (1 << 20);

      std::printf("%4d %6d %9.2f | %9.1f %7.3f | %9.1f %7.1f | %9.1f %7.1f "
                  "%8.3f %5.1f%% | %s\n",
                  S, p.chunk_rows, chunk_mib, seg.mean, seg_eff, comm_us,
                  comm_gbps, ovl.mean, e2e_tflops, ovl_eff, bubble,
                  !cfg.verify ? "-" : (ver_ok ? "ok" : "FAIL"));
      std::fflush(stdout);

      if (csv)
        std::fprintf(csv,
                     "%d,%d,%d,%d,%d,%d,%.3f,%s,%d,%.2f,%.2f,%.2f,%.2f,"
                     "%.4f,%.2f,%.4f,%.1f,%s\n",
                     sh.m, sh.n, sh.k, world, S, p.chunk_rows, chunk_mib,
                     cfg.push ? "push" : "pull", p.n_comm_streams, t_full_us,
                     seg.mean, comm_us, ovl.mean, seg_eff, comm_gbps, ovl_eff,
                     e2e_tflops, !cfg.verify ? "-" : (ver_ok ? "ok" : "FAIL"));
      p.destroy();
    }
    std::printf("\n");
  }

  std::printf(
      "seg_eff = t_full/t_seg (segmentation-only GEMM efficiency).\n"
      "ovl_eff = (t_seg + t_comm - t_ovl) / min(t_seg, t_comm); 1.0 means "
      "the\nshorter phase is fully hidden. bub%% = t_ovl over max(t_seg, "
      "t_comm).\nThe sweet spot is the S with the lowest t_ovl: finer chunks "
      "start compute\nearlier but pay tile quantization + per-chunk "
      "signalling; coarser chunks\nexpose the first chunk's flight time.\n");

  if (csv) std::fclose(csv);
  return 0;
}
