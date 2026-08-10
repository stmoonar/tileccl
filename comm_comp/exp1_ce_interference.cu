// exp1_ce_interference.cu
//
// Experiment 1: does copy-engine (CE) traffic cost an *independent* GEMM any
// FLOPS?
//
// A CUTLASS sm90 fp16 GEMM runs on --gemm-dev while CE copies shuttle
// unrelated data between ranks, shaped like the transfers flux's AG+GEMM
// issues (cudaMemcpyPeerAsync bursts between peer ranks). The GEMM never
// consumes the copied data; the control is the same GEMM with no CE traffic.
// We report per-iteration GEMM latency (CUDA events) alone vs overlapped:
//
//     S_c = t_overlap / t_alone     (1.00 => CE traffic is free for compute)
//
// Traffic patterns (--patterns), g = GEMM device, chosen to separate "local
// HBM bandwidth contention" from "the CE engine per se" from "fabric load":
//   pull        g's CE reads each peer's shard over NVLink and writes local
//               HBM (flux AG pull mode: world-1 legs on g)
//   push        each peer's CE writes its shard into g's HBM (flux AG push
//               mode: world-1 legs, executed on the peers)
//   allgather   every rank pulls from every other rank -- the full ambient
//               traffic of a real all-gather (world*(world-1) legs)
//   bystander   ring traffic among the other ranks only; g's HBM and CE are
//               untouched (fabric-only control, expect S_c ~ 1.00)
//   engine-only g's CE copies between two *other* ranks: engine busy, zero
//               local HBM traffic (needs world >= 3)
//   local       plain D2D copy on g (CE reads AND writes local HBM -- the
//               heaviest per-byte local load, upper bound for contention)
//
// The host keeps every leg saturated with a ring of chunked bursts
// (re-enqueued as their events complete), so CE is busy for the entire GEMM
// window; if a leg is ever observed idle the row is flagged and S_c is a
// lower bound. Copy offsets cycle through --comm-buf so CE traffic streams
// HBM instead of hitting a hot L2 line.
//
// --msgs sweeps the CE message size (the granularity axis of experiment 2,
// here in the independent/no-dependency setting).
//
// Caveat: lock clocks first (nvidia-smi -lgc), DVFS masquerades as
// contention. See comm_comp/README.md for build & run lines.

#include "common.cuh"
#include "gemm_sm90.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using cc::GemmCfgTma;
using cc::GemmOp;

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct Config {
  std::vector<Shape3> sizes;
  std::vector<std::string> patterns;
  std::vector<uint64_t> msgs;
  uint64_t comm_buf = 1ull << 30;  // per (dst-rank) cycling region
  double window_ms = 200;
  int iters = 0;  // 0 = auto from window
  int gemm_dev = 0;
  int ndev = 0;  // 0 = all visible
  std::string csv;
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --sizes LIST     GEMM shapes, N or MxNxK (default 2048,4096,8192)\n"
      "  --patterns LIST  subset of pull,push,allgather,bystander,\n"
      "                   engine-only,local (default: all that fit ndev)\n"
      "  --msgs LIST      CE message sizes to sweep (default 64M)\n"
      "  --comm-buf B     cycling copy region per dst rank (default 1G)\n"
      "  --window-ms MS   target GEMM timing window (default 200)\n"
      "  --iters N        GEMM iterations, 0 = auto\n"
      "  --gemm-dev N     device running the GEMM (default 0)\n"
      "  --ndev N         use only the first N GPUs (default: all)\n"
      "  --csv PATH       append machine-readable rows\n",
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
    } else if (a == "--patterns") c.patterns = split_csv(next());
    else if (a == "--msgs") {
      for (const auto& t : split_csv(next())) c.msgs.push_back(parse_bytes(t));
    } else if (a == "--comm-buf") c.comm_buf = parse_bytes(next());
    else if (a == "--window-ms") c.window_ms = std::atof(next().c_str());
    else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--gemm-dev") c.gemm_dev = std::atoi(next().c_str());
    else if (a == "--ndev") c.ndev = std::atoi(next().c_str());
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (c.sizes.empty())
    for (int n : {2048, 4096, 8192}) c.sizes.push_back({n, n, n});
  if (c.msgs.empty()) c.msgs.push_back(64ull << 20);
  return c;
}

// ---------------------------------------------------------------------------
// legs: one (stream, src, dst) copy lane; the pump keeps each saturated
// ---------------------------------------------------------------------------

struct Leg {
  int exec_dev = -1;  // device whose stream (and thus CE) issues the copy
  int src_dev = -1, dst_dev = -1;
  uint8_t* src = nullptr;  // base of the cycling region on src_dev
  uint8_t* dst = nullptr;  // base of the cycling region on dst_dev
  uint64_t region_msgs = 1;
  uint64_t msg = 0;
  uint64_t next = 0;
  cudaStream_t stream = nullptr;

  void enqueue_msg() {
    const uint64_t off = (next++ % region_msgs) * msg;
    if (src_dev == dst_dev) {
      CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, msg,
                                 cudaMemcpyDeviceToDevice, stream));
    } else {
      CUDA_CHECK(cudaMemcpyPeerAsync(dst + off, dst_dev, src + off, src_dev,
                                     msg, stream));
    }
  }
  void enqueue_msgs(uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) enqueue_msg();
  }
};

// Per-rank buffers. shard[r] is the copy source on rank r; gather[r] is the
// copy destination region on rank r, sliced into world-1 slots so several
// legs can write into one rank without aliasing.
struct Buffers {
  int world = 0;
  uint64_t bytes = 0;  // per rank, both shard and gather
  std::vector<uint8_t*> shard, gather;

  void init(int world_, uint64_t bytes_) {
    world = world_;
    bytes = bytes_;
    shard.assign(world, nullptr);
    gather.assign(world, nullptr);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMalloc(&shard[r], bytes));
      CUDA_CHECK(cudaMemset(shard[r], 1 + r, bytes));
      CUDA_CHECK(cudaMalloc(&gather[r], bytes));
      CUDA_CHECK(cudaMemset(gather[r], 0x80 + r, bytes));
    }
    // memsets sit on the legacy default streams; make sure they are done
    // before the first calibration copies run on non-blocking streams
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
  void destroy() {
    for (int r = 0; r < world; ++r) {
      cudaSetDevice(r);
      cudaFree(shard[r]);
      cudaFree(gather[r]);
    }
  }
};

// slot index of source rank s among "all ranks except dst"
static int slot_of(int s, int dst, int world) {
  int idx = 0;
  for (int r = 0; r < world; ++r) {
    if (r == dst) continue;
    if (r == s) return idx;
    ++idx;
  }
  return 0;  // unreachable
}

// Build the legs for a pattern; returns false if the pattern needs more
// ranks than we have.
static bool build_legs(const std::string& pat, int g, int world, Buffers& buf,
                       uint64_t msg, std::vector<Leg>& legs) {
  legs.clear();
  const int nslots = std::max(1, world - 1);
  const uint64_t slot_bytes = buf.bytes / nslots;
  if (msg > slot_bytes) {
    std::fprintf(stderr, "--msgs %llu exceeds comm-buf slot %llu\n",
                 (unsigned long long)msg, (unsigned long long)slot_bytes);
    std::exit(1);
  }
  auto add = [&](int exec, int s, int d) {
    Leg l;
    l.exec_dev = exec;
    l.src_dev = s;
    l.dst_dev = d;
    l.msg = msg;
    l.region_msgs = std::max<uint64_t>(1, slot_bytes / msg);
    l.src = buf.shard[s];
    l.dst = buf.gather[d] + (uint64_t)slot_of(s, d, world) * slot_bytes;
    legs.push_back(l);
  };
  std::vector<int> others;
  for (int r = 0; r < world; ++r)
    if (r != g) others.push_back(r);

  if (pat == "local") {
    add(g, g, g);  // shard[g] -> gather[g], same-device slot 0 is fine
  } else if (pat == "pull") {
    if (world < 2) return false;
    for (int p : others) add(g, p, g);
  } else if (pat == "push") {
    if (world < 2) return false;
    for (int p : others) add(p, p, g);
  } else if (pat == "allgather") {
    if (world < 2) return false;
    for (int r = 0; r < world; ++r)
      for (int p = 0; p < world; ++p)
        if (p != r) add(r, p, r);
  } else if (pat == "bystander") {
    if (world < 3) return false;
    for (size_t i = 0; i < others.size(); ++i)
      add(others[i], others[(i + 1) % others.size()], others[i]);
  } else if (pat == "engine-only") {
    if (world < 3) return false;
    add(g, others[0], others[1]);
  } else {
    std::fprintf(stderr, "unknown pattern '%s'\n", pat.c_str());
    std::exit(1);
  }
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&l.stream, cudaStreamNonBlocking));
  }
  return true;
}

static void destroy_legs(std::vector<Leg>& legs) {
  for (auto& l : legs) {
    cudaSetDevice(l.exec_dev);
    cudaStreamSynchronize(l.stream);
    cudaStreamDestroy(l.stream);
  }
  legs.clear();
}

// ---------------------------------------------------------------------------
// pump: ring of chunked bursts per leg, re-enqueued as their events complete
// ---------------------------------------------------------------------------

struct Pump {
  static constexpr int kRing = 4;
  std::vector<Leg>* legs = nullptr;
  uint64_t chunk_msgs = 1;
  std::vector<std::array<cudaEvent_t, kRing>> ev;
  uint64_t completed_bytes = 0;
  int gaps = 0;

  void start(std::vector<Leg>* legs_, uint64_t chunk_msgs_) {
    legs = legs_;
    chunk_msgs = chunk_msgs_;
    completed_bytes = 0;
    gaps = 0;
    ev.resize(legs->size());
    for (size_t i = 0; i < legs->size(); ++i) {
      Leg& l = (*legs)[i];
      CUDA_CHECK(cudaSetDevice(l.exec_dev));
      for (int r = 0; r < kRing; ++r) {
        CUDA_CHECK(cudaEventCreate(&ev[i][r]));
        l.enqueue_msgs(chunk_msgs);
        CUDA_CHECK(cudaEventRecord(ev[i][r], l.stream));
      }
    }
  }

  void poll() {
    for (size_t i = 0; i < legs->size(); ++i) {
      Leg& l = (*legs)[i];
      CUDA_CHECK(cudaSetDevice(l.exec_dev));
      if (cudaStreamQuery(l.stream) == cudaSuccess) ++gaps;
      for (int r = 0; r < kRing; ++r) {
        if (cudaEventQuery(ev[i][r]) == cudaSuccess) {
          completed_bytes += chunk_msgs * l.msg;
          l.enqueue_msgs(chunk_msgs);
          CUDA_CHECK(cudaEventRecord(ev[i][r], l.stream));
        }
      }
    }
    cudaGetLastError();  // clear cudaErrorNotReady from the queries
  }

  void drain() {
    for (size_t i = 0; i < legs->size(); ++i) {
      Leg& l = (*legs)[i];
      CUDA_CHECK(cudaSetDevice(l.exec_dev));
      CUDA_CHECK(cudaStreamSynchronize(l.stream));
      for (int r = 0; r < kRing; ++r) cudaEventDestroy(ev[i][r]);
    }
    ev.clear();
  }
};

// Aggregate bandwidth of the legs with nothing else running, and a chunk
// size (~8 ms of msgs) for the pump. Wall-clock timed; the burst is long
// enough that enqueue overhead is noise.
struct CommCal {
  double gbps = 0;
  uint64_t chunk_msgs = 1;
};

static CommCal calibrate(std::vector<Leg>& legs, double window_ms) {
  // warm the P2P paths
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    l.enqueue_msgs(4);
  }
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    CUDA_CHECK(cudaStreamSynchronize(l.stream));
  }
  // per-msg time on the slowest leg (they're symmetric within a pattern)
  auto t0 = std::chrono::steady_clock::now();
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    l.enqueue_msgs(16);
  }
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    CUDA_CHECK(cudaStreamSynchronize(l.stream));
  }
  const double ms16 = wall_ms_since(t0);
  const double per_msg_ms = std::max(1e-6, ms16 / 16.0);

  CommCal cal;
  cal.chunk_msgs = (uint64_t)std::max(
      1.0, std::min(50000.0, 8.0 / per_msg_ms));
  const uint64_t burst =
      (uint64_t)std::max(16.0, std::min(20000.0, window_ms / per_msg_ms));
  t0 = std::chrono::steady_clock::now();
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    l.enqueue_msgs(burst);
  }
  for (auto& l : legs) {
    CUDA_CHECK(cudaSetDevice(l.exec_dev));
    CUDA_CHECK(cudaStreamSynchronize(l.stream));
  }
  const double ms = wall_ms_since(t0);
  cal.gbps = (double)burst * legs.size() * legs[0].msg / (ms * 1e-3) / 1e9;
  return cal;
}

// ---------------------------------------------------------------------------
// GEMM runner: CUTLASS sm90 fp16, per-iteration events
// ---------------------------------------------------------------------------

constexpr int kMaxIters = 2048;

struct GemmRunner {
  int dev = -1;
  Shape3 s;
  GemmOp<GemmCfgTma> op;
  cudaStream_t stream = nullptr;
  void *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
  cudaEvent_t e_beg = nullptr, e_end = nullptr;
  std::vector<cudaEvent_t> ev;

  void init(int dev_, Shape3 s_) {
    dev = dev_;
    s = s_;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMalloc(&A, (size_t)s.m * s.k * 2));
    CUDA_CHECK(cudaMalloc(&B, (size_t)s.k * s.n * 2));
    CUDA_CHECK(cudaMalloc(&C, (size_t)s.m * s.n * 2));
    CUDA_CHECK(cudaMalloc(&D, (size_t)s.m * s.n * 2));
    fill_half(A, (size_t)s.m * s.k, 1);
    fill_half(B, (size_t)s.k * s.n, 2);
    fill_half(C, (size_t)s.m * s.n, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    op.init(s.m, s.n, s.k, A, B, C, D, 1.0f, 0.0f, dev);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&e_beg));
    CUDA_CHECK(cudaEventCreate(&e_end));
    ev.resize(kMaxIters);
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  }

  // Warmup + timed run; does NOT synchronize. `tick` keeps the pump fed
  // while the host is still enqueueing.
  void enqueue_timed(int warm, int iters,
                     const std::function<void()>& tick = nullptr) {
    CUDA_CHECK(cudaSetDevice(dev));
    for (int i = 0; i < warm; ++i) op.run(stream);
    CUDA_CHECK(cudaEventRecord(e_beg, stream));
    for (int i = 0; i < iters; ++i) {
      op.run(stream);
      CUDA_CHECK(cudaEventRecord(ev[i], stream));
      if (tick && (i & 15) == 15) {
        tick();  // may switch the current device (multi-GPU pump)
        CUDA_CHECK(cudaSetDevice(dev));
      }
    }
    CUDA_CHECK(cudaEventRecord(e_end, stream));
  }

  Stats collect(int iters) const {
    std::vector<double> per(iters);
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_beg, ev[0]));
    per[0] = ms * 1000.0;
    for (int i = 1; i < iters; ++i) {
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev[i - 1], ev[i]));
      per[i] = ms * 1000.0;
    }
    return make_stats(per);
  }

  double flop() const { return 2.0 * s.m * s.n * s.k; }

  void destroy() {
    CUDA_CHECK(cudaSetDevice(dev));
    for (auto& e : ev) cudaEventDestroy(e);
    ev.clear();
    cudaEventDestroy(e_beg);
    cudaEventDestroy(e_end);
    cudaStreamDestroy(stream);
    op.destroy();
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
  }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int world = 0;
  CUDA_CHECK(cudaGetDeviceCount(&world));
  if (cfg.ndev > 0) world = std::min(world, cfg.ndev);
  if (cfg.gemm_dev >= world) {
    std::fprintf(stderr, "--gemm-dev %d out of range (world %d)\n",
                 cfg.gemm_dev, world);
    return 1;
  }
  std::vector<int> devs;
  for (int r = 0; r < world; ++r) devs.push_back(r);
  enable_peer_all(devs);

  if (cfg.patterns.empty()) {
    cfg.patterns = {"pull", "push", "allgather"};
    if (world >= 3) {
      cfg.patterns.push_back("bystander");
      cfg.patterns.push_back("engine-only");
    }
    cfg.patterns.push_back("local");
  }

  std::printf("=== exp1: CE traffic vs independent sm90 GEMM ===\n");
  std::printf("world %d, GEMM on GPU %d, fp16 CUTLASS "
              "(TmaWarpSpecializedCooperative 128x256x64)\n",
              world, cfg.gemm_dev);
  for (int r = 0; r < world; ++r) print_device_line(r);
  std::printf("[note] lock clocks (nvidia-smi -lgc) before trusting "
              "slowdowns.\n\n");

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv,
                   "pattern,m,n,k,msg_bytes,legs,alone_us,ovl_us,alone_p95,"
                   "ovl_p95,S_c,tflops_alone,tflops_ovl,comm_alone_gbps,"
                   "comm_ovl_gbps,gaps\n");
  }

  Buffers buf;
  buf.init(world, cfg.comm_buf);

  for (const Shape3& sh : cfg.sizes) {
    GemmRunner g;
    g.init(cfg.gemm_dev, sh);

    // calibrate iteration count, then measure the GEMM alone
    g.enqueue_timed(3, 5);
    CUDA_CHECK(cudaEventSynchronize(g.e_end));
    const double est_us = g.collect(5).mean;
    int iters = cfg.iters > 0
                    ? std::min(cfg.iters, kMaxIters)
                    : (int)std::max(5.0, std::min((double)kMaxIters,
                                                  cfg.window_ms * 1000.0 /
                                                      std::max(1.0, est_us)));
    g.enqueue_timed(3, iters);
    CUDA_CHECK(cudaEventSynchronize(g.e_end));
    const Stats alone = g.collect(iters);
    const double tflops_alone = g.flop() / (alone.mean * 1e-6) / 1e12;

    std::printf("--- GEMM %dx%dx%d fp16: alone %.1f us/iter (p95 %.1f), "
                "%.1f TFLOP/s, %d iters ---\n",
                sh.m, sh.n, sh.k, alone.mean, alone.p95, tflops_alone, iters);
    std::printf("%-11s %-6s | %11s %9s %6s %8s | %10s %10s | %s\n", "pattern",
                "msg", "ovl us/iter", "ovl p95", "S_c", "TFLOP/s",
                "alone GB/s", "ovl GB/s", "note");

    for (const auto& pat : cfg.patterns) {
      for (uint64_t msg : cfg.msgs) {
        std::vector<Leg> legs;
        if (!build_legs(pat, cfg.gemm_dev, world, buf, msg, legs)) {
          std::printf("%-11s : skipped (needs more GPUs)\n", pat.c_str());
          continue;
        }
        const CommCal cal = calibrate(legs, cfg.window_ms);

        Pump pump;
        pump.start(&legs, cal.chunk_msgs);
        const auto t0 = std::chrono::steady_clock::now();
        g.enqueue_timed(3, iters, [&] { pump.poll(); });
        CUDA_CHECK(cudaSetDevice(cfg.gemm_dev));
        while (cudaEventQuery(g.e_end) != cudaSuccess) {
          pump.poll();
          std::this_thread::yield();
          CUDA_CHECK(cudaSetDevice(cfg.gemm_dev));
        }
        cudaGetLastError();
        const double wall_s = wall_ms_since(t0) * 1e-3;
        pump.drain();

        const Stats ovl = g.collect(iters);
        const double s_c = alone.mean > 0 ? ovl.mean / alone.mean : 0;
        const double tflops_ovl = g.flop() / (ovl.mean * 1e-6) / 1e12;
        const double ovl_gbps =
            wall_s > 0 ? pump.completed_bytes / wall_s / 1e9 : 0;
        const char* note = pump.gaps > 0 ? "[!] CE went idle mid-window" : "";

        char msg_str[16];
        if (msg >= (1ull << 20))
          std::snprintf(msg_str, sizeof(msg_str), "%lluM",
                        (unsigned long long)(msg >> 20));
        else
          std::snprintf(msg_str, sizeof(msg_str), "%lluK",
                        (unsigned long long)(msg >> 10));

        std::printf("%-11s %-6s | %11.1f %9.1f %6.3f %8.1f | %10.2f %10.2f "
                    "| %s\n",
                    pat.c_str(), msg_str, ovl.mean, ovl.p95, s_c, tflops_ovl,
                    cal.gbps, ovl_gbps, note);
        std::fflush(stdout);

        if (csv)
          std::fprintf(csv,
                       "%s,%d,%d,%d,%llu,%zu,%.2f,%.2f,%.2f,%.2f,%.4f,%.1f,"
                       "%.1f,%.3f,%.3f,%d\n",
                       pat.c_str(), sh.m, sh.n, sh.k, (unsigned long long)msg,
                       legs.size(), alone.mean, ovl.mean, alone.p95, ovl.p95,
                       s_c, tflops_alone, tflops_ovl, cal.gbps, ovl_gbps,
                       pump.gaps);
        destroy_legs(legs);
      }
    }

    // Paired baseline: re-measure `alone` after all patterns ran. Every S_c
    // above is a ratio against the *opening* baseline, so environmental drift
    // (another job landing on the box, clocks sagging) silently pollutes all
    // of them; this makes it visible and flags the whole block.
    g.enqueue_timed(3, iters);
    CUDA_CHECK(cudaEventSynchronize(g.e_end));
    const Stats alone2 = g.collect(iters);
    const double drift =
        alone.mean > 0 ? alone2.mean / alone.mean - 1.0 : 0.0;
    std::printf("baseline recheck: alone %.1f -> %.1f us/iter (drift %+.1f%%)%s\n",
                alone.mean, alone2.mean, drift * 100.0,
                std::fabs(drift) > 0.05
                    ? "  [!] drift >5%: environment not quiet, every S_c "
                      "above is unreliable"
                    : "");
    std::printf("\n");
    g.destroy();
  }

  std::printf(
      "S_c = GEMM iter time overlapped / alone. Compare patterns:\n"
      "  bystander ~1.00 and pull/push > 1.00  => local HBM contention\n"
      "  engine-only ~1.00                     => CE engine itself is free\n"
      "  local >= pull/push                    => per-byte HBM cost ordering\n");

  buf.destroy();
  if (csv) std::fclose(csv);
  return 0;
}
