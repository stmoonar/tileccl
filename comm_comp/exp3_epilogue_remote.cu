// exp3_epilogue_remote.cu
//
// Experiment 3: does writing D to REMOTE memory in the GEMM epilogue cost
// compute performance?
//
// This is the one overlap scheme where communication and computation share
// SM resources: the epilogue's stores go over NVLink instead of to local
// HBM. Notably flux avoids this on sm90 -- its GEMM+RS epilogue TMA-stores
// to LOCAL memory and separate producer warps *pull* peer tiles -- while its
// sm80 path does write straight to peer pointers from the epilogue. Here we
// measure directly what that choice is worth on Hopper.
//
// Same CUTLASS sm90 fp16 GEMM, only the D pointer (and epilogue flavor)
// changes. Modes:
//   local          D on the GEMM device                      (baseline)
//   remote         D entirely on a peer rank (--peer), every epilogue store
//                  crosses NVLink
//   scatter-local  D row-partitioned into `world` segment GEMMs, all
//                  writing locally (control: isolates the segmentation cost
//                  that scatter needs)
//   scatter        the GEMM+RS write pattern: segment r's rows are written
//                  into rank r's buffer -- 1/world of the writes stay
//                  local, the rest cross NVLink to world-1 different peers
//
// Epilogue flavors (--epi):
//   tma     Sm90 TMA-store epilogue (smem staged; what flux sm90 uses for
//           its local stores). The TMA descriptor is built over the peer
//           UVA pointer.
//   nosmem  DefaultEpilogue, direct register->global stores in accumulator
//           layout (no smem restage; narrow transactions, NOT 128-bit
//           vectorized) -- morally flux's sm80 remote-write path on sm90
//           hardware. Its absolute TFLOP/s trails tma; the meaningful read
//           is remote-vs-local WITHIN each epilogue flavor.
//
// Small-K shapes are included by default: the lower the arithmetic
// intensity, the larger the epilogue's share of the kernel and the more a
// remote-write penalty shows. Per row we report TFLOP/s, the slowdown vs
// the same-epilogue local baseline, and the implied D write bandwidth.
//
// --verify checks D bitwise against the local-mode result (same kernel
// config => identical arithmetic; only the destination differs).
//
// See comm_comp/README.md for build & run lines.

#include "common.cuh"
#include "gemm_sm90.cuh"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct Config {
  std::vector<Shape3> sizes;
  std::vector<std::string> modes;  // local, remote, scatter-local, scatter
  std::vector<std::string> epis;   // tma, nosmem
  int peer = -1;                   // -1 = (gemm_dev + 1) % world
  double window_ms = 200;
  int iters = 0;
  int gemm_dev = 0;
  int ndev = 0;
  bool verify = false;
  std::string csv;
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --sizes LIST    GEMM shapes, N or MxNxK\n"
      "                  (default 8192,8192x8192x2048,8192x8192x512)\n"
      "  --modes LIST    subset of local,remote,scatter-local,scatter\n"
      "                  (default: all)\n"
      "  --epi LIST      subset of tma,nosmem (default: both)\n"
      "  --peer N        target rank for 'remote' (default gemm-dev+1)\n"
      "  --window-ms MS  target timing window (default 200)\n"
      "  --iters N       iterations, 0 = auto\n"
      "  --gemm-dev N    device running the GEMM (default 0)\n"
      "  --ndev N        use only the first N GPUs (default: all)\n"
      "  --verify        compare D bitwise against the local-mode result\n"
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
    } else if (a == "--modes") c.modes = split_csv(next());
    else if (a == "--epi") c.epis = split_csv(next());
    else if (a == "--peer") c.peer = std::atoi(next().c_str());
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
  if (c.sizes.empty()) {
    c.sizes.push_back({8192, 8192, 8192});
    c.sizes.push_back({8192, 8192, 2048});
    c.sizes.push_back({8192, 8192, 512});
  }
  if (c.modes.empty()) c.modes = {"local", "remote", "scatter-local", "scatter"};
  for (const auto& m : c.modes)
    if (m != "local" && m != "remote" && m != "scatter-local" &&
        m != "scatter") {
      std::fprintf(stderr, "unknown mode '%s'\n", m.c_str());
      std::exit(1);
    }
  if (c.epis.empty()) c.epis = {"tma", "nosmem"};
  for (const auto& e : c.epis)
    if (e != "tma" && e != "nosmem") {
      std::fprintf(stderr, "unknown epilogue '%s'\n", e.c_str());
      std::exit(1);
    }
  return c;
}

// ---------------------------------------------------------------------------
// one measurement = a list of pre-initialized segment launches
// ---------------------------------------------------------------------------

constexpr int kMaxIters = 2048;

// Type-erased so tma and nosmem ops can share the timing loop.
struct Plan {
  std::vector<std::function<void(cudaStream_t)>> launches;
  std::vector<std::function<void()>> destructors;

  void destroy() {
    for (auto& d : destructors) d();
    launches.clear();
    destructors.clear();
  }
};

template <class Cfg>
static void add_op(Plan& plan, int M, int N, int K, const void* A,
                   const void* B, const void* C, void* D, int dev) {
  auto* op = new cc::GemmOp<Cfg>();
  op->init(M, N, K, A, B, C, D, 1.0f, 0.0f, dev);
  plan.launches.push_back([op](cudaStream_t s) { op->run(s); });
  plan.destructors.push_back([op] {
    op->destroy();
    delete op;
  });
}

struct Bench {
  int dev = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t e_beg = nullptr;
  std::vector<cudaEvent_t> ev;

  void init(int dev_) {
    dev = dev_;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&e_beg));
    ev.resize(kMaxIters);
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  }

  Stats time(Plan& plan, int warm, int iters) {
    CUDA_CHECK(cudaSetDevice(dev));
    for (int i = 0; i < warm; ++i)
      for (auto& l : plan.launches) l(stream);
    CUDA_CHECK(cudaEventRecord(e_beg, stream));
    for (int i = 0; i < iters; ++i) {
      for (auto& l : plan.launches) l(stream);
      CUDA_CHECK(cudaEventRecord(ev[i], stream));
    }
    CUDA_CHECK(cudaEventSynchronize(ev[iters - 1]));
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

  void destroy() {
    CUDA_CHECK(cudaSetDevice(dev));
    for (auto& e : ev) cudaEventDestroy(e);
    ev.clear();
    cudaEventDestroy(e_beg);
    cudaStreamDestroy(stream);
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
  if (world < 2) {
    std::fprintf(stderr, "exp3 needs at least 2 GPUs\n");
    return 1;
  }
  const int g = cfg.gemm_dev;
  if (g >= world) {
    std::fprintf(stderr, "--gemm-dev %d out of range (world %d)\n", g, world);
    return 1;
  }
  const int peer = cfg.peer >= 0 ? cfg.peer : (g + 1) % world;
  if (peer >= world || peer == g) {
    std::fprintf(stderr, "--peer %d invalid (world %d, gemm-dev %d)\n", peer,
                 world, g);
    return 1;
  }
  std::vector<int> devs;
  for (int r = 0; r < world; ++r) devs.push_back(r);
  enable_peer_all(devs);

  std::printf("=== exp3: epilogue remote write vs local write (sm90) ===\n");
  std::printf("world %d, GEMM on GPU %d, remote peer GPU %d\n", world, g, peer);
  for (int r = 0; r < world; ++r) print_device_line(r);
  std::printf("[note] flux sm90 GEMM+RS deliberately does NOT remote-write "
              "in the epilogue\n(it TMA-stores locally and pulls); its sm80 "
              "path does. This measures that choice.\n\n");

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv,
                   "m,n,k,epi,mode,peer,iter_us,iter_p95,slowdown,tflops,"
                   "d_write_gbps,verify\n");
  }

  Bench bench;
  bench.init(g);

  for (const Shape3& sh : cfg.sizes) {
    if ((sh.k % cc::kAlign) || (sh.n % cc::kAlign) || (sh.m % world)) {
      std::printf("skip %dx%dx%d: need N,K %% %d == 0 and M %% world == 0\n",
                  sh.m, sh.n, sh.k, cc::kAlign);
      continue;
    }
    const int m_seg = sh.m / world;
    const double flop = 2.0 * sh.m * sh.n * sh.k;
    const size_t d_bytes = (size_t)sh.m * sh.n * 2;

    // operands on g; a full-size D buffer on every rank
    CUDA_CHECK(cudaSetDevice(g));
    void *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, (size_t)sh.m * sh.k * 2));
    CUDA_CHECK(cudaMalloc(&B, (size_t)sh.k * sh.n * 2));
    CUDA_CHECK(cudaMalloc(&C, d_bytes));
    fill_half(A, (size_t)sh.m * sh.k, 1);
    fill_half(B, (size_t)sh.k * sh.n, 2);
    fill_half(C, (size_t)sh.m * sh.n, 3);
    std::vector<cc::ElementC*> Dbuf(world, nullptr);
    for (int r = 0; r < world; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMalloc(&Dbuf[r], d_bytes));
      CUDA_CHECK(cudaMemset(Dbuf[r], 0, d_bytes));
    }
    CUDA_CHECK(cudaSetDevice(g));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("--- %dx%dx%d fp16 (D %.0f MiB, %.1f flop/byte-of-D) ---\n",
                sh.m, sh.n, sh.k, (double)d_bytes / (1 << 20),
                flop / (double)d_bytes);
    std::printf("%-7s %-13s | %9s %9s %8s %8s | %9s | %s\n", "epi", "mode",
                "us/iter", "p95", "slowdn", "TFLOP/s", "D GB/s", "ver");

    // rows [r0, r0+mrows) of a [M,N] fp16 row-major buffer
    auto rows = [&](cc::ElementC* base, int r0) -> void* {
      return base + (size_t)r0 * sh.n;
    };

    for (const auto& epi : cfg.epis) {
      const bool tma = epi == "tma";
      double local_us = 0;  // same-epi baseline for the slowdown column
      std::vector<uint8_t> d_ref;  // local-mode D, for --verify

      for (const auto& mode : cfg.modes) {
        // build the launch plan
        Plan plan;
        // (dst_rank, dst_row_offset, src_row_offset, rows) per segment
        struct SegDesc { int dst; size_t drow, srow; int mrows; };
        std::vector<SegDesc> descs;
        if (mode == "local") {
          descs.push_back({g, 0, 0, sh.m});
        } else if (mode == "remote") {
          descs.push_back({peer, 0, 0, sh.m});
        } else {  // scatter / scatter-local
          for (int r = 0; r < world; ++r) {
            // scatter: GEMM+RS placement -- segment r lands in rank r's
            // buffer, in the slot reserved for source rank g.
            // scatter-local: same segmentation, but D stays local at its
            // natural row offset.
            const int dst = (mode == "scatter") ? r : g;
            const size_t drow =
                (mode == "scatter") ? (size_t)g * m_seg : (size_t)r * m_seg;
            descs.push_back({dst, drow, (size_t)r * m_seg, m_seg});
          }
        }
        for (const auto& d : descs) {
          const void* a = (const cc::ElementAB*)A + d.srow * sh.k;
          const void* c = (const cc::ElementC*)C + d.srow * sh.n;
          void* dp = rows(Dbuf[d.dst], (int)d.drow);
          if (tma)
            add_op<cc::GemmCfgTma>(plan, d.mrows, sh.n, sh.k, a, B, c, dp, g);
          else
            add_op<cc::GemmCfgNoSmem>(plan, d.mrows, sh.n, sh.k, a, B, c, dp,
                                      g);
        }

        // time it
        Plan* p = &plan;
        Stats st;
        {
          // short probe to calibrate the iteration count for this mode
          bench.time(*p, 2, 3);
          const Stats probe = bench.time(*p, 0, 3);
          int iters =
              cfg.iters > 0
                  ? std::min(cfg.iters, kMaxIters)
                  : (int)std::max(5.0, std::min((double)kMaxIters,
                                                cfg.window_ms * 1000.0 /
                                                    std::max(1.0, probe.mean)));
          st = bench.time(*p, 2, iters);
        }
        if (mode == "local") local_us = st.mean;
        const double slowdown = local_us > 0 ? st.mean / local_us : 0;
        const double tflops = flop / (st.mean * 1e-6) / 1e12;
        const double d_gbps = d_bytes / (st.mean * 1e-6) / 1e9;

        // verify against the local-mode result
        const char* ver = "-";
        if (cfg.verify) {
          if (mode == "local") {
            d_ref.resize(d_bytes);
            CUDA_CHECK(cudaMemcpy(d_ref.data(), Dbuf[g], d_bytes,
                                  cudaMemcpyDeviceToHost));
            ver = "ref";
          } else if (d_ref.size() == d_bytes) {
            bool ok = true;
            std::vector<uint8_t> h(d_bytes);
            if (mode == "remote") {
              CUDA_CHECK(cudaMemcpy(h.data(), Dbuf[peer], d_bytes,
                                    cudaMemcpyDeviceToHost));
              ok = std::memcmp(h.data(), d_ref.data(), d_bytes) == 0;
            } else {
              const size_t seg_bytes = (size_t)m_seg * sh.n * 2;
              for (const auto& d : descs) {
                CUDA_CHECK(cudaMemcpy(h.data(),
                                      rows(Dbuf[d.dst], (int)d.drow),
                                      seg_bytes, cudaMemcpyDeviceToHost));
                if (std::memcmp(h.data(), d_ref.data() + d.srow * sh.n * 2,
                                seg_bytes) != 0) {
                  ok = false;
                  break;
                }
              }
            }
            ver = ok ? "ok" : "FAIL";
          }
        }

        std::printf("%-7s %-13s | %9.1f %9.1f %8.3f %8.1f | %9.1f | %s\n",
                    epi.c_str(), mode.c_str(), st.mean, st.p95, slowdown,
                    tflops, d_gbps, ver);
        std::fflush(stdout);

        if (csv)
          std::fprintf(csv, "%d,%d,%d,%s,%s,%d,%.2f,%.2f,%.4f,%.1f,%.2f,%s\n",
                       sh.m, sh.n, sh.k, epi.c_str(), mode.c_str(),
                       mode == "remote" ? peer : -1, st.mean, st.p95, slowdown,
                       tflops, d_gbps, ver);
        plan.destroy();
      }
    }
    std::printf("\n");

    CUDA_CHECK(cudaSetDevice(g));
    cudaFree(A); cudaFree(B); cudaFree(C);
    for (int r = 0; r < world; ++r) {
      cudaSetDevice(r);
      cudaFree(Dbuf[r]);
    }
    CUDA_CHECK(cudaSetDevice(g));
  }

  std::printf(
      "slowdn = iter time / same-epilogue local baseline.\n"
      "remote vs local      : full NVLink write pressure from the epilogue\n"
      "scatter vs scatter-local : the realistic GEMM+RS pattern (1/world "
      "local)\n"
      "tma vs nosmem        : is the penalty specific to TMA descriptors "
      "over\n                       peer memory, or generic to remote "
      "stores?\n"
      "If 'remote' is close to 1.0, flux's sm90 pull-based RS is leaving\n"
      "nothing on the table latency-wise; if it is >> 1.0, the pull design "
      "is\njustified on Hopper NVLink.\n");

  bench.destroy();
  if (csv) std::fclose(csv);
  return 0;
}
