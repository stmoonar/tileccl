// ce_gemm_overlap.cu
//
// Does Copy-Engine traffic slow down an *independent* GEMM?
//
// interference_matrix answers this with synthetic probes (rate counters); this
// experiment answers it the way a framework user would see it: a real cuBLAS
// GEMM on stream G, CE copies on stream C, both on the same device, GEMM data
// completely unrelated to the transfer. We measure per-iteration GEMM latency
// with CUDA events, alone vs overlapped:
//
//     S_c = t_overlap / t_alone        (GEMM slowdown under CE traffic)
//     S_m = BW_alone  / BW_overlap     (CE slowdown under GEMM, phase B)
//
// Methodology:
//   * Phase A (primary): the host keeps stream C saturated with copies via a
//     ring of chunk-sized bursts (re-enqueued as their events complete), so CE
//     is busy for the *entire* GEMM timing window regardless of how much the
//     copies themselves slow down. If stream C is ever observed idle while the
//     GEMM is still running, the row is flagged -- the overlap was not fully
//     covered and S_c is a lower bound.
//   * Phase B: symmetric -- a ring of GEMM launches keeps the SMs busy while a
//     calibrated CE burst is timed with events.
//   * Copy offsets cycle through a large buffer (--comm-buf) so CE traffic
//     streams HBM instead of hitting L2 on a hot line.
//
// Comm kinds (controls that separate "HBM bandwidth contention" from "CE
// engine per se"):
//   p2p   cudaMemcpyPeerAsync between --src and --dst (push / pull / bidir;
//         push reads the GEMM device's HBM, pull writes it, bidir does both)
//   d2d   local device-to-device copy on the GEMM device (CE reads AND writes
//         local HBM -- the heaviest local-bandwidth load per byte moved)
//   h2d / d2h  pinned-host transfers over PCIe (CE active, but almost no local
//         HBM bandwidth consumed -- if GEMM only slows under p2p/d2d and not
//         here, the interference is memory-system contention, not the engine)
//
// Caveats: lock clocks first (nvidia-smi -lgc) -- DVFS masquerades as
// contention. cuBLAS may pick different kernels per shape; S_c compares the
// same shape against itself so that cancels out.
//
// Build:  make            (links -lcublas, see tests/Makefile)
// Run:    ./ce_gemm_overlap
//         ./ce_gemm_overlap --sizes 4096,8192,8192x8192x128 --comms p2p,d2d,h2d
//         ./ce_gemm_overlap --dir bidir --msg 64M --gemm-dev dst

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t _st = (expr);                                              \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[cuBLAS] %s:%d: %s -> status %d\n", __FILE__,     \
                   __LINE__, #expr, (int)_st);                                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// config / CLI
// ---------------------------------------------------------------------------

struct Shape {
  int m = 0, n = 0, k = 0;
};

struct Config {
  std::vector<Shape> sizes;              // default set in main
  std::vector<std::string> comms;        // default: p2p if peer ok, else d2d
  std::string dir = "push";              // push | pull | bidir (p2p only)
  uint64_t msg = 16ull << 20;            // CE message size
  uint64_t comm_buf = 256ull << 20;      // cycling buffer per side
  double window_ms = 300;                // target timing window
  std::string gemm_dev = "src";          // src | dst
  int src_dev = 0;
  int dst_dev = 1;
  std::string dtype = "fp16";            // fp16 | tf32 | fp32
  int iters = 0;                         // 0 = auto from window
  bool do_sm = true;                     // phase B (S_m)
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

// "4096" -> 4096^3, "8192x8192x128" -> m x n x k
static Shape parse_shape(const std::string& tok) {
  Shape s;
  const size_t x1 = tok.find_first_of("xX");
  if (x1 == std::string::npos) {
    s.m = s.n = s.k = std::atoi(tok.c_str());
  } else {
    const size_t x2 = tok.find_first_of("xX", x1 + 1);
    if (x2 == std::string::npos) {
      std::fprintf(stderr, "bad shape '%s' (want N or MxNxK)\n", tok.c_str());
      std::exit(1);
    }
    s.m = std::atoi(tok.substr(0, x1).c_str());
    s.n = std::atoi(tok.substr(x1 + 1, x2 - x1 - 1).c_str());
    s.k = std::atoi(tok.substr(x2 + 1).c_str());
  }
  if (s.m <= 0 || s.n <= 0 || s.k <= 0) {
    std::fprintf(stderr, "bad shape '%s'\n", tok.c_str());
    std::exit(1);
  }
  return s;
}

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --sizes LIST      GEMM shapes: N (cube) or MxNxK, comma separated\n"
      "                    (default 1024,2048,4096,8192)\n"
      "  --comms LIST      subset of p2p,d2d,h2d,d2h\n"
      "                    (default: p2p if peer access works, else d2d)\n"
      "  --dir D           p2p direction: push (src->dst) | pull (dst->src) |\n"
      "                    bidir. With the GEMM on src (default), push reads\n"
      "                    its HBM and pull writes it.\n"
      "  --msg BYTES       CE message size (default 16M)\n"
      "  --comm-buf BYTES  cycling copy buffer per side (default 256M)\n"
      "  --window-ms MS    target timing window (default 300)\n"
      "  --iters N         GEMM iterations, 0 = auto from window (max 4096)\n"
      "  --gemm-dev D      src | dst -- where the GEMM runs (default src)\n"
      "  --src N / --dst N GPU indices (default 0 / 1)\n"
      "  --dtype T         fp16 | tf32 | fp32 (default fp16)\n"
      "  --no-sm           skip phase B (CE slowdown under GEMM)\n"
      "  --csv PATH        append machine-readable rows to PATH\n",
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
      c.sizes.clear();
      for (const auto& t : split_csv(next())) c.sizes.push_back(parse_shape(t));
    } else if (a == "--comms") c.comms = split_csv(next());
    else if (a == "--dir") c.dir = next();
    else if (a == "--msg") c.msg = parse_bytes(next());
    else if (a == "--comm-buf") c.comm_buf = parse_bytes(next());
    else if (a == "--window-ms") c.window_ms = std::atof(next().c_str());
    else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--gemm-dev") c.gemm_dev = next();
    else if (a == "--src") c.src_dev = std::atoi(next().c_str());
    else if (a == "--dst") c.dst_dev = std::atoi(next().c_str());
    else if (a == "--dtype") c.dtype = next();
    else if (a == "--no-sm") c.do_sm = false;
    else if (a == "--csv") c.csv = next();
    else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (c.dir != "push" && c.dir != "pull" && c.dir != "bidir") {
    std::fprintf(stderr, "--dir must be push, pull or bidir\n");
    std::exit(1);
  }
  for (const auto& k : c.comms)
    if (k != "p2p" && k != "d2d" && k != "h2d" && k != "d2h") {
      std::fprintf(stderr, "unknown comm kind '%s'\n", k.c_str());
      std::exit(1);
    }
  if (c.gemm_dev != "src" && c.gemm_dev != "dst") {
    std::fprintf(stderr, "--gemm-dev must be src or dst\n");
    std::exit(1);
  }
  if (c.dtype != "fp16" && c.dtype != "tf32" && c.dtype != "fp32") {
    std::fprintf(stderr, "--dtype must be fp16, tf32 or fp32\n");
    std::exit(1);
  }
  if (c.msg < 4096) {
    std::fprintf(stderr, "--msg too small (>= 4K)\n");
    std::exit(1);
  }
  return c;
}

// ---------------------------------------------------------------------------
// data init
// ---------------------------------------------------------------------------

__global__ void fill_fp16_kernel(__half* p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += step) {
    const uint32_t h = (uint32_t)i * 2654435761u;
    p[i] = __float2half(((h >> 16) & 0xffffu) * (1.0f / 65536.0f) - 0.5f);
  }
}

__global__ void fill_fp32_kernel(float* p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += step) {
    const uint32_t h = (uint32_t)i * 2654435761u;
    p[i] = ((h >> 16) & 0xffffu) * (1.0f / 65536.0f) - 0.5f;
  }
}

// ---------------------------------------------------------------------------
// per-iteration stats
// ---------------------------------------------------------------------------

struct Stats {
  double mean = 0, p50 = 0, p95 = 0, mx = 0;
};

static Stats make_stats(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  std::sort(v.begin(), v.end());
  s.p50 = v[v.size() / 2];
  s.p95 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.95))];
  s.mx = v.back();
  return s;
}

// ---------------------------------------------------------------------------
// GEMM runner (cuBLAS on its own stream, per-iteration events)
// ---------------------------------------------------------------------------

constexpr int kMaxIters = 4096;

struct Gemm {
  int dev = -1;
  Shape s;
  cudaDataType_t abtype = CUDA_R_16F, ctype = CUDA_R_16F;
  cublasComputeType_t comp = CUBLAS_COMPUTE_32F;
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  void *A = nullptr, *B = nullptr, *C = nullptr;
  cudaEvent_t e_beg = nullptr, e_end = nullptr;
  std::vector<cudaEvent_t> ev;

  void init(int dev_, Shape s_, const std::string& dtype) {
    dev = dev_;
    s = s_;
    CUDA_CHECK(cudaSetDevice(dev));
    const bool half = dtype == "fp16";
    abtype = ctype = half ? CUDA_R_16F : CUDA_R_32F;
    comp = dtype == "tf32" ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    const size_t es = half ? 2 : 4;
    CUDA_CHECK(cudaMalloc(&A, (size_t)s.m * s.k * es));
    CUDA_CHECK(cudaMalloc(&B, (size_t)s.k * s.n * es));
    CUDA_CHECK(cudaMalloc(&C, (size_t)s.m * s.n * es));
    if (half) {
      fill_fp16_kernel<<<256, 256>>>((__half*)A, (size_t)s.m * s.k);
      fill_fp16_kernel<<<256, 256>>>((__half*)B, (size_t)s.k * s.n);
    } else {
      fill_fp32_kernel<<<256, 256>>>((float*)A, (size_t)s.m * s.k);
      fill_fp32_kernel<<<256, 256>>>((float*)B, (size_t)s.k * s.n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUDA_CHECK(cudaEventCreate(&e_beg));
    CUDA_CHECK(cudaEventCreate(&e_end));
    ev.resize(kMaxIters);
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  }

  void enqueue_once() {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.m, s.n, s.k,
                              &alpha, A, abtype, s.m, B, abtype, s.k, &beta, C,
                              ctype, s.m, comp, CUBLAS_GEMM_DEFAULT));
  }

  // Enqueue warmup + a timed run; does NOT synchronize -- the caller decides
  // whether to just wait (alone) or to pump the comm stream meanwhile.
  // `tick` (if set) is called every 64 enqueues: submitting thousands of
  // cuBLAS calls takes tens of ms of host time, longer than the comm pump's
  // ring depth -- without ticking the pump here, CE drains at the start of
  // the window for small shapes (observed as the '[!] CE went idle' flag).
  void enqueue_timed(int warm, int iters,
                     const std::function<void()>& tick = nullptr) {
    CUDA_CHECK(cudaSetDevice(dev));
    for (int i = 0; i < warm; ++i) enqueue_once();
    CUDA_CHECK(cudaEventRecord(e_beg, stream));
    for (int i = 0; i < iters; ++i) {
      enqueue_once();
      CUDA_CHECK(cudaEventRecord(ev[i], stream));
      if (tick && (i & 63) == 63) tick();
    }
    CUDA_CHECK(cudaEventRecord(e_end, stream));
  }

  // After e_end completed: per-iteration times in us and the total window.
  Stats collect(int iters, double* total_us) const {
    std::vector<double> per(iters);
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_beg, ev[0]));
    per[0] = ms * 1000.0;
    for (int i = 1; i < iters; ++i) {
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev[i - 1], ev[i]));
      per[i] = ms * 1000.0;
    }
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_beg, e_end));
    if (total_us) *total_us = ms * 1000.0;
    return make_stats(per);
  }

  double flop() const { return 2.0 * s.m * s.n * s.k; }

  void destroy() {
    CUDA_CHECK(cudaSetDevice(dev));
    for (auto& e : ev) cudaEventDestroy(e);
    ev.clear();
    cudaEventDestroy(e_beg);
    cudaEventDestroy(e_end);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(A); cudaFree(B); cudaFree(C);
  }
};

// ---------------------------------------------------------------------------
// CE communication: buffers + streams + one-message enqueue
// ---------------------------------------------------------------------------

struct Comm {
  std::string kind;   // p2p | d2d | h2d | d2h
  std::string dir;    // push | pull | bidir (p2p only)
  int gemm_dev = 0, src_dev = 0, dst_dev = 1;
  uint64_t msg = 0;
  uint32_t region_msgs = 1;  // messages per stream region (offset cycling)
  int nstreams = 1;
  uint8_t* d_a = nullptr;    // src-side (p2p) / local buffer
  uint8_t* d_b = nullptr;    // dst-side (p2p) / second local buffer (d2d)
  uint8_t* h_buf = nullptr;  // pinned host buffer (h2d/d2h)
  cudaStream_t st[2] = {nullptr, nullptr};
  uint64_t next[2] = {0, 0};

  void init(const std::string& kind_, const Config& cfg, int gemm_dev_) {
    kind = kind_;
    dir = cfg.dir;
    gemm_dev = gemm_dev_;
    src_dev = cfg.src_dev;
    dst_dev = cfg.dst_dev;
    msg = cfg.msg;
    nstreams = (kind == "p2p" && dir == "bidir") ? 2 : 1;
    region_msgs =
        (uint32_t)std::max<uint64_t>(1, cfg.comm_buf / msg / nstreams);
    const size_t bytes = (size_t)region_msgs * nstreams * msg;
    if (kind == "p2p") {
      CUDA_CHECK(cudaSetDevice(src_dev));
      CUDA_CHECK(cudaMalloc(&d_a, bytes));
      CUDA_CHECK(cudaMemset(d_a, 1, bytes));
      CUDA_CHECK(cudaSetDevice(dst_dev));
      CUDA_CHECK(cudaMalloc(&d_b, bytes));
      CUDA_CHECK(cudaMemset(d_b, 2, bytes));
    } else if (kind == "d2d") {
      CUDA_CHECK(cudaSetDevice(gemm_dev));
      CUDA_CHECK(cudaMalloc(&d_a, bytes));
      CUDA_CHECK(cudaMemset(d_a, 1, bytes));
      CUDA_CHECK(cudaMalloc(&d_b, bytes));
      CUDA_CHECK(cudaMemset(d_b, 2, bytes));
    } else {  // h2d / d2h
      CUDA_CHECK(cudaSetDevice(gemm_dev));
      CUDA_CHECK(cudaMalloc(&d_a, bytes));
      CUDA_CHECK(cudaMemset(d_a, 1, bytes));
      CUDA_CHECK(cudaMallocHost(&h_buf, bytes));
      std::memset(h_buf, 3, bytes);
    }
    CUDA_CHECK(cudaSetDevice(gemm_dev));
    for (int i = 0; i < nstreams; ++i)
      CUDA_CHECK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
  }

  void enqueue_msg(int si) {
    const uint64_t off =
        ((next[si]++ % region_msgs) + (uint64_t)si * region_msgs) * msg;
    if (kind == "p2p") {
      const bool push = (si == 0) ? (dir != "pull") : false;  // si 1 = pull leg
      if (push)
        CUDA_CHECK(cudaMemcpyPeerAsync(d_b + off, dst_dev, d_a + off, src_dev,
                                       msg, st[si]));
      else
        CUDA_CHECK(cudaMemcpyPeerAsync(d_a + off, src_dev, d_b + off, dst_dev,
                                       msg, st[si]));
    } else if (kind == "d2d") {
      CUDA_CHECK(cudaMemcpyAsync(d_b + off, d_a + off, msg,
                                 cudaMemcpyDeviceToDevice, st[si]));
    } else if (kind == "h2d") {
      CUDA_CHECK(cudaMemcpyAsync(d_a + off, h_buf + off, msg,
                                 cudaMemcpyHostToDevice, st[si]));
    } else {  // d2h
      CUDA_CHECK(cudaMemcpyAsync(h_buf + off, d_a + off, msg,
                                 cudaMemcpyDeviceToHost, st[si]));
    }
  }

  void enqueue_msgs(int si, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) enqueue_msg(si);
  }

  void sync() {
    for (int i = 0; i < nstreams; ++i) CUDA_CHECK(cudaStreamSynchronize(st[i]));
  }

  void destroy() {
    for (int i = 0; i < nstreams; ++i)
      if (st[i]) cudaStreamDestroy(st[i]);
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (h_buf) cudaFreeHost(h_buf);
    *this = Comm{};
  }
};

static void sleep_us(long long us) {
  std::this_thread::sleep_for(std::chrono::microseconds(us));
}

// Calibrated comm-alone measurement. Returns GB/s and fills the burst size
// (messages per stream that fill ~window_ms) and the pump chunk (~10 ms).
struct CommCal {
  double gbps = 0;
  uint32_t burst_msgs = 0;   // per stream
  uint32_t chunk_msgs = 0;   // per stream, for the pump
};

static double comm_burst_ms(Comm& c, uint32_t n_per_stream) {
  CUDA_CHECK(cudaSetDevice(c.gemm_dev));
  if (c.nstreams == 1) {
    cudaEvent_t b, e;
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(b, c.st[0]));
    c.enqueue_msgs(0, n_per_stream);
    CUDA_CHECK(cudaEventRecord(e, c.st[0]));
    CUDA_CHECK(cudaStreamSynchronize(c.st[0]));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, b, e));
    cudaEventDestroy(b);
    cudaEventDestroy(e);
    return (double)ms;
  }
  // bidir: host wall around both streams (events on different streams would
  // need a common reference; the window is long enough that wall is fine)
  const auto t0 = std::chrono::steady_clock::now();
  for (int si = 0; si < c.nstreams; ++si) c.enqueue_msgs(si, n_per_stream);
  c.sync();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static CommCal comm_calibrate(Comm& c, double window_ms) {
  CommCal cal;
  comm_burst_ms(c, 4);  // warm the path (first p2p copy pays setup)
  const double warm_ms = comm_burst_ms(c, 16);
  const double per_msg_ms = std::max(1e-6, warm_ms / 16.0);
  cal.burst_msgs = (uint32_t)std::max(
      16.0, std::min(200000.0, window_ms / per_msg_ms));
  cal.chunk_msgs =
      (uint32_t)std::max(1.0, std::min(50000.0, 10.0 / per_msg_ms));
  const double ms = comm_burst_ms(c, cal.burst_msgs);
  cal.gbps = (double)cal.burst_msgs * c.nstreams * c.msg / (ms * 1e-3) / 1e9;
  return cal;
}

// ---------------------------------------------------------------------------
// pumps: keep one side saturated while the other side is being timed
// ---------------------------------------------------------------------------

// Ring of comm chunks; re-enqueued as their events complete. Detects gaps
// (stream observed fully idle) which would invalidate the overlap window.
struct CommPump {
  static constexpr int kRing = 4;
  Comm* c = nullptr;
  uint32_t chunk_msgs = 0;
  cudaEvent_t ev[2][kRing] = {};
  uint64_t completed_bytes = 0;
  int gaps = 0;

  void start(Comm* c_, uint32_t chunk_msgs_) {
    c = c_;
    chunk_msgs = chunk_msgs_;
    completed_bytes = 0;
    gaps = 0;
    CUDA_CHECK(cudaSetDevice(c->gemm_dev));
    for (int si = 0; si < c->nstreams; ++si)
      for (int r = 0; r < kRing; ++r) {
        CUDA_CHECK(cudaEventCreate(&ev[si][r]));
        c->enqueue_msgs(si, chunk_msgs);
        CUDA_CHECK(cudaEventRecord(ev[si][r], c->st[si]));
      }
  }

  void poll() {
    for (int si = 0; si < c->nstreams; ++si) {
      if (cudaStreamQuery(c->st[si]) == cudaSuccess) ++gaps;
      for (int r = 0; r < kRing; ++r) {
        if (cudaEventQuery(ev[si][r]) == cudaSuccess) {
          completed_bytes += (uint64_t)chunk_msgs * c->msg;
          c->enqueue_msgs(si, chunk_msgs);
          CUDA_CHECK(cudaEventRecord(ev[si][r], c->st[si]));
        }
      }
    }
    cudaGetLastError();  // clear cudaErrorNotReady from the queries
  }

  void drain() {
    c->sync();
    for (int si = 0; si < c->nstreams; ++si)
      for (int r = 0; r < kRing; ++r)
        if (ev[si][r]) { cudaEventDestroy(ev[si][r]); ev[si][r] = nullptr; }
  }
};

struct GemmPump;
static double comm_burst_ms_pumped(Comm& c, uint32_t n_per_stream,
                                   GemmPump* gp);

// Ring of GEMM batches; keeps the SMs busy while a comm burst is timed.
struct GemmPump {
  static constexpr int kRing = 4;
  Gemm* g = nullptr;
  uint32_t batch = 1;  // GEMMs per ring slot (~5 ms of work)
  cudaEvent_t ev[kRing] = {};
  int gaps = 0;

  void start(Gemm* g_, double alone_us_per_iter) {
    g = g_;
    batch = (uint32_t)std::max(1.0, 5000.0 / std::max(1.0, alone_us_per_iter));
    gaps = 0;
    CUDA_CHECK(cudaSetDevice(g->dev));
    for (int r = 0; r < kRing; ++r) {
      CUDA_CHECK(cudaEventCreate(&ev[r]));
      for (uint32_t i = 0; i < batch; ++i) g->enqueue_once();
      CUDA_CHECK(cudaEventRecord(ev[r], g->stream));
    }
  }

  void poll() {
    if (cudaStreamQuery(g->stream) == cudaSuccess) ++gaps;
    for (int r = 0; r < kRing; ++r) {
      if (cudaEventQuery(ev[r]) == cudaSuccess) {
        for (uint32_t i = 0; i < batch; ++i) g->enqueue_once();
        CUDA_CHECK(cudaEventRecord(ev[r], g->stream));
      }
    }
    cudaGetLastError();
  }

  void drain() {
    CUDA_CHECK(cudaStreamSynchronize(g->stream));
    for (int r = 0; r < kRing; ++r)
      if (ev[r]) { cudaEventDestroy(ev[r]); ev[r] = nullptr; }
  }
};

// Timed comm burst that keeps polling the GEMM pump while it waits -- a
// blocking sync here would let the GEMM ring drain after ~20 ms and the rest
// of the burst would run uncontended, silently underestimating S_m.
static double comm_burst_ms_pumped(Comm& c, uint32_t n_per_stream,
                                   GemmPump* gp) {
  CUDA_CHECK(cudaSetDevice(c.gemm_dev));
  if (c.nstreams == 1) {
    cudaEvent_t b, e;
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(b, c.st[0]));
    c.enqueue_msgs(0, n_per_stream);
    CUDA_CHECK(cudaEventRecord(e, c.st[0]));
    while (cudaEventQuery(e) != cudaSuccess) {
      gp->poll();
      std::this_thread::yield();
    }
    cudaGetLastError();
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, b, e));
    cudaEventDestroy(b);
    cudaEventDestroy(e);
    return (double)ms;
  }
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const auto t0 = std::chrono::steady_clock::now();
  c.enqueue_msgs(0, n_per_stream);
  CUDA_CHECK(cudaEventRecord(e0, c.st[0]));
  c.enqueue_msgs(1, n_per_stream);
  CUDA_CHECK(cudaEventRecord(e1, c.st[1]));
  while (cudaEventQuery(e0) != cudaSuccess ||
         cudaEventQuery(e1) != cudaSuccess) {
    gp->poll();
    std::this_thread::yield();
  }
  cudaGetLastError();
  const auto t1 = std::chrono::steady_clock::now();
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);
  if (cfg.sizes.empty())
    for (int n : {1024, 2048, 4096, 8192}) cfg.sizes.push_back({n, n, n});

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (cfg.src_dev >= ndev) {
    std::fprintf(stderr, "src GPU %d not present (%d found)\n", cfg.src_dev,
                 ndev);
    return 1;
  }
  const bool have_dst = cfg.dst_dev < ndev && cfg.dst_dev != cfg.src_dev;

  int can_peer = 0;
  if (have_dst) {
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, cfg.src_dev, cfg.dst_dev));
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
  }
  if (cfg.comms.empty()) cfg.comms.push_back(can_peer ? "p2p" : "d2d");

  const int gemm_dev = cfg.gemm_dev == "dst" && have_dst ? cfg.dst_dev
                                                         : cfg.src_dev;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, gemm_dev));

  std::printf("=== CE / GEMM overlap: does CE traffic cost compute time? ===\n");
  std::printf("GEMM on GPU %d  %-28s sm_%d%d  %d SMs  clk %d MHz\n", gemm_dev,
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              prop.clockRate / 1000);
  std::printf("dtype %s   msg %llu MiB   comm-buf %llu MiB   window ~%.0f ms   "
              "p2p dir %s   peer %s\n",
              cfg.dtype.c_str(), (unsigned long long)(cfg.msg >> 20),
              (unsigned long long)(cfg.comm_buf >> 20), cfg.window_ms,
              cfg.dir.c_str(), can_peer ? "yes" : "NO");
  std::printf("[note] lock clocks (nvidia-smi -lgc) before trusting slowdowns;"
              " DVFS can masquerade as contention.\n\n");

  FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "a");
    if (csv)
      std::fprintf(csv,
                   "m,n,k,dtype,comm,dir,msg_bytes,gemm_alone_us,gemm_ovl_us,"
                   "gemm_alone_p95,gemm_ovl_p95,S_c,tflops_alone,tflops_ovl,"
                   "comm_alone_gbps,comm_ovl_gbps_approx,comm_ovl_gbps,S_m,"
                   "gaps\n");
  }

  std::map<std::string, Comm> comm_objs;
  std::map<std::string, CommCal> comm_cal;

  auto get_comm = [&](const std::string& kind) -> Comm* {
    if (kind == "p2p" && !can_peer) return nullptr;
    if ((kind == "p2p") && !have_dst) return nullptr;
    auto it = comm_objs.find(kind);
    if (it == comm_objs.end()) {
      Comm c;
      c.init(kind, cfg, gemm_dev);
      it = comm_objs.emplace(kind, c).first;
      comm_cal[kind] = comm_calibrate(it->second, cfg.window_ms);
    }
    return &it->second;
  };

  for (const Shape& sh : cfg.sizes) {
    Gemm g;
    g.init(gemm_dev, sh, cfg.dtype);

    // Calibrate the iteration count to the window, then measure GEMM alone
    // with exactly the same enqueue structure as the overlapped run.
    CUDA_CHECK(cudaSetDevice(gemm_dev));
    g.enqueue_timed(3, 5);
    CUDA_CHECK(cudaEventSynchronize(g.e_end));
    double cal_us = 0;
    (void)g.collect(5, &cal_us);
    const double est_iter_us = cal_us / 5.0;
    int iters = cfg.iters > 0
                    ? std::min(cfg.iters, kMaxIters)
                    : (int)std::max(5.0, std::min((double)kMaxIters,
                                                  cfg.window_ms * 1000.0 /
                                                      est_iter_us));

    g.enqueue_timed(3, iters);
    CUDA_CHECK(cudaEventSynchronize(g.e_end));
    double alone_total_us = 0;
    const Stats alone = g.collect(iters, &alone_total_us);
    const double tflops_alone = g.flop() / (alone.mean * 1e-6) / 1e12;

    std::printf(
        "--- GEMM %dx%dx%d %s: alone %.1f us/iter (p95 %.1f), %.1f TFLOP/s, "
        "%d iters ---\n",
        sh.m, sh.n, sh.k, cfg.dtype.c_str(), alone.mean, alone.p95,
        tflops_alone, iters);
    std::printf("%-4s %-5s | %11s %9s %6s %8s | %10s %10s %6s | %s\n", "comm",
                "dir", "ovl us/iter", "ovl p95", "S_c", "TFLOP/s",
                "alone GB/s", "ovl GB/s", "S_m", "note");

    for (const auto& kind : cfg.comms) {
      Comm* c = get_comm(kind);
      if (!c) {
        std::printf("%-4s : skipped (needs P2P peer access)\n", kind.c_str());
        continue;
      }
      const CommCal& cal = comm_cal[kind];
      const char* dir_str = kind == "p2p" ? cfg.dir.c_str() : "-";

      // ---- phase A: GEMM timed, CE kept saturated by the pump ------------
      CommPump pump;
      pump.start(c, cal.chunk_msgs);
      const auto tA0 = std::chrono::steady_clock::now();
      g.enqueue_timed(3, iters, [&] { pump.poll(); });
      while (cudaEventQuery(g.e_end) != cudaSuccess) {
        pump.poll();
        std::this_thread::yield();
      }
      cudaGetLastError();
      const auto tA1 = std::chrono::steady_clock::now();
      pump.drain();
      double ovl_total_us = 0;
      const Stats ovl = g.collect(iters, &ovl_total_us);
      const double wall_s = std::chrono::duration<double>(tA1 - tA0).count();
      const double ovl_gbps_approx =
          wall_s > 0 ? pump.completed_bytes / wall_s / 1e9 : 0;
      const double s_c = alone.mean > 0 ? ovl.mean / alone.mean : 0;
      const double tflops_ovl = g.flop() / (ovl.mean * 1e-6) / 1e12;

      // ---- phase B: CE timed, GEMM kept running by the pump --------------
      double ovl_gbps = 0, s_m = 0;
      if (cfg.do_sm) {
        GemmPump gp;
        gp.start(&g, alone.mean);
        sleep_us(2000);  // let the first GEMMs actually start
        const double ms = comm_burst_ms_pumped(*c, cal.burst_msgs, &gp);
        gp.drain();
        ovl_gbps =
            (double)cal.burst_msgs * c->nstreams * c->msg / (ms * 1e-3) / 1e9;
        s_m = ovl_gbps > 0 ? cal.gbps / ovl_gbps : 0;
      }

      const char* note = pump.gaps > 0 ? "[!] CE went idle mid-window" : "";
      std::printf("%-4s %-5s | %11.1f %9.1f %6.3f %8.1f |"
                  " %10.2f %10.2f %6.2f | %s\n",
                  kind.c_str(), dir_str, ovl.mean, ovl.p95, s_c, tflops_ovl,
                  cal.gbps, cfg.do_sm ? ovl_gbps : ovl_gbps_approx, s_m, note);
      std::fflush(stdout);

      if (csv)
        std::fprintf(csv,
                     "%d,%d,%d,%s,%s,%s,%llu,%.2f,%.2f,%.2f,%.2f,%.4f,%.1f,"
                     "%.1f,%.3f,%.3f,%.3f,%.3f,%d\n",
                     sh.m, sh.n, sh.k, cfg.dtype.c_str(), kind.c_str(), dir_str,
                     (unsigned long long)cfg.msg, alone.mean, ovl.mean,
                     alone.p95, ovl.p95, s_c, tflops_alone, tflops_ovl,
                     cal.gbps, ovl_gbps_approx, ovl_gbps, s_m, pump.gaps);
    }
    std::printf("\n");
    g.destroy();
  }

  std::printf(
      "S_c = GEMM iter time overlapped / alone (1.00 = CE is free for "
      "compute).\nS_m = CE bandwidth alone / under GEMM. 'ovl GB/s' is "
      "event-timed in phase B\n(or pump-approximate with --no-sm). Compare "
      "p2p/d2d against h2d/d2h: if S_c\nis high only for the former, the cost "
      "is HBM/L2 bandwidth contention, not\nthe copy engine itself.\n");

  for (auto& kv : comm_objs) kv.second.destroy();
  if (csv) std::fclose(csv);
  return 0;
}
