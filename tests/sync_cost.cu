// sync_cost.cu
//
// Microbenchmark for the *synchronization* half of fine-grained P2P
// pipelines. In a fused kernel every tile costs
//     payload stores  +  release fence  +  signal  (+ dst-side polling)
// and at small tile sizes the sync side can dominate the payload side. This
// tool measures each primitive in isolation so that per-tile cost can be
// decomposed instead of being misattributed to the transfer mechanism.
//
// Measurements:
//   pingpong   one-way flag latency between the two GPUs, from a ping-pong
//              of st.release.sys / ld.acquire.sys flags: RTT/2. This is the
//              floor for any "tile ready" notification.
//   signal     per-message cost of the producer-side sequence, for several
//              payload sizes S:
//                a) S bytes of plain uint4 stores into the peer
//                b) a) + st.release.sys flag        (release covers stores)
//                c) a) + __threadfence_system + plain flag store
//              (b - a) and (c - a) are the fence+signal overheads; their
//              growth with S shows the cost of draining outstanding P2P
//              writes at the fence.
//   atomic     round-trip latency of a dependent atomicAdd_system chain on a
//              peer-resident counter -- the cost of remote atomics used as
//              signals/counters (MoE token counting etc.).
//   mbarrier   local smem mbarrier arrive+wait cycle, in cycles/op via
//              clock64 -- the in-kernel cost of TMA-style pipeline sync.
//
// Cross-GPU timing note: the two GPUs' clocks are not synchronized, so all
// numbers here derive from round trips timed on one device, never from
// comparing timestamps across devices.
//
// Build:  make            (see tests/Makefile)
// Run:    ./sync_cost
//         ./sync_cost --rounds 5000 --sizes 0,2K,32K

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

// ---------------------------------------------------------------------------
// system-scope primitives
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

// One thread per GPU. The initiator writes the peer's flag then waits for its
// local flag; the responder mirrors it. Total time / rounds = RTT.
__global__ void pingpong_kernel(uint32_t* local_flag, uint32_t* peer_flag,
                                uint32_t rounds, int initiator) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (uint32_t i = 1; i <= rounds; ++i) {
    if (initiator) {
      st_release_sys(peer_flag, i);
      while (ld_acquire_sys(local_flag) < i) {
      }
    } else {
      while (ld_acquire_sys(local_flag) < i) {
      }
      st_release_sys(peer_flag, i);
    }
  }
}

// Producer-side per-message sequence: payload stores, then (variant-dependent)
// fence + signal. One warp; payload is `nvec` uint4s into the peer.
//   variant 0: payload only
//   variant 1: payload + st.release.sys flag
//   variant 2: payload + __threadfence_system + plain flag store
__global__ void signal_seq_kernel(const uint4* __restrict__ src,
                                  uint4* __restrict__ peer_payload,
                                  uint32_t* peer_flag, uint32_t nvec,
                                  uint32_t rounds, int variant) {
  const uint32_t lane = threadIdx.x & 31;
  if (threadIdx.x >= 32 || blockIdx.x != 0) return;
  for (uint32_t i = 1; i <= rounds; ++i) {
    for (uint32_t v = lane; v < nvec; v += 32) peer_payload[v] = src[v];
    if (lane == 0) {
      if (variant == 1) {
        st_release_sys(peer_flag, i);
      } else if (variant == 2) {
        __threadfence_system();
        st_plain(peer_flag, i);
      }
    }
    __syncwarp();
  }
}

// Dependent chain of system-scope atomics on a peer counter: the returned
// value feeds the next op's address bits, so the chain serializes and the
// per-op time is the full remote-atomic round trip. `x >> 31` is always 0 at
// runtime (the counter never reaches 2^31) but is opaque to the compiler, so
// the address dependency -- and with it the serialization -- survives.
__global__ void atomic_rtt_kernel(unsigned int* peer_ctr, uint32_t rounds,
                                  unsigned int* sink) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  unsigned int x = 0;
  for (uint32_t i = 0; i < rounds; ++i)
    x = atomicAdd_system(peer_ctr + (x >> 31), 1u);
  if (x == 0xdeadbeefu) *sink = x;
}

// Local smem mbarrier arrive+wait cycle, timed in-kernel with clock64.
__global__ void mbarrier_cycle_kernel(uint32_t rounds,
                                      unsigned long long* out_cycles) {
  __shared__ __align__(8) uint64_t bar;
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint32_t phase = 0;
  const unsigned long long t0 = clock64();
  for (uint32_t i = 0; i < rounds; ++i) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr)
                 : "memory");
    uint32_t ok = 0;
    do {
      asm volatile(
          "{\n"
          ".reg .pred P;\n"
          "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
          "selp.b32 %0, 1, 0, P;\n"
          "}\n"
          : "=r"(ok)
          : "r"(addr), "r"(phase));
    } while (!ok);
    phase ^= 1;
  }
  *out_cycles = (clock64() - t0) / rounds;
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------

struct Config {
  uint32_t rounds = 2000;
  std::vector<uint64_t> sizes{0, 512, 2048, 8192, 32768, 131072};
  int src_dev = 0;
  int dst_dev = 1;
  int iters = 5;  // repetitions per measurement; median reported
};

static uint64_t parse_bytes(const std::string& s) {
  char* end = nullptr;
  uint64_t v = std::strtoull(s.c_str(), &end, 10);
  if (end && *end) {
    switch (*end) {
      case 'k': case 'K': v <<= 10; break;
      case 'm': case 'M': v <<= 20; break;
      default:
        std::fprintf(stderr, "bad size suffix in '%s'\n", s.c_str());
        std::exit(1);
    }
  }
  return v;
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
    if (a == "--rounds") c.rounds = (uint32_t)std::atoi(next().c_str());
    else if (a == "--sizes") {
      c.sizes.clear();
      std::string list = next();
      size_t beg = 0;
      while (beg <= list.size()) {
        size_t end = list.find(',', beg);
        if (end == std::string::npos) end = list.size();
        if (end > beg) c.sizes.push_back(parse_bytes(list.substr(beg, end - beg)));
        beg = end + 1;
      }
    } else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--src") c.src_dev = std::atoi(next().c_str());
    else if (a == "--dst") c.dst_dev = std::atoi(next().c_str());
    else if (a == "-h" || a == "--help") {
      std::printf("usage: %s [--rounds N] [--sizes LIST] [--iters N]"
                  " [--src N] [--dst N]\n", argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      std::exit(1);
    }
  }
  return c;
}

// Median-of-iters wall time of `launch` on `stream`, in us.
template <typename Launch>
static double median_us(Launch&& launch, cudaStream_t stream, int iters) {
  std::vector<double> t;
  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));
  launch(stream);  // warmup
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(beg, stream));
    launch(stream);
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    t.push_back((double)ms * 1000.0);
  }
  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return 1;
  }
  int can_peer = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_peer, cfg.src_dev, cfg.dst_dev));
  cudaDeviceProp psrc;
  CUDA_CHECK(cudaGetDeviceProperties(&psrc, cfg.src_dev));

  std::printf("=== P2P synchronization-primitive costs ===\n");
  std::printf("src GPU %d (%s) -> dst GPU %d   P2P: %s   rounds/measurement: %u\n\n",
              cfg.src_dev, psrc.name, cfg.dst_dev, can_peer ? "yes" : "NO",
              cfg.rounds);
  if (!can_peer) {
    std::fprintf(stderr, "P2P unavailable; every measurement here needs it.\n");
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

  // Allocations. Flags/counters live on their owner's HBM.
  uint32_t *flag_src = nullptr, *flag_dst = nullptr;
  unsigned int* ctr_dst = nullptr;
  unsigned int* sink = nullptr;
  uint4 *payload_dst = nullptr, *src_buf = nullptr;
  unsigned long long* cycles_out = nullptr;
  const uint64_t max_size =
      *std::max_element(cfg.sizes.begin(), cfg.sizes.end());
  const uint64_t buf_bytes = std::max<uint64_t>(max_size, 16);

  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaMalloc(&flag_dst, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&ctr_dst, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&payload_dst, buf_bytes));
  CUDA_CHECK(cudaMemset(flag_dst, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(ctr_dst, 0, sizeof(unsigned int)));
  cudaStream_t s_dst;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_dst, cudaStreamNonBlocking));

  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaMalloc(&flag_src, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&src_buf, buf_bytes));
  CUDA_CHECK(cudaMalloc(&cycles_out, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(flag_src, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(src_buf, 5, buf_bytes));
  cudaStream_t s_src;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_src, cudaStreamNonBlocking));

  // ---- pingpong ------------------------------------------------------------
  {
    double rtt_us = 0;
    for (int it = 0; it < cfg.iters; ++it) {
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      CUDA_CHECK(cudaMemset(flag_src, 0, sizeof(uint32_t)));
      CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
      CUDA_CHECK(cudaMemset(flag_dst, 0, sizeof(uint32_t)));
      CUDA_CHECK(cudaDeviceSynchronize());
      // Responder first, then the timed initiator.
      pingpong_kernel<<<1, 32, 0, s_dst>>>(flag_dst, flag_src, cfg.rounds, 0);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      cudaEvent_t beg, end;
      CUDA_CHECK(cudaEventCreate(&beg));
      CUDA_CHECK(cudaEventCreate(&end));
      CUDA_CHECK(cudaEventRecord(beg, s_src));
      pingpong_kernel<<<1, 32, 0, s_src>>>(flag_src, flag_dst, cfg.rounds, 1);
      CUDA_CHECK(cudaEventRecord(end, s_src));
      CUDA_CHECK(cudaStreamSynchronize(s_src));
      CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
      CUDA_CHECK(cudaStreamSynchronize(s_dst));
      CUDA_CHECK(cudaSetDevice(cfg.src_dev));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
      CUDA_CHECK(cudaEventDestroy(beg));
      CUDA_CHECK(cudaEventDestroy(end));
      const double v = (double)ms * 1000.0 / cfg.rounds;
      rtt_us = it == 0 ? v : std::min(rtt_us, v);
    }
    std::printf("flag ping-pong (st.release.sys / ld.acquire.sys):\n"
                "  RTT %8.3f us    one-way ~ %.3f us\n\n", rtt_us, rtt_us / 2);
  }

  // ---- signal sequences ----------------------------------------------------
  {
    std::printf("producer sequence, per message (us), one warp, %u msgs/launch:\n",
                cfg.rounds);
    std::printf("%10s %12s %12s %12s %14s %14s\n", "payload", "stores",
                "+rel.flag", "+fence+flag", "d(rel)", "d(fence)");
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    for (uint64_t S : cfg.sizes) {
      const uint32_t nvec = (uint32_t)(S / 16);
      double t[3];
      for (int variant = 0; variant < 3; ++variant) {
        auto launch = [&](cudaStream_t st) {
          signal_seq_kernel<<<1, 32, 0, st>>>(src_buf, payload_dst, flag_dst,
                                              nvec, cfg.rounds, variant);
        };
        t[variant] = median_us(launch, s_src, cfg.iters) / cfg.rounds;
      }
      std::printf("%9lluB %12.3f %12.3f %12.3f %14.3f %14.3f\n",
                  (unsigned long long)S, t[0], t[1], t[2], t[1] - t[0],
                  t[2] - t[0]);
    }
    std::printf("  d(rel)/d(fence): overhead of the release-flag / fence+flag\n"
                "  tails on top of the bare payload stores.\n\n");
  }

  // ---- remote atomic RTT ---------------------------------------------------
  {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    auto launch = [&](cudaStream_t st) {
      atomic_rtt_kernel<<<1, 32, 0, st>>>(ctr_dst, cfg.rounds, sink);
    };
    const double us = median_us(launch, s_src, cfg.iters) / cfg.rounds;
    std::printf("dependent atomicAdd_system on peer counter: %8.3f us/op\n\n",
                us);
  }

  // ---- mbarrier cycle ------------------------------------------------------
  {
    CUDA_CHECK(cudaSetDevice(cfg.src_dev));
    const uint32_t rounds = std::max(cfg.rounds, 100000u);
    mbarrier_cycle_kernel<<<1, 32, 0, s_src>>>(rounds, cycles_out);
    CUDA_CHECK(cudaStreamSynchronize(s_src));
    unsigned long long cyc = 0;
    CUDA_CHECK(cudaMemcpy(&cyc, cycles_out, sizeof(cyc), cudaMemcpyDeviceToHost));
    std::printf("local smem mbarrier arrive+wait cycle: %llu cycles/op\n",
                cyc);
  }

  CUDA_CHECK(cudaSetDevice(cfg.src_dev));
  CUDA_CHECK(cudaStreamDestroy(s_src));
  CUDA_CHECK(cudaSetDevice(cfg.dst_dev));
  CUDA_CHECK(cudaStreamDestroy(s_dst));
  return 0;
}
