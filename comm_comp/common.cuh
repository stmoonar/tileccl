// common.cuh
//
// Shared plumbing for the comm_comp experiments: error checking, CLI parsing
// helpers, per-iteration event timing, peer-access setup and data-fill
// kernels. Single-process multi-GPU throughout -- rank r == CUDA device r,
// P2P over NVLink via cudaDeviceEnablePeerAccess + UVA pointers, so no MPI /
// NCCL / nvshmem is needed.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define CUTLASS_CHECK(expr)                                                   \
  do {                                                                        \
    cutlass::Status _st = (expr);                                             \
    if (_st != cutlass::Status::kSuccess) {                                   \
      std::fprintf(stderr, "[CUTLASS] %s:%d: %s -> %s\n", __FILE__, __LINE__, \
                   #expr, cutlassGetStatusString(_st));                       \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

inline uint64_t parse_bytes(const std::string& s) {
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

inline std::vector<std::string> split_csv(const std::string& s) {
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

struct Shape3 {
  int m = 0, n = 0, k = 0;
};

// "4096" -> 4096^3, "8192x8192x128" -> m x n x k
inline Shape3 parse_shape(const std::string& tok) {
  Shape3 s;
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

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

struct Stats {
  double mean = 0, p50 = 0, p95 = 0, mn = 0, mx = 0;
};

inline Stats make_stats(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  std::sort(v.begin(), v.end());
  s.mn = v.front();
  s.p50 = v[v.size() / 2];
  s.p95 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.95))];
  s.mx = v.back();
  return s;
}

// ---------------------------------------------------------------------------
// device / peer setup
// ---------------------------------------------------------------------------

// Enable P2P access between every pair in `devs`. Exits if any pair cannot
// peer (the experiments assume full NVLink connectivity, true on 4xH800).
inline void enable_peer_all(const std::vector<int>& devs) {
  for (int a : devs) {
    CUDA_CHECK(cudaSetDevice(a));
    for (int b : devs) {
      if (a == b) continue;
      int can = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can, a, b));
      if (!can) {
        std::fprintf(stderr, "GPU %d cannot peer-access GPU %d\n", a, b);
        std::exit(1);
      }
      cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
      if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();  // clear
      } else {
        CUDA_CHECK(e);
      }
    }
  }
}

inline void print_device_line(int dev) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  // (cudaDeviceProp::clockRate is deprecated in CUDA 12.x; query the
  // attribute instead so this builds warning-free on 12.9.)
  int khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
  std::printf("GPU %d: %s  sm_%d%d  %d SMs  %d MHz\n", dev, prop.name,
              prop.major, prop.minor, prop.multiProcessorCount, khz / 1000);
}

// ---------------------------------------------------------------------------
// fill kernels
// ---------------------------------------------------------------------------

static __global__ void fill_half_kernel(__half* p, size_t n, uint32_t salt) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += step) {
    const uint32_t h = ((uint32_t)i + salt) * 2654435761u;
    p[i] = __float2half(((h >> 16) & 0xffffu) * (1.0f / 65536.0f) - 0.5f);
  }
}

inline void fill_half(void* p, size_t n, uint32_t salt = 0) {
  fill_half_kernel<<<256, 256>>>((__half*)p, n, salt);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// misc
// ---------------------------------------------------------------------------

inline void sleep_us(long long us) {
  std::this_thread::sleep_for(std::chrono::microseconds(us));
}

inline double wall_ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}
