// exp4_transport.cuh
//
// Device-side plumbing for exp4_ag_tile_transport: the flag protocol, the
// TMA (cp.async.bulk) helpers, the tile-blocked layout, and the kernels.
//
// The PTX helpers come verbatim from proven code elsewhere in this repo:
//   - ld_acquire_sys / st_release_sys        signalling/signal_fanin.cu
//   - mbarrier / cp.async.bulk 1D + pipeline  tests/p2p_ce_vs_tma.cu
// The only new instruction is the 2D tensor-map load (tma_load_tensor2d),
// whose descriptor is host-encoded with cuTensorMapEncodeTiled over a PEER
// pointer -- the same driver call cute::make_tma_copy performs for the peer
// descriptors in rs_dma_sm90.cuh, so the "TMA over NVLink UVA" premise is
// already established in-repo.
//
// Driver API note: cuStreamWriteValue32 and cuTensorMapEncodeTiled are
// resolved at runtime via cudaGetDriverEntryPoint (the mechanism CUTLASS
// itself uses), so cuda.h is included for its TYPES only and the Makefile
// stays free of -lcuda.

#pragma once

#include <cuda.h>  // types only (CUtensorMap, CUstream, ...); no cu* link deps
#include <cuda_fp16.h>

#include "common.cuh"

namespace e4 {

// ---------------------------------------------------------------------------
// geometry constants
// ---------------------------------------------------------------------------

constexpr int kRbRows = 128;                       // rows per row-block
constexpr int kPanelCols = 64;                     // columns per panel
constexpr int kPanelElems = kRbRows * kPanelCols;  // halves per panel
constexpr uint32_t kPanelBytes = kPanelElems * 2;  // 16 KiB
constexpr int kThreads = 512;                      // block size, both roles
constexpr int kStages = 8;      // TMA pipeline depth: 8 x 16 KiB smem
constexpr int kFlagStride = 32; // ints per flag -> one 128 B line each
constexpr uint32_t kArmed = 0x7FFFFFFFu;  // "local chunk, always ready"
constexpr int kMaxWorld = 8;
// Extended panel sweep: M=65536, K=8192 on four GPUs gives 16384 panels/shard
// and 65536 compute units / flag slots at H=1.  This permits H=16384, i.e.
// one 256-MiB ready group per peer shard.
constexpr int kMaxChunks = 65536;
constexpr int kMaxUnits = 65536;
constexpr int kLogIters = 8;     // per-tile timestamp ring depth

inline size_t smem_bytes() {
  return (size_t)kStages * kPanelBytes + kStages * sizeof(uint64_t);
}

// ---------------------------------------------------------------------------
// layout / ordering inlines (single implementation shared by the transform
// kernel, the comm-block addressing and the consumer -- plus an independent
// host reference in --verify to guard against a common-mode bug here)
// ---------------------------------------------------------------------------

// element offset of panel (rb, p) in a tile-blocked buffer [rb][p][128][64]
__host__ __device__ __forceinline__ size_t blocked_off(int rb, int p, int P) {
  return ((size_t)rb * P + p) * kPanelElems;
}

__host__ __device__ __forceinline__ int flag_idx(int c) {
  return c * kFlagStride;
}

// i-th row-block in consume order: own shard first, then the ring peers
// (rank+1, rank+2, ...) -- rank-major when one comm stream serializes the
// ring (and for the TMA job list), chunk-major across peers when per-peer CE
// streams run in parallel. Mirrors exp2's segment order and flux's
// problem_blocks_m_offset rotation.
__host__ __device__ __forceinline__ int rb_order_at(int i, int rank, int world,
                                                    int rb_per_shard, int G,
                                                    int chunk_major) {
  if (i < rb_per_shard) return rank * rb_per_shard + i;
  const int j = i - rb_per_shard;
  const int n_peers = world - 1;
  int owner, off;
  if (!chunk_major) {
    owner = (rank + 1 + j / rb_per_shard) % world;
    off = j % rb_per_shard;
  } else {
    const int cj = j / G;      // (chunk slot, peer) pair index
    const int within = j % G;
    owner = (rank + 1 + cj % n_peers) % world;
    off = (cj / n_peers) * G + within;
  }
  return owner * rb_per_shard + off;
}

// ---------------------------------------------------------------------------
// flag PTX (signalling/signal_fanin.cu:98-112)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ uint32_t atomic_add_acq_rel_gpu(uint32_t* p,
                                                            uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(p), "r"(v)
               : "memory");
  return old;
}

__device__ __forceinline__ uint32_t ld_acquire_flag(const uint32_t* p,
                                                     int system_scope) {
  return system_scope ? ld_acquire_sys(p) : ld_acquire_gpu(p);
}

// %globaltimer: ns-resolution clock, consistent across the SMs of one GPU.
// Cross-GPU offsets are probed at startup; headline stats only ever subtract
// same-GPU values.
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  // The memory clobber prevents nvcc from moving surrounding loads/stores
  // across the sample; NVIDIA's inline-PTX guide recommends this form for
  // timing boundaries.
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}

__device__ __forceinline__ uint32_t smid() {
  uint32_t id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

// ---------------------------------------------------------------------------
// mbarrier / cp.async.bulk helpers (tests/p2p_ce_vs_tma.cu:88-169)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)),
               "r"(count));
}

// Makes prior generic-proxy shared-memory writes (mbarrier init) visible to
// the async proxy that TMA operates in.
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

// Branch-free try_wait: avoids duplicated PTX labels when inlined more than
// once in a single kernel.
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

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  while (!mbarrier_try_wait(bar, phase)) {
  }
}

// 2D tensor-map load: global (peer or local, described by *tmap) -> shared,
// completion signalled through `bar`'s tx count. Coordinates are in ELEMENTS,
// x = inner (K) offset, y = outer (row) offset.
__device__ __forceinline__ void tma_load_tensor2d(void* smem_dst,
                                                  const CUtensorMap* tmap,
                                                  int x, int y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_addr(smem_dst)),
      "l"(tmap), "r"(x), "r"(y), "r"(smem_addr(bar))
      : "memory");
}

// shared::cta -> global (local HBM here).
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

// Waits until at most N bulk-store groups still have the *source* smem live.
template <int N>
__device__ __forceinline__ void tma_store_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

// Waits until at most N bulk-store groups are outstanding end to end, i.e.
// the writes have landed in HBM -- required before publishing the flag.
template <int N>
__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// driver API, resolved at runtime (no -lcuda; CUTLASS-style entry points)
// ---------------------------------------------------------------------------

struct DriverApi {
  typedef CUresult (*PFN_w32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
  typedef CUresult (*PFN_enc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t,
                              void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*,
                              CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  PFN_w32 write32 = nullptr;  // optional: falls back to flag_set_kernel
  PFN_w32 wait32 = nullptr;   // required for consumer-resident CE start gate
  PFN_enc encode = nullptr;   // required

  void init() {
    cudaDriverEntryPointQueryResult st{};
    void* fn = nullptr;
    if (cudaGetDriverEntryPoint("cuStreamWriteValue32", &fn, cudaEnableDefault,
                                &st) == cudaSuccess &&
        st == cudaDriverEntryPointSuccess)
      write32 = (PFN_w32)fn;
    fn = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPoint("cuStreamWaitValue32", &fn,
                                       cudaEnableDefault, &st));
    if (st != cudaDriverEntryPointSuccess || !fn) {
      std::fprintf(stderr, "cuStreamWaitValue32 not available\n");
      std::exit(1);
    }
    wait32 = (PFN_w32)fn;
    fn = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn,
                                       cudaEnableDefault, &st));
    if (st != cudaDriverEntryPointSuccess || !fn) {
      std::fprintf(stderr, "cuTensorMapEncodeTiled not available\n");
      std::exit(1);
    }
    encode = (PFN_enc)fn;
  }
};

// 2D row-major tensor map over one rank's [m_shard, K] fp16 shard.
inline void encode_tmap_2d(const DriverApi& drv, CUtensorMap* out, void* base,
                           int K, int m_shard) {
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)m_shard};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};  // bytes, multiple of 16
  cuuint32_t box[2] = {kPanelCols, kRbRows};
  cuuint32_t estr[2] = {1, 1};
  CUresult res = drv.encode(out, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, base,
                            dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_NONE,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (res != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled -> %d\n", (int)res);
    std::exit(1);
  }
}

// ---------------------------------------------------------------------------
// kernel config
// ---------------------------------------------------------------------------

struct TmaMaps {
  // [0..world-2] = ring peers; last slot = own A_src for local mode.
  CUtensorMap m[kMaxWorld];
};

struct KCfg {
  int rank = 0, world = 0;
  int P = 0;             // K/64 panels per row-block
  int rb_per_shard = 0;
  int G = 0, cps = 0;    // row-blocks per chunk, chunks per shard
  int panel_mode = 0;    // 0: G row-blocks/chunk; 1: H panels/chunk
  int H = 0;             // shard-linear panels per ready flag in panel mode
  int n_comm = 0, n_compute = 0;
  int n_units = 0, slices = 0;
  int intensity = 0;     // FMA ops per loaded 16 B vector (4 chains x I/4)
  uint32_t epoch = 0;    // value producers publish this iteration
  uint32_t epoch_cmp = 0;  // spin threshold; 0 = pass-through (compute-only /
                           // bystander: one acquire load, no wait)
  int comm_enabled = 0;  // TMA comm blocks do their jobs (0 -> exit at once)
  int local_mode = 0;    // TMA comm blocks read own shard (map[3])
  int flag_scope_sys = 0; // CE stream memop: sys; same-GPU TMA producer: gpu
  int chunk_major = 0;   // consume order (see rb_order_at)
  int log_slot = -1;     // ts_log ring slot, <0 = no logging
  int timing_slot = -1;  // per-iteration persistent-block timing slot
  int timing_stride = 0; // allocated block records per timing slot
};

struct BlockTiming {
  uint64_t start_ns;
  uint64_t done_ns;
  uint32_t sm_id;
  uint32_t role;  // 0 = communication block, 1 = compute block
};

// ---------------------------------------------------------------------------
// the single consumer/comm kernel (both variants; roles split on blockIdx)
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(kThreads, 1) void ag_consume_kernel(
    const __grid_constant__ TmaMaps maps, const __grid_constant__ KCfg cfg,
    const __half* __restrict__ src_blocked,  // A_shadow (CE) | A_staged (TMA)
    __half* __restrict__ staged,             // TMA comm store target
    uint32_t* flags, uint64_t* ts_log, float* sink,
    unsigned long long* bitsum, unsigned* err, BlockTiming* block_timing,
    uint32_t* grid_arrivals, uint32_t* grid_ready,
    uint32_t* chunk_done) {
  BlockTiming* my_timing = nullptr;
  if (cfg.timing_slot >= 0) {
    my_timing = block_timing +
                (size_t)cfg.timing_slot * cfg.timing_stride + blockIdx.x;
    if (threadIdx.x == 0) {
      my_timing->sm_id = smid();
      my_timing->role = blockIdx.x < (unsigned)cfg.n_comm ? 0u : 1u;
      my_timing->start_ns = globaltimer();
    }
  }
  // Defines the block start boundary: no worker may begin its communication
  // or compute role before thread 0 has sampled the start timestamp.
  __syncthreads();

  // Cooperative-launch grid barrier.  Besides proving that all persistent
  // blocks are resident, grid_ready is the CE stream's device-side start gate:
  // no copy can begin before every waiting consumer block has arrived here.
  if (threadIdx.x == 0) {
    const uint32_t old = atomicAdd(grid_arrivals, 1u);
    if (old + 1u == (uint32_t)gridDim.x)
      st_release_sys(grid_ready, cfg.epoch);
    while (ld_acquire_sys(grid_ready) < cfg.epoch) __nanosleep(64);
  }
  __syncthreads();

  // ---------------- comm role: pull peer panels via TMA ----------------
  if (blockIdx.x < (unsigned)cfg.n_comm) {
    if (!cfg.comm_enabled) {
      if (threadIdx.x == 0 && my_timing)
        my_timing->done_ns = globaltimer();
      return;
    }
    if (threadIdx.x != 0) return;  // lane 0 owns the whole pipeline

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* buf = smem;
    uint64_t* bar =
        reinterpret_cast<uint64_t*>(smem + (size_t)kStages * kPanelBytes);
#pragma unroll
    for (int s = 0; s < kStages; ++s) mbarrier_init(&bar[s], 1);
    fence_proxy_async_shared();

    const int n_jobs = (cfg.world - 1) * cfg.cps;
    uint64_t slot_base = 0;   // monotonic stage counter across chunks
    uint32_t phase_bits = 0;  // bit s = expected parity of bar[s]

    auto process_job = [&](int j, int panel_lane, int panel_stride) {
      const int peer_i = j / cfg.cps;
      const int c_in = j % cfg.cps;
      const int owner = (cfg.rank + 1 + peer_i) % cfg.world;
      const CUtensorMap* map =
          cfg.local_mode ? &maps.m[kMaxWorld - 1] : &maps.m[peer_i];
      const int rb0 = cfg.panel_mode ? 0 : c_in * cfg.G;
      const int items = cfg.panel_mode
                            ? (cfg.H - panel_lane + panel_stride - 1) /
                                  panel_stride
                            : cfg.G * cfg.P;

      auto issue_load = [&](int it) {
        const uint32_t s = (uint32_t)((slot_base + (uint64_t)it) % kStages);
        const int panel_in_chunk = panel_lane + it * panel_stride;
        const int flat_panel =
            cfg.panel_mode ? c_in * cfg.H + panel_in_chunk : it;
        const int rb_c = flat_panel / cfg.P;
        const int p = flat_panel % cfg.P;
        const int rb_l = rb0 + rb_c;  // row-block within owner shard
        mbarrier_arrive_expect_tx(&bar[s], kPanelBytes);
        tma_load_tensor2d(buf + (size_t)s * kPanelBytes, map, p * kPanelCols,
                          rb_l * kRbRows, &bar[s]);
      };

      int issued = 0;
      for (; issued < items && issued < kStages - 2; ++issued)
        issue_load(issued);
      for (int d = 0; d < items; ++d) {
        const uint32_t s = (uint32_t)((slot_base + (uint64_t)d) % kStages);
        mbarrier_wait(&bar[s], (phase_bits >> s) & 1u);
        phase_bits ^= 1u << s;
        const int panel_in_chunk = panel_lane + d * panel_stride;
        const int flat_panel =
            cfg.panel_mode ? c_in * cfg.H + panel_in_chunk : d;
        const int rb_c = flat_panel / cfg.P;
        const int p = flat_panel % cfg.P;
        const int rb_g = owner * cfg.rb_per_shard + rb0 + rb_c;
        tma_store_1d(staged + blocked_off(rb_g, p, cfg.P),
                     buf + (size_t)s * kPanelBytes, kPanelBytes);
        tma_store_commit();
        if (issued < items) {
          // recycle the oldest buffer only after its store drained it
          tma_store_wait_read<1>();
          issue_load(issued++);
        }
      }
      slot_base += items;
      // the flag says "data is IN HBM", so wait end-to-end, not just .read
      tma_store_wait_all<0>();
      const int chunk = owner * cfg.cps + c_in;
      if (!cfg.panel_mode) {
        st_release_gpu(&flags[flag_idx(chunk)], cfg.epoch);
      } else {
        // Every communication block contributes `items` completed panels to
        // this ready group.  acq_rel RMWs form a release sequence across the
        // contributors, so the block that observes all H panels also observes
        // every preceding TMA-store completion before publishing the flag.
        const uint32_t old =
            atomic_add_acq_rel_gpu(&chunk_done[chunk], (uint32_t)items);
        if (old + (uint32_t)items == (uint32_t)cfg.H)
          st_release_gpu(&flags[flag_idx(chunk)], cfg.epoch);
      }
    };

    if (cfg.panel_mode) {
      // One scheduling rule for the entire H sweep: globally number all remote
      // panels in rank-major ready-group order and statically stripe them over
      // the communication blocks.  This gives every block the same total
      // panel count without switching algorithms when n_jobs stops dividing
      // n_comm.  Adjacent striped panels belonging to the same ready group are
      // kept in one process_job call so its TMA loads/stores remain pipelined.
      const int total_panels = n_jobs * cfg.H;
      int panel_linear = (int)blockIdx.x;
      while (panel_linear < total_panels) {
        const int j = panel_linear / cfg.H;
        const int panel_lane = panel_linear % cfg.H;
        process_job(j, panel_lane, cfg.n_comm);
        const int panels_from_this_block =
            (cfg.H - panel_lane + cfg.n_comm - 1) / cfg.n_comm;
        panel_linear += panels_from_this_block * cfg.n_comm;
      }
    } else {
      for (int j = blockIdx.x; j < n_jobs; j += cfg.n_comm)
        process_job(j, 0, 1);
    }
    if (my_timing) my_timing->done_ns = globaltimer();
    return;  // exit when the job list is exhausted (no block migrates onto
             // the freed SM at 1 block/SM; spinning here would only burn
             // power and perturb clocks)
  }

  // ---------------- compute role: flag-gated fixed workload ----------------
  const int cb = blockIdx.x - cfg.n_comm;
  float facc = 0.f;
  unsigned long long bacc = 0ull;
  const int chain = cfg.intensity >> 2;

  for (int i = cb; i < cfg.n_units; i += cfg.n_compute) {
    const int unit_in_rb = cfg.panel_mode ? cfg.P : cfg.slices;
    const int rb = rb_order_at(i / unit_in_rb, cfg.rank, cfg.world,
                               cfg.rb_per_shard, cfg.G, cfg.chunk_major);
    const int sl = i % unit_in_rb;
    const int c = cfg.panel_mode ? (rb * cfg.P + sl) / cfg.H
                                 : rb / cfg.G;
    uint64_t t0 = 0, t1 = 0;
    if (threadIdx.x == 0) {
      t0 = globaltimer();
      uint32_t v =
          ld_acquire_flag(&flags[flag_idx(c)], cfg.flag_scope_sys);
      while (v < cfg.epoch_cmp) {  // pipeline_e2e.cu spin + backoff
        __nanosleep(128);
        v = ld_acquire_flag(&flags[flag_idx(c)], cfg.flag_scope_sys);
      }
      if (cfg.epoch_cmp && v != cfg.epoch && v != kArmed) atomicAdd(err, 1u);
      t1 = globaltimer();
    }
    __syncthreads();

    // Row mode streams one K-slice. Panel mode streams exactly one 128x64
    // panel, so total synthetic compute is fixed while H changes.
    const int pp = cfg.panel_mode ? 1 : cfg.P / cfg.slices;
    const int p_begin = cfg.panel_mode ? sl : sl * pp;
    const uint4* v4 = reinterpret_cast<const uint4*>(
        src_blocked + blocked_off(rb, p_begin, cfg.P));
    const int n_vec = pp * (kPanelElems / 8);
    for (int idx = threadIdx.x; idx < n_vec; idx += kThreads) {
      const uint4 u = v4[idx];
      bacc += (unsigned long long)u.x + u.y + u.z + u.w;
      const __half2* h2 = reinterpret_cast<const __half2*>(&u);
      float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]);
      float2 f2 = __half22float2(h2[2]), f3 = __half22float2(h2[3]);
      // 4 independent chains (ILP) x intensity/4 -> `intensity` FMAs per 16 B
      float a0 = f0.x + f0.y, a1 = f1.x + f1.y;
      float a2 = f2.x + f2.y, a3 = f3.x + f3.y;
#pragma unroll 4
      for (int t = 0; t < chain; ++t) {
        a0 = fmaf(a0, 0.999999f, 0.001f);
        a1 = fmaf(a1, 0.999999f, 0.001f);
        a2 = fmaf(a2, 0.999999f, 0.002f);
        a3 = fmaf(a3, 0.999999f, 0.002f);
      }
      facc += a0 + a1 + a2 + a3;
    }

    if (threadIdx.x == 0 && cfg.log_slot >= 0) {
      uint64_t* e = ts_log + ((size_t)cfg.log_slot * kMaxUnits + i) * 3;
      e[0] = t0;
      e[1] = t1;
      e[2] = globaltimer();
    }
  }

  // one atomic per warp per kernel, not per unit (bitsum stays deterministic
  // across grid splits: wrapping add is commutative)
  for (int off = 16; off; off >>= 1) {
    facc += __shfl_down_sync(0xffffffffu, facc, off);
    bacc += __shfl_down_sync(0xffffffffu, bacc, off);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(sink, facc);
    atomicAdd(bitsum, bacc);
  }
  __syncthreads();
  if (threadIdx.x == 0 && my_timing)
    my_timing->done_ns = globaltimer();
}

// ---------------------------------------------------------------------------
// support kernels
// ---------------------------------------------------------------------------

// row-major shard -> tile-blocked buffer (init-time; may read peers over UVA)
__global__ void transform_blocked_kernel(const __half* __restrict__ src,
                                         __half* __restrict__ dst, int owner,
                                         int rb_per_shard, int P) {
  const int K = P * kPanelCols;
  for (int pan = blockIdx.x; pan < rb_per_shard * P; pan += gridDim.x) {
    const int rb_l = pan / P, p = pan % P;
    const __half* s = src + (size_t)rb_l * kRbRows * K + (size_t)p * kPanelCols;
    __half* d = dst + blocked_off(owner * rb_per_shard + rb_l, p, P);
    for (int idx = threadIdx.x; idx < kPanelElems; idx += blockDim.x) {
      d[idx] = s[(size_t)(idx / kPanelCols) * K + idx % kPanelCols];
    }
  }
}

// arrival mode: grid-stride observers record when each flag reaches `epoch`.
// arr_log slot layout: [kMaxChunks] arrivals + [kMaxChunks] = kernel-entry t.
__global__ void observer_kernel(const uint32_t* flags, int n_chunks,
                                uint32_t epoch, uint64_t* arr_slot) {
  if (blockIdx.x == 0 && threadIdx.x == 0) arr_slot[kMaxChunks] = globaltimer();
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n_chunks;
       t += blockDim.x * gridDim.x) {
    while (ld_acquire_sys(&flags[flag_idx(t)]) < epoch) __nanosleep(256);
    arr_slot[t] = globaltimer();
  }
}

// CE-side flag publish fallback when cuStreamWriteValue32 is unavailable
// (or forced with --flag-kernel): costs a launch and briefly occupies an SM,
// which is exactly why it is only a labeled cross-check row.
__global__ void flag_set_kernel(uint32_t* flag, uint32_t v) {
  st_release_sys(flag, v);
}

__global__ void timer_probe_kernel(uint64_t* out) { *out = globaltimer(); }

__global__ void compare_u4_kernel(const uint4* a, const uint4* b, size_t n,
                                  unsigned* mismatches) {
  bool bad = false;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 x = a[i], y = b[i];
    bad |= x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w;
  }
  if (bad) atomicAdd(mismatches, 1u);
}

}  // namespace e4
