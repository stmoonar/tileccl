// exp3_gemm_rs_fused.cu
//
// Experiment 3 (v2, flux-faithful): what does flux's sm90 fused GEMM+RS
// cost the GEMM?
//
// This is the flux sm90 design reproduced on stock CUTLASS 4.6 (see
// rs_gemm_kernel_sm90.cuh): the epilogue TMA-stores D LOCALLY and publishes
// a per-tile system-scope flag; two otherwise-idle producer warps per CTA
// pull peer tiles over NVLink with TMA and reduce them into a local buffer
// with red.global.add.v8.f16. Communication lives INSIDE the GEMM kernel,
// so any cost shows up directly as kernel-time inflation.
//
// Deployment-realistic harness: one process per GPU (fork), buffers
// exchanged via cudaIPC, all ranks run symmetrically -- every GPU computes
// its GEMM while pulling from and being pulled by the other three, exactly
// like a TP row-parallel layer under torchrun. A flux-style device
// barrier-all aligns ranks before each timed iteration.
//
//   baseline: the same-config stock CUTLASS cooperative GEMM (comm_none),
//             also run symmetrically on all ranks
//   fused:    the GEMM+RS kernel; its time covers compute + all pulls +
//             reduction, and (fused - baseline) is the total price of the
//             in-kernel communication: 2 warps of SM residency, epilogue
//             flag publishes, NVLink pull bandwidth, and the tail where
//             DMA warps wait for the slowest peer's tiles
//
//   slowdown = t_fused / t_baseline    per rank; also reported: effective
//   pull bandwidth (world-1)/world * M*N*2 / t_fused, and achieved TFLOP/s.
//
// --verify checks reduce_buffer == sum over ranks of their D[segment rank]
// (reference summed in fp32 from the ranks' actual D outputs; tolerance
// covers the fp16 red.add accumulation).
//
// Build & run (Linux target box):   make exp3_gemm_rs_fused
//   ./exp3_gemm_rs_fused                       # 8192^3 on all GPUs
//   ./exp3_gemm_rs_fused --m 8192 --n 8192 --k 2048 --verify

#include "common.cuh"
#include "gemm_sm90.cuh"
#include "rs_gemm_kernel_sm90.cuh"
#include "ipc_multiproc.cuh"

#include "cutlass/device_kernel.h"

#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// kernel types
// ---------------------------------------------------------------------------

using BaselineCfg = cc::GemmCfgTma;  // stock cooperative GEMM (comm_none)
using CollectiveEpilogueRs = typename BaselineCfg::CollectiveEpilogue;

using RsDma = rsutil::Sm90ReduceScatterDma<
    /*Stages=*/1,  // flux uses 1 DMA stage for the cooperative schedule
    typename BaselineCfg::TileShape,
    typename CollectiveEpilogueRs::EpilogueTile,
    typename CollectiveEpilogueRs::SmemLayoutAtomD,
    cc::ElementC,
    typename CollectiveEpilogueRs::StrideD>;

// mainloop rebuilt with the DMA smem carved out on top of the epilogue's
using CollectiveMainloopRs =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cc::ElementAB, cc::LayoutA, cc::kAlign,
        cc::ElementAB, cc::LayoutB, cc::kAlign,
        cc::ElementAcc,
        typename BaselineCfg::TileShape, typename BaselineCfg::ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogueRs::SharedStorage) +
            sizeof(typename RsDma::TensorStorage) +
            sizeof(typename RsDma::PipelineStorage) + 64)>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using RsKernel = cutlass::gemm::kernel::Sm90GemmRsKernel<
    cute::Shape<int, int, int>, CollectiveMainloopRs, CollectiveEpilogueRs,
    RsDma>;

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct Config {
  int m = 8192, n = 8192, k = 8192;
  int iters = 50;
  int warmup = 5;
  int ndev = 0;
  bool verify = false;
  std::string csv;
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --m/--n/--k N   GEMM shape per rank (default 8192^3)\n"
      "                  M %% (tile_M*world) == 0, N %% tile_N == 0 required\n"
      "  --iters N       timed iterations (default 50)\n"
      "  --warmup N      warmup iterations (default 5)\n"
      "  --ndev N        use only the first N GPUs (default: all)\n"
      "  --verify        check the reduce-scatter result numerically\n"
      "  --csv PATH      append machine-readable rows (rank 0 writes)\n",
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
    if (a == "--m") c.m = std::atoi(next().c_str());
    else if (a == "--n") c.n = std::atoi(next().c_str());
    else if (a == "--k") c.k = std::atoi(next().c_str());
    else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--warmup") c.warmup = std::atoi(next().c_str());
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
  return c;
}

// ---------------------------------------------------------------------------
// RS kernel launcher (mini adapter: params held, cluster launch)
// ---------------------------------------------------------------------------

struct RsRunner {
  typename RsKernel::Params params;
  void* workspace = nullptr;
  dim3 grid, block, cluster;
  int smem = 0;

  void init(typename RsKernel::Arguments const& args) {
    if (!RsKernel::can_implement(args)) {
      std::fprintf(stderr, "RsKernel::can_implement failed\n");
      std::exit(1);
    }
    const size_t ws = RsKernel::get_workspace_size(args);
    if (ws) CUDA_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK(RsKernel::initialize_workspace(args, workspace));
    params = RsKernel::to_underlying_arguments(args, workspace);
    smem = RsKernel::SharedStorageSize;
    if (smem >= (48 << 10)) {
      CUDA_CHECK(cudaFuncSetAttribute(
          cutlass::device_kernel<RsKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    block = RsKernel::get_block_shape();
    grid = RsKernel::get_grid_shape(params);
    using CS = typename BaselineCfg::ClusterShape;
    cluster = dim3(cute::size<0>(CS{}), cute::size<1>(CS{}), cute::size<2>(CS{}));
  }

  void run(cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster.x;
    attrs[0].val.clusterDim.y = cluster.y;
    attrs[0].val.clusterDim.z = cluster.z;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, cutlass::device_kernel<RsKernel>,
                                  params));
  }
};

// ---------------------------------------------------------------------------
// verification
// ---------------------------------------------------------------------------

struct HalfPtrs {
  const __half* p[mproc::kMaxWorld];
};

static __global__ void verify_kernel(HalfPtrs douts, const __half* reduce_buf,
                                     int world, int rank, long long m_seg,
                                     long long N, float* max_diff) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = m_seg * N;
  const long long step = (long long)gridDim.x * blockDim.x;
  float local_max = 0.f;
  for (; i < total; i += step) {
    const long long row = i / N, col = i % N;
    float ref = 0.f;
    for (int p = 0; p < world; ++p) {
      ref += __half2float(douts.p[p][(rank * m_seg + row) * N + col]);
    }
    const float got = __half2float(reduce_buf[i]);
    local_max = fmaxf(local_max, fabsf(ref - got));
  }
  // positive floats compare like their bit patterns
  atomicMax(reinterpret_cast<int*>(max_diff), __float_as_int(local_max));
}

// ---------------------------------------------------------------------------
// timing helper
// ---------------------------------------------------------------------------

template <class LaunchFn>
static Stats time_iters(const Config& cfg, mproc::MultiProc& mp,
                        const mproc::SyncPtrs& sync, cudaStream_t stream,
                        LaunchFn&& launch) {
  std::vector<cudaEvent_t> eb(cfg.iters), ee(cfg.iters);
  for (int i = 0; i < cfg.iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&eb[i]));
    CUDA_CHECK(cudaEventCreate(&ee[i]));
  }
  for (int i = 0; i < cfg.warmup; ++i) {
    mproc::barrier_all_on_stream(sync, mp.rank, mp.world, stream);
    launch();
  }
  for (int i = 0; i < cfg.iters; ++i) {
    mproc::barrier_all_on_stream(sync, mp.rank, mp.world, stream);
    CUDA_CHECK(cudaEventRecord(eb[i], stream));
    launch();
    CUDA_CHECK(cudaEventRecord(ee[i], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> per(cfg.iters);
  for (int i = 0; i < cfg.iters; ++i) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, eb[i], ee[i]));
    per[i] = ms * 1000.0;
    cudaEventDestroy(eb[i]);
    cudaEventDestroy(ee[i]);
  }
  return make_stats(per);
}

// ---------------------------------------------------------------------------
// main (every rank runs this; rank 0 prints)
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  // probe in a throwaway child: the parent must not touch CUDA before fork
  int world = mproc::probe_device_count();
  if (cfg.ndev > 0) world = std::min(world, cfg.ndev);
  if (world < 2 || (world & (world - 1))) {
    std::fprintf(stderr, "need a power-of-2 world >= 2, have %d GPUs\n", world);
    return 1;
  }
  constexpr int TILE_M = cute::size<0>(typename BaselineCfg::TileShape{});
  constexpr int TILE_N = cute::size<1>(typename BaselineCfg::TileShape{});
  if (cfg.m % (TILE_M * world) || cfg.n % TILE_N || cfg.k % cc::kAlign) {
    std::fprintf(stderr,
                 "shape %dx%dx%d invalid: need M %% %d == 0, N %% %d == 0, "
                 "K %% %d == 0\n",
                 cfg.m, cfg.n, cfg.k, TILE_M * world, TILE_N, cc::kAlign);
    return 1;
  }

  mproc::MultiProc mp;
  mp.init(world);
  const int rank = mp.rank;
  CUDA_CHECK(cudaSetDevice(rank));

  const int64_t M = cfg.m, N = cfg.n, K = cfg.k;
  const int m_seg = cfg.m / world;
  const int m_tiles = cfg.m / TILE_M, n_tiles = (cfg.n + TILE_N - 1) / TILE_N;
  const double flop = 2.0 * M * N * K;

  // ---- buffers -------------------------------------------------------------
  void *A = nullptr, *B = nullptr;
  cc::ElementC *D_out = nullptr, *reduce_buf = nullptr;
  int *flags = nullptr, *sync_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&A, M * K * 2));
  CUDA_CHECK(cudaMalloc(&B, K * N * 2));
  CUDA_CHECK(cudaMalloc(&D_out, M * N * 2));
  CUDA_CHECK(cudaMalloc(&reduce_buf, (int64_t)m_seg * N * 2));
  CUDA_CHECK(cudaMalloc(&flags, sizeof(int) * 2 * m_tiles * n_tiles));
  CUDA_CHECK(cudaMalloc(&sync_buf, sizeof(int) * mproc::kMaxWorld));
  fill_half(A, M * K, 1000 + rank);
  fill_half(B, K * N, 77 + rank);
  CUDA_CHECK(cudaMemset(D_out, 0, M * N * 2));
  CUDA_CHECK(cudaMemset(reduce_buf, 0, (int64_t)m_seg * N * 2));
  CUDA_CHECK(cudaMemset(flags, 0, sizeof(int) * 2 * m_tiles * n_tiles));
  CUDA_CHECK(cudaMemset(sync_buf, 0, sizeof(int) * mproc::kMaxWorld));
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- IPC exchange --------------------------------------------------------
  void* dout_ptrs_v[mproc::kMaxWorld] = {};
  void* flag_ptrs_v[mproc::kMaxWorld] = {};
  void* sync_ptrs_v[mproc::kMaxWorld] = {};
  mp.exchange(0, D_out, dout_ptrs_v);
  mp.exchange(1, flags, flag_ptrs_v);
  mp.exchange(2, sync_buf, sync_ptrs_v);

  cc::ElementC* dout_ptrs[mproc::kMaxWorld];
  int* flag_ptrs[mproc::kMaxWorld];
  mproc::SyncPtrs sync = {};
  for (int r = 0; r < world; ++r) {
    dout_ptrs[r] = (cc::ElementC*)dout_ptrs_v[r];
    flag_ptrs[r] = (int*)flag_ptrs_v[r];
    sync.p[r] = (int*)sync_ptrs_v[r];
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // ---- baseline: stock cooperative GEMM, same config ----------------------
  cc::GemmOp<BaselineCfg> baseline;
  baseline.init(cfg.m, cfg.n, cfg.k, A, B, D_out, D_out, 1.0f, 0.0f, rank);

  // ---- fused GEMM+RS -------------------------------------------------------
  using StrideA = typename RsKernel::StrideA;
  using StrideB = typename RsKernel::StrideB;
  using StrideC = typename RsKernel::StrideC;
  using StrideD = typename RsKernel::StrideD;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {cfg.m, cfg.k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {cfg.n, cfg.k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {cfg.m, cfg.n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {cfg.m, cfg.n, 1});

  cutlass::KernelHardwareInfo hw;
  hw.device_id = rank;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(rank);

  typename RsKernel::Arguments rs_args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {cfg.m, cfg.n, cfg.k},
      {reinterpret_cast<const cc::ElementAB*>(A), stride_A,
       reinterpret_cast<const cc::ElementAB*>(B), stride_B},
      {{1.0f, 0.0f}, D_out, stride_C, D_out, stride_D},
      hw,
      {/*max_swizzle_size=*/1,
       cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::
           RasterOrderOptions::AlongN},
      {dout_ptrs, stride_D, rank, world, reduce_buf, flag_ptrs}};

  RsRunner fused;
  fused.init(rs_args);

  if (rank == 0) {
    std::printf("=== exp3 v2: fused GEMM+RS (flux sm90 design) vs stock GEMM ===\n");
    std::printf("world %d (1 process per GPU, cudaIPC), %dx%dx%d fp16 per rank\n",
                world, cfg.m, cfg.n, cfg.k);
    std::printf("tile 128x256x64 cluster 1x2x1, mainloop stages: baseline %d, "
                "fused %d (DMA smem carveout)\n",
                (int)BaselineCfg::CollectiveMainloop::DispatchPolicy::Stages,
                (int)CollectiveMainloopRs::DispatchPolicy::Stages);
    std::printf("RS pull volume per rank: %.1f MiB, grid %d blocks\n\n",
                (double)(world - 1) / world * M * N * 2 / (1 << 20),
                fused.grid.x * fused.grid.y * fused.grid.z);
    std::fflush(stdout);
  }
  mp.barrier();

  // ---- timing --------------------------------------------------------------
  const Stats base_st = time_iters(cfg, mp, sync, stream,
                                   [&] { baseline.run(stream); });
  mp.barrier();
  const Stats fuse_st = time_iters(cfg, mp, sync, stream,
                                   [&] { fused.run(stream); });
  mp.barrier();

  // ---- verify --------------------------------------------------------------
  double max_diff = -1;
  if (cfg.verify) {
    // D_out on every rank currently holds that rank's full GEMM output from
    // the last fused iteration; the reduce buffer holds the fused RS result.
    HalfPtrs hp = {};
    for (int r = 0; r < world; ++r) hp.p[r] = (const __half*)dout_ptrs[r];
    float* d_diff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_diff, sizeof(float)));
    // async on the same stream: a legacy-stream memset would not order
    // against the non-blocking stream the kernel runs on
    CUDA_CHECK(cudaMemsetAsync(d_diff, 0, sizeof(float), stream));
    verify_kernel<<<256, 256, 0, stream>>>(hp, (const __half*)reduce_buf,
                                           world, rank, m_seg, N, d_diff);
    CUDA_CHECK(cudaGetLastError());
    float h_diff = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_diff, d_diff, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_diff);
    max_diff = h_diff;
  }

  // ---- report --------------------------------------------------------------
  mp.shm->results[rank][0] = base_st.mean;
  mp.shm->results[rank][1] = base_st.p95;
  mp.shm->results[rank][2] = fuse_st.mean;
  mp.shm->results[rank][3] = fuse_st.p95;
  mp.shm->results[rank][4] = max_diff;
  mp.barrier();

  if (rank == 0) {
    std::printf("%4s | %12s %8s | %12s %8s | %8s %8s %9s | %s\n", "rank",
                "base us", "TFLOP/s", "fused us", "TFLOP/s", "slowdn",
                "ovhd us", "pull GB/s", "verify");
    FILE* csv = nullptr;
    if (!cfg.csv.empty()) {
      csv = std::fopen(cfg.csv.c_str(), "a");
      if (csv)
        std::fprintf(csv,
                     "m,n,k,world,rank,base_us,base_p95,fused_us,fused_p95,"
                     "slowdown,pull_gbps,max_diff\n");
    }
    double worst = 0;
    for (int r = 0; r < world; ++r) {
      const double b = mp.shm->results[r][0], f = mp.shm->results[r][2];
      const double slow = b > 0 ? f / b : 0;
      worst = std::max(worst, slow);
      const double pull_gbps =
          (double)(world - 1) / world * M * N * 2 / (f * 1e-6) / 1e9;
      const double vd = mp.shm->results[r][4];
      char ver[32];
      if (!cfg.verify) std::snprintf(ver, sizeof(ver), "-");
      else std::snprintf(ver, sizeof(ver), "%s (%.3g)",
                         vd <= 0.25 ? "ok" : "FAIL", vd);
      std::printf("%4d | %12.1f %8.1f | %12.1f %8.1f | %8.3f %8.1f %9.1f | %s\n",
                  r, b, flop / (b * 1e-6) / 1e12, f, flop / (f * 1e-6) / 1e12,
                  slow, f - b, pull_gbps, ver);
      if (csv)
        std::fprintf(csv, "%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.2f,%.4g\n",
                     cfg.m, cfg.n, cfg.k, world, r, b, mp.shm->results[r][1],
                     f, mp.shm->results[r][3], slow, pull_gbps, vd);
    }
    if (csv) std::fclose(csv);
    std::printf(
        "\nfused time covers compute + NVLink pulls + fp16 reduction; the\n"
        "baseline is the identical stock GEMM run symmetrically on all "
        "ranks.\nworst-rank slowdown: %.3f. flux pays this to keep the "
        "epilogue local\ninstead of remote-writing (comm_comp v1 exp3 "
        "measures that other option).\n",
        worst);
  }
  mp.barrier();

  // ---- cleanup -------------------------------------------------------------
  for (int r = 0; r < world; ++r) {
    if (r != rank) {
      CUDA_CHECK(cudaIpcCloseMemHandle(dout_ptrs_v[r]));
      CUDA_CHECK(cudaIpcCloseMemHandle(flag_ptrs_v[r]));
      CUDA_CHECK(cudaIpcCloseMemHandle(sync_ptrs_v[r]));
    }
  }
  baseline.destroy();
  cudaStreamDestroy(stream);
  cudaFree(A); cudaFree(B); cudaFree(D_out); cudaFree(reduce_buf);
  cudaFree(flags); cudaFree(sync_buf);

  if (rank == 0) return mp.wait_children();
  return 0;
}
