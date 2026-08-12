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
//   ctrl:     the IDENTICAL RS kernel with communication disabled at runtime
//             (fetch/reduce warps idle, no flag publishes). Same static
//             scheduler, same XOR swizzle, same smem carveout / mainloop
//             stage count as fused, so ctrl/baseline prices the kernel's
//             STRUCTURAL changes and fused/ctrl prices the COMMUNICATION
//             alone -- without this split the two are confounded
//   fused:    the GEMM+RS kernel; its time covers compute + all pulls +
//             reduction
//
//   Per rank: struct = t_ctrl/t_base, comm = t_fused/t_ctrl, and
//   total = t_fused/t_base = struct * comm. Also reported: effective pull
//   bandwidth (world-1)/world * M*N*2 / t_fused, and achieved TFLOP/s.
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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

// shm scratch slot for agreeing a warmup iteration count across ranks; the
// reported results occupy 0..12 of the 16 available
static constexpr int kWarmupSlot = 13;

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
  double warmup_ms = 0;  // wall-clock warmup floor, on top of `warmup`
  int ndev = 0;
  bool verify = false;
  std::string csv;
  std::string dump;                       // per-iteration dump prefix
  std::string order = "base,ctrl,fused";  // timing order (run-order control)
};

static void usage(const char* prog) {
  std::printf(
      "usage: %s [options]\n"
      "  --m/--n/--k N     GEMM shape per rank (default 8192^3)\n"
      "                    M %% (tile_M*world) == 0, N %% tile_N == 0 required\n"
      "  --iters N         timed iterations (default 50)\n"
      "  --warmup N        warmup iterations (default 5)\n"
      "  --warmup-ms MS    keep warming until MS milliseconds have elapsed, on\n"
      "                    top of --warmup. Iteration counts are the wrong unit\n"
      "                    for warmup: 30 iterations is ~0.7 s at M=32768 but\n"
      "                    ~7 ms at M=512, so a fixed count leaves small shapes\n"
      "                    effectively unwarmed and overstates their cost\n"
      "  --ndev N          use only the first N GPUs (default: all)\n"
      "  --verify          check the reduce-scatter result numerically\n"
      "  --csv PATH        append machine-readable rows (rank 0 writes)\n"
      "  --dump-iters PRE  raw per-iteration times -> PRE.rank<N>.csv, every\n"
      "                    rank. mean+p95 cannot tell a distribution shift\n"
      "                    from three slow iterations; the series can\n"
      "  --order LIST      permutation of base,ctrl,fused fixing the order the\n"
      "                    three variants are timed in (default\n"
      "                    base,ctrl,fused). Whatever runs first absorbs any\n"
      "                    residual ramp-up, which biases struct_sd=ctrl/base;\n"
      "                    run the reverse to price that bias\n",
      prog);
}

// Split --order into the three variant names, rejecting anything that is not
// a permutation (a typo that silently dropped a variant would leave its Stats
// zeroed and every ratio nonsense).
static std::vector<std::string> parse_order(const std::string& spec) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= spec.size()) {
    const size_t comma = spec.find(',', pos);
    const size_t end = comma == std::string::npos ? spec.size() : comma;
    out.push_back(spec.substr(pos, end - pos));
    if (comma == std::string::npos) break;
    pos = end + 1;
  }
  std::vector<std::string> want = {"base", "ctrl", "fused"};
  std::vector<std::string> got = out;
  std::sort(got.begin(), got.end());
  if (got != want) {
    std::fprintf(stderr,
                 "--order '%s' is not a permutation of base,ctrl,fused\n",
                 spec.c_str());
    std::exit(1);
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
    if (a == "--m") c.m = std::atoi(next().c_str());
    else if (a == "--n") c.n = std::atoi(next().c_str());
    else if (a == "--k") c.k = std::atoi(next().c_str());
    else if (a == "--iters") c.iters = std::atoi(next().c_str());
    else if (a == "--warmup") c.warmup = std::atoi(next().c_str());
    else if (a == "--warmup-ms") c.warmup_ms = std::atof(next().c_str());
    else if (a == "--ndev") c.ndev = std::atoi(next().c_str());
    else if (a == "--verify") c.verify = true;
    else if (a == "--csv") c.csv = next();
    else if (a == "--dump-iters") c.dump = next();
    else if (a == "--order") c.order = next();
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

// Reports the max RELATIVE deviation |ref - got| / (|ref| + 1). The kernel
// accumulates in fp16 (red.global.add), the reference in fp32, so the
// legitimate gap is ~world ulps of the running sum -- relative ~world*2^-11.
// An absolute tolerance is a trap: it silently couples to the data scale.
static __global__ void verify_kernel(HalfPtrs douts, const __half* reduce_buf,
                                     int world, int rank, long long m_seg,
                                     long long N, float* max_rel) {
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
    local_max = fmaxf(local_max, fabsf(ref - got) / (fabsf(ref) + 1.0f));
  }
  // positive floats compare like their bit patterns
  atomicMax(reinterpret_cast<int*>(max_rel), __float_as_int(local_max));
}

// ---------------------------------------------------------------------------
// timing helper
// ---------------------------------------------------------------------------

// per_out (optional) receives the raw per-iteration series in launch order --
// make_stats sorts its copy, so the time axis is lost after this point.
template <class LaunchFn>
static Stats time_iters(const Config& cfg, mproc::MultiProc& mp,
                        const mproc::SyncPtrs& sync, cudaStream_t stream,
                        LaunchFn&& launch,
                        std::vector<double>* per_out = nullptr) {
  std::vector<cudaEvent_t> eb(cfg.iters), ee(cfg.iters);
  for (int i = 0; i < cfg.iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&eb[i]));
    CUDA_CHECK(cudaEventCreate(&ee[i]));
  }
  for (int i = 0; i < cfg.warmup; ++i) {
    mproc::barrier_all_on_stream(sync, mp.rank, mp.world, stream);
    launch();
  }
  if (cfg.warmup_ms > 0) {
    // Every rank must run the SAME number of warmup iterations: they
    // synchronise on-stream inside each one, so a rank that stopped early on
    // its own clock would hang the rest. Time one iteration, then agree on
    // rank 0's figure via shm before looping.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto t0 = std::chrono::steady_clock::now();
    mproc::barrier_all_on_stream(sync, mp.rank, mp.world, stream);
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    mp.shm->results[mp.rank][kWarmupSlot] =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    mp.barrier();
    const double ref = mp.shm->results[0][kWarmupSlot];
    const int extra = ref > 0 ? (int)(cfg.warmup_ms / ref) : 0;
    mp.barrier();
    for (int i = 0; i < extra; ++i) {
      mproc::barrier_all_on_stream(sync, mp.rank, mp.world, stream);
      launch();
    }
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
  if (per_out) *per_out = per;
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

  // RS-off control: same kernel, same params, communication switched off
  typename RsKernel::Arguments ctrl_args = rs_args;
  ctrl_args.rs_dma.comm_enabled = false;
  RsRunner ctrl;
  ctrl.init(ctrl_args);

  if (rank == 0) {
    std::printf("=== exp3 v2: fused GEMM+RS (flux sm90 design) vs stock GEMM ===\n");
    std::printf("world %d (1 process per GPU, cudaIPC), %dx%dx%d fp16 per rank\n",
                world, cfg.m, cfg.n, cfg.k);
    std::printf("tile 128x256x64 cluster 1x2x1, mainloop stages: baseline %d, "
                "ctrl/fused %d (DMA smem carveout)\n",
                (int)BaselineCfg::CollectiveMainloop::DispatchPolicy::Stages,
                (int)CollectiveMainloopRs::DispatchPolicy::Stages);
    // flux forces epilogue StagesD=1 to buy back smem for the mainloop; we
    // keep the builder default on baseline+ctrl+fused alike, so the ctrl
    // column absorbs whatever that choice costs.
    std::printf("epilogue StagesC %d StagesD %d (builder default; flux "
                "gemm_v3_reduce_scatter forces StagesD=1)\n",
                (int)CollectiveEpilogueRs::DispatchPolicy::StagesC,
                (int)CollectiveEpilogueRs::DispatchPolicy::StagesD);
    std::printf("RS pull volume per rank: %.1f MiB, grid %d blocks\n\n",
                (double)(world - 1) / world * M * N * 2 / (1 << 20),
                fused.grid.x * fused.grid.y * fused.grid.z);
    std::fflush(stdout);
  }
  mp.barrier();

  // ---- timing --------------------------------------------------------------
  // Order is configurable (--order). It matters: whichever variant runs first
  // absorbs any residual clock ramp or first-touch cost, and with the default
  // base,ctrl,fused that lands entirely on `base` -- which inflates base and
  // so pushes struct_sd=ctrl/base BELOW 1. Running the reverse order prices
  // that bias instead of assuming it away.
  const std::vector<std::string> order = parse_order(cfg.order);
  Stats base_st, ctrl_st, fuse_st;
  std::vector<double> base_per, ctrl_per, fuse_per;
  for (const std::string& which : order) {
    if (which == "base")
      base_st = time_iters(cfg, mp, sync, stream,
                           [&] { baseline.run(stream); }, &base_per);
    else if (which == "ctrl")
      ctrl_st = time_iters(cfg, mp, sync, stream,
                           [&] { ctrl.run(stream); }, &ctrl_per);
    else
      fuse_st = time_iters(cfg, mp, sync, stream,
                           [&] { fused.run(stream); }, &fuse_per);
    mp.barrier();
  }

  // ---- per-iteration dump --------------------------------------------------
  // mean and p95 together cannot separate "the whole distribution moved" from
  // "3 of 50 iterations were slow" -- and the CSV's *_us columns are means, so
  // a fat tail leaks straight into every reported ratio. Every rank writes its
  // own file: the fused kernel waits on the slowest peer, so a tail on one
  // rank is only interpretable next to the other three.
  if (!cfg.dump.empty()) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s.rank%d.csv", cfg.dump.c_str(), rank);
    FILE* df = std::fopen(path, "w");
    if (!df) {
      std::fprintf(stderr, "rank %d: cannot write %s\n", rank, path);
    } else {
      std::fprintf(df, "m,n,k,world,rank,variant,iter,us\n");
      auto emit = [&](const char* name, const std::vector<double>& v) {
        for (size_t i = 0; i < v.size(); ++i)
          std::fprintf(df, "%d,%d,%d,%d,%d,%s,%zu,%.3f\n", cfg.m, cfg.n, cfg.k,
                       world, rank, name, i, v[i]);
      };
      emit("base", base_per);
      emit("ctrl", ctrl_per);
      emit("fused", fuse_per);
      std::fclose(df);
    }
  }

  // ---- verify --------------------------------------------------------------
  // verify reads D_out (the last GEMM output) and reduce_buf (the RS result),
  // which the default order leaves in place because fused runs last. Under a
  // custom --order a later base/ctrl pass has since overwritten D_out, so
  // refresh both with one untimed fused iteration first.
  if (cfg.verify && order.back() != "fused") {
    mproc::barrier_all_on_stream(sync, mp.rank, mp.world, stream);
    fused.run(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    mp.barrier();
  }
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
  mp.shm->results[rank][5] = ctrl_st.mean;
  mp.shm->results[rank][6] = ctrl_st.p95;
  // p50 and min are computed by make_stats and were previously dropped. p50 is
  // the tail-immune view of the same ratio: if comm_sd rises on means but not
  // on p50s, the rise is outliers, not cost.
  mp.shm->results[rank][7] = base_st.p50;
  mp.shm->results[rank][8] = ctrl_st.p50;
  mp.shm->results[rank][9] = fuse_st.p50;
  mp.shm->results[rank][10] = base_st.mn;
  mp.shm->results[rank][11] = ctrl_st.mn;
  mp.shm->results[rank][12] = fuse_st.mn;
  mp.barrier();

  if (rank == 0) {
    std::printf("%4s | %12s %8s | %10s | %12s %8s | %6s %6s %6s | %9s | %s\n",
                "rank", "base us", "TFLOP/s", "ctrl us", "fused us", "TFLOP/s",
                "struct", "comm", "total", "pull GB/s", "verify");
    FILE* csv = nullptr;
    if (!cfg.csv.empty()) {
      csv = std::fopen(cfg.csv.c_str(), "a");
      if (csv)
        std::fprintf(csv,
                     "m,n,k,world,rank,base_us,base_p95,ctrl_us,ctrl_p95,"
                     "fused_us,fused_p95,struct_sd,comm_sd,total_sd,"
                     "pull_gbps,max_rel_diff,"
                     // appended (old columns keep their position/meaning)
                     "base_p50,ctrl_p50,fused_p50,base_min,ctrl_min,fused_min,"
                     "comm_sd_p50,comm_sd_skew,comm_abs_us,order,iters\n");
    }
    // The fused kernel cannot finish before the SLOWEST peer has produced its
    // tiles, so fused/ctrl_self charges a rank for its peers' spread. Dividing
    // by max_r(ctrl_r) instead gives the skew-normalised tax.
    double ctrl_max = 0;
    for (int r = 0; r < world; ++r)
      ctrl_max = std::max(ctrl_max, mp.shm->results[r][5]);
    double worst_total = 0, worst_comm = 0;
    for (int r = 0; r < world; ++r) {
      const double b = mp.shm->results[r][0], f = mp.shm->results[r][2];
      const double c = mp.shm->results[r][5];
      const double sd_struct = b > 0 ? c / b : 0;  // kernel structure alone
      const double sd_comm = c > 0 ? f / c : 0;    // communication alone
      const double sd_total = b > 0 ? f / b : 0;
      worst_total = std::max(worst_total, sd_total);
      worst_comm = std::max(worst_comm, sd_comm);
      const double pull_gbps =
          (double)(world - 1) / world * M * N * 2 / (f * 1e-6) / 1e9;
      const double vd = mp.shm->results[r][4];
      char ver[32];
      // fp16 chain of `world` adds rounds at the ulp of the LARGEST running
      // partial, so elements whose final sum is small can see a few percent
      // relative deviation; a real protocol bug shows up as rel ~ 1.
      if (!cfg.verify) std::snprintf(ver, sizeof(ver), "-");
      else std::snprintf(ver, sizeof(ver), "%s (rel %.3g)",
                         vd <= 0.05 ? "ok" : "FAIL", vd);
      std::printf("%4d | %12.1f %8.1f | %10.1f | %12.1f %8.1f | %6.3f %6.3f "
                  "%6.3f | %9.1f | %s\n",
                  r, b, flop / (b * 1e-6) / 1e12, c, f,
                  flop / (f * 1e-6) / 1e12, sd_struct, sd_comm, sd_total,
                  pull_gbps, ver);
      if (csv) {
        const double bp50 = mp.shm->results[r][7];
        const double cp50 = mp.shm->results[r][8];
        const double fp50 = mp.shm->results[r][9];
        // --order is comma-separated, which would split into extra CSV
        // columns and shift every field after it; emit it pipe-separated
        std::string order_csv = cfg.order;
        std::replace(order_csv.begin(), order_csv.end(), ',', '|');
        std::fprintf(csv,
                     "%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,"
                     "%.4f,%.2f,%.4g,"
                     "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.2f,%s,%d\n",
                     cfg.m, cfg.n, cfg.k, world, r, b, mp.shm->results[r][1],
                     c, mp.shm->results[r][6], f, mp.shm->results[r][3],
                     sd_struct, sd_comm, sd_total, pull_gbps, vd,
                     bp50, cp50, fp50, mp.shm->results[r][10],
                     mp.shm->results[r][11], mp.shm->results[r][12],
                     cp50 > 0 ? fp50 / cp50 : 0,
                     ctrl_max > 0 ? f / ctrl_max : 0,
                     // absolute comm cost in us: the only cross-shape
                     // comparable metric, since comm_sd's denominator moves
                     // with K
                     f - c, order_csv.c_str(), cfg.iters);
      }
    }
    if (csv) std::fclose(csv);

    // Second view of the same three numbers, immune to the two things that
    // make the mean-based table hard to read at large M: outlier iterations
    // and inter-rank spread.
    std::printf("\n%4s | %10s %10s %10s | %8s %8s %8s | %9s\n", "rank",
                "base p50", "ctrl p50", "fusd p50", "comm p50", "comm avg",
                "comm skew", "comm abs us");
    for (int r = 0; r < world; ++r) {
      const double bp = mp.shm->results[r][7], cp = mp.shm->results[r][8];
      const double fp = mp.shm->results[r][9];
      const double c = mp.shm->results[r][5], f = mp.shm->results[r][2];
      std::printf("%4d | %10.1f %10.1f %10.1f | %8.3f %8.3f %8.3f | %9.1f\n", r,
                  bp, cp, fp, cp > 0 ? fp / cp : 0, c > 0 ? f / c : 0,
                  ctrl_max > 0 ? f / ctrl_max : 0, f - c);
    }
    std::printf(
        "comm p50  = fused_p50/ctrl_p50: tail-immune. If it stays flat while\n"
        "            comm avg rises, the rise is outlier iterations, not cost.\n"
        "comm skew = fused_mean/max_r(ctrl_mean): charges each rank only for\n"
        "            comm, not for its peers' spread (fused waits on the\n"
        "            slowest producer either way).\n"
        "comm abs  = fused-ctrl in us. Compare THIS across shapes -- comm_sd's\n"
        "            denominator moves with K, so ratios are not comparable\n"
        "            once K changes.\n"
        "timing order: %s (%d iters, %d warmup + %.0f ms)\n\n",
        cfg.order.c_str(), cfg.iters, cfg.warmup, cfg.warmup_ms);

    std::printf(
        "struct = ctrl/base: cost of the kernel restructuring alone (static\n"
        "scheduler, swizzle, smem carveout; comm switched off).\n"
        "comm = fused/ctrl: cost of the communication alone (flag publishes,\n"
        "NVLink pulls, fp16 reduction, slowest-peer tail).\n"
        "worst-rank: total %.3f, comm %.3f. flux pays this to keep the\n"
        "epilogue local instead of remote-writing (v1 exp3 measures that "
        "option).\n",
        worst_total, worst_comm);
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
