// gemm_sm90.cuh
//
// Thin wrapper around a CUTLASS 3.x-API sm90 (Hopper) fp16 GEMM, built with
// the CollectiveBuilder the same way flux's sm90 path does:
//   * mainloop: TMA warp-specialized cooperative (KernelTmaWarpSpecializedCooperative)
//   * epilogue: TMA store (TmaWarpSpecializedCooperative) by default, with a
//     NoSmemWarpSpecialized variant for exp3 that stores D straight from the
//     accumulator layout (no smem restage, narrow per-thread transactions).
//     It separates "TMA store to remote memory" from "plain global stores to
//     remote memory"; note its stores are NOT 128-bit vectorized -- compare
//     remote vs local within the same epilogue, not tma vs nosmem absolutes.
//
// D = alpha * A @ B + beta * C, fp16 in / fp32 accumulate / fp16 out,
// A row-major [M,K], B column-major [K,N] (i.e. N-major buffer of K*N),
// C/D row-major [M,N].
//
// A GemmOp is initialized ONCE for a fixed (shape, pointers) tuple and then
// launched repeatedly with run(stream). This keeps host-side argument
// canonicalization and TMA-descriptor construction out of the timed loops --
// important for exp2, where fine-grained segmentation would otherwise measure
// host overhead instead of wave quantization.

#pragma once

#include "common.cuh"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"

namespace cc {

using ElementAB  = cutlass::half_t;
using ElementC   = cutlass::half_t;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;     // A [M,K]
using LayoutB = cutlass::layout::ColumnMajor;  // B [K,N]
using LayoutC = cutlass::layout::RowMajor;     // C/D [M,N]
constexpr int kAlign = 8;                      // 16B / sizeof(half)

template <class KernelSchedule, class EpilogueSchedule>
struct GemmTypes {
  // 128x256x64 fp16 cooperative, cluster 1x2x1 -- the config family flux's
  // tuned H800 tables use. Cluster along N (not M) so that exp2's small-M
  // segments don't waste cluster slots.
  using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, ElementAcc,
          ElementC, LayoutC, kAlign,
          ElementC, LayoutC, kAlign,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          ElementAB, LayoutA, kAlign,
          ElementAB, LayoutB, kAlign,
          ElementAcc,
          TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Default: TMA-store epilogue (what flux's sm90 GEMMs use).
using GemmCfgTma = GemmTypes<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                             cutlass::epilogue::TmaWarpSpecializedCooperative>;
// exp3 variant: epilogue writes D with plain global stores, no smem staging.
using GemmCfgNoSmem =
    GemmTypes<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
              cutlass::epilogue::NoSmemWarpSpecialized>;

// One pre-initialized GEMM launch. All pointers may be UVA peer pointers --
// on Hopper both TMA and regular stores accept peer-mapped addresses, which
// is exactly what exp3 exercises.
template <class Cfg>
struct GemmOp {
  using Gemm = typename Cfg::Gemm;
  Gemm op;
  void* workspace = nullptr;

  void init(int M, int N, int K, const void* A, const void* B, const void* C,
            void* D, float alpha, float beta, int dev) {
    CUDA_CHECK(cudaSetDevice(dev));  // workspace + TMA setup on the right GPU
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw;
    hw.device_id = dev;
    hw.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {reinterpret_cast<const ElementAB*>(A), stride_A,
         reinterpret_cast<const ElementAB*>(B), stride_B},
        {{alpha, beta},
         reinterpret_cast<const ElementC*>(C), stride_C,
         reinterpret_cast<ElementC*>(D), stride_D},
        hw};

    CUTLASS_CHECK(Gemm::can_implement(args));
    const size_t ws = Gemm::get_workspace_size(args);
    if (ws) CUDA_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK(op.initialize(args, workspace));
  }

  void run(cudaStream_t stream) { CUTLASS_CHECK(op.run(stream)); }

  void destroy() {
    if (workspace) {
      cudaFree(workspace);
      workspace = nullptr;
    }
  }
};

}  // namespace cc
