// rs_dma_sm90.cuh
//
// Sm90ReduceScatterDma, ported from flux src/gemm_rs/sm90_reduce_scatter_utils.hpp
// (intra-node, fuse-reduction path only; nnodes folded to 1, nvshmem and the
// non-fused path stripped). This is the communication half of flux's sm90
// GEMM+RS: it runs in two otherwise-idle warps of the producer warpgroup of
// every CTA and, following the same tile stream as the compute warps,
//
//   fetch  (1 warp): for the work tile at rows [m], figure out which rank's
//          output segment it belongs to (src_rank = m / tile_m_perrank),
//          spin on that PEER's "epilogue done" flag for the corresponding
//          tile of OUR segment, then TMA-LOAD the peer's tile over NVLink
//          into a smem pipeline;
//   reduce (1 warp): drain the pipeline into registers and combine into the
//          LOCAL reduce buffer [M/world, N]: plain store for our own tile
//          (the base value), red.global.add.v8.f16 for peer tiles, with a
//          gpu-scope counter serializing base-store before adds.
//
// The epilogue itself never writes remote memory -- that is the design
// decision exp3 measures.

#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/epilogue/collective/detail.hpp"

#include "rs_barrier.cuh"

#include <cstdio>
#include <cstdlib>

namespace rsutil {

using namespace cute;

#define RS_CHECK(cond)                                                     \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "[rs_dma] %s:%d: check failed: %s\n", __FILE__, \
                   __LINE__, #cond);                                       \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static constexpr int kMaxWorld = 8;

template <
    int Stages,
    class TileShape_,
    class EpilogueTile_,
    class SmemLayoutAtom_,
    class Element_,
    class StrideMNL_>
struct Sm90ReduceScatterDma {
 public:
  using TileShape = TileShape_;
  using EpilogueTile = EpilogueTile_;
  using SmemLayoutAtom = SmemLayoutAtom_;
  using Element = Element_;
  using StrideMNL = StrideMNL_;
  static constexpr int kAlignment = 128 / sizeof_bits_v<Element>;

  constexpr static bool is_m_major =
      cutlass::epilogue::collective::detail::is_m_major<StrideMNL>();
  static_assert(not is_m_major, "only n-major supported");
  using SmemShapeTma = decltype(make_shape(
      max_common_vector(make_layout(get<0>(EpilogueTile{})),
                        make_layout(get<0>(EpilogueTile{}))),
      max_common_vector(make_layout(get<1>(EpilogueTile{})),
                        make_layout(get<1>(EpilogueTile{})))));
  using SmemLayoutTma = decltype(tile_to_shape(
      SmemLayoutAtom{}, SmemShapeTma{}, Step<_1, _2>{}));
  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutTma{},
      make_shape(size<0>(shape(EpilogueTile{})), size<1>(shape(EpilogueTile{})),
                 Int<Stages>{}),
      Step<_1, _2, _3>{}));

  using FetchPipeline = cutlass::PipelineTransactionAsync<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;
  using PipelineParams = typename FetchPipeline::Params;
  static constexpr int TmaTransactionBytes =
      size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}) * sizeof(Element);

  static constexpr int ThreadCount = 32;

  struct Arguments {
    Element **output_scatter_ptrs;  // [world] IPC-mapped D pointers
    StrideMNL stride;               // D stride (same on every rank)
    int rank = 0;
    int world_size = 0;
    void *local_reduce_buffer = nullptr;  // [M/world, N] on this rank
    int **barrier_ptrs;                   // [world] IPC-mapped flag arrays
  };

  struct Params {
    using TMA_Fetch = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<Element const *>(nullptr),
                    repeat_like(StrideMNL{}, int32_t(0)), StrideMNL{}),
        SmemLayoutTma{}));

    int rank;
    int world_size;
    int tile_m_perrank;  // M tiles per rank segment

    StrideMNL stride;

    Element *local_ptr[kMaxWorld];
    TMA_Fetch tma_load_fetch[kMaxWorld];
    Element *local_reduce_buffer;
    int *local_barrier_ptr[kMaxWorld];
    Layout<Shape<int, int>> tile_layout;  // (m_tiles, n_tiles)
  };

  struct TensorStorage {
    alignas(cutlass::detail::alignment_for_swizzle(SmemLayout{}))
        array_aligned<Element, size(SmemLayout{})> tensor;
  };

  using PipelineStorage = typename FetchPipeline::SharedStorage;

  template <class ProblemShape>
  static Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args) {
    Params params;
    auto [M, N, K] = problem_shape;
    constexpr int L = 1;

    params.rank = args.rank;
    params.world_size = args.world_size;
    RS_CHECK(params.world_size <= kMaxWorld);
    RS_CHECK((params.world_size & (params.world_size - 1)) == 0);  // XOR swizzle

    auto [tile_M, tile_N, tile_K] = TileShape{};
    RS_CHECK(M % tile_M == 0);
    RS_CHECK(N % tile_N == 0);
    RS_CHECK(M % (tile_M * params.world_size) == 0);
    RS_CHECK(args.barrier_ptrs != nullptr);
    RS_CHECK(args.local_reduce_buffer != nullptr);

    params.tile_m_perrank = M / (tile_M * params.world_size);
    params.stride = args.stride;
    params.local_reduce_buffer = static_cast<Element *>(args.local_reduce_buffer);

    for (int r = 0; r < params.world_size; ++r) {
      Element *ptr = args.output_scatter_ptrs[r];
      int *barrier_ptr = args.barrier_ptrs[r];
      RS_CHECK(ptr != nullptr && barrier_ptr != nullptr);
      params.local_ptr[r] = ptr;
      params.local_barrier_ptr[r] = barrier_ptr;
      auto tensor_fetch =
          make_tensor(ptr, make_layout(make_shape(M, N, L), args.stride));
      params.tma_load_fetch[r] =
          make_tma_copy(SM90_TMA_LOAD{}, tensor_fetch, SmemLayoutTma{});
    }

    int m_tiles = ceil_div(M, size<0>(TileShape{}));
    int n_tiles = ceil_div(N, size<1>(TileShape{}));
    params.tile_layout = make_layout(make_shape(m_tiles, n_tiles));
    return params;
  }

  const Params *params_ptr;
  Element *smem_tensor;

  CUTLASS_HOST_DEVICE
  Sm90ReduceScatterDma() {}

  CUTLASS_HOST_DEVICE
  Sm90ReduceScatterDma(Params const &params, TensorStorage const &shared_tensor)
      : params_ptr(&params),
        smem_tensor(const_cast<Element *>(shared_tensor.tensor.data())) {}

  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  fetch(FetchPipeline fetch_pipeline, PipelineState fetch_write_state,
        ProblemShapeMNKL const &problem_shape, TileCoordMNKL const &tile_coord) {
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      return fetch_write_state;  // out-of-bound tile
    }

    int thread_idx = cutlass::canonical_lane_idx();
    int src_rank = m / params_ptr->tile_m_perrank;
    int m_fetch = m + (params_ptr->rank - src_rank) * params_ptr->tile_m_perrank;

    Tensor mFetch =
        params_ptr->tma_load_fetch[src_rank].get_tma_tensor(make_shape(M, N, L));
    Tensor gFetch = local_tile(mFetch, take<0, 2>(TileShape{}),
                               make_coord(m_fetch, n, l));  // (TILE_M,TILE_N)
    Tensor gFetch_epi =
        flat_divide(gFetch, EpilogueTile{});  // (EPI_M,EPI_N,#EPI_M,#EPI_N)
    Tensor sFetch_epi = make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});

    ThrCopy thrblk_g2s_fetch = params_ptr->tma_load_fetch[src_rank].get_slice(_0{});
    Tensor bGS_gFetch = thrblk_g2s_fetch.partition_S(gFetch_epi);
    Tensor bGS_sFetch = thrblk_g2s_fetch.partition_D(sFetch_epi);

    bool issue_tma_load = cute::elect_one_sync();

    // wait for the peer's epilogue to publish the tile we are about to pull
    int fetch_tile_idx = params_ptr->tile_layout(m_fetch, n);
    using BarrierSync =
        cutlass::detail::NamedBarrierSync<ThreadCount, (int)RsNamedBarriers::Fetch>;
    using Barrier = GenericSystemBarrier<BarrierSync>;
    Barrier::wait_eq_reset(params_ptr->local_barrier_ptr[src_rank], thread_idx,
                           fetch_tile_idx * 2, 1);

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gFetch_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gFetch_epi); ++epi_m) {
        constexpr uint16_t mcast_mask = 0;
        uint64_t *tma_barrier = fetch_pipeline.producer_get_barrier(fetch_write_state);
        fetch_pipeline.producer_acquire(fetch_write_state);
        if (issue_tma_load) {
          copy(params_ptr->tma_load_fetch[src_rank].with(*tma_barrier, mcast_mask),
               bGS_gFetch(_, _, _, epi_m, epi_n),
               bGS_sFetch(_, _, _, fetch_write_state.index()));
          fetch_pipeline.producer_expect_transaction(fetch_write_state);
        }
        fetch_pipeline.producer_commit(fetch_write_state);
        ++fetch_write_state;
      }
    }
    return fetch_write_state;
  }

  CUTLASS_DEVICE void
  fetch_tail(FetchPipeline fetch_pipeline, PipelineState fetch_write_state) {
    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      fetch_pipeline.producer_tail(fetch_write_state);
    }
  }

  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  reduce(FetchPipeline fetch_pipeline, PipelineState fetch_read_state,
         ProblemShapeMNKL const &problem_shape, TileCoordMNKL const &tile_coord) {
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      return fetch_read_state;
    }

    int thread_idx = cutlass::canonical_lane_idx();

    int dst_rank = m / params_ptr->tile_m_perrank;
    // position of the fetched tile within OUR output segment / reduce buffer
    int m_reduce = m % params_ptr->tile_m_perrank;
    // its tile index in the full-output tile layout (for the counter flag)
    int m_reduce_in_output =
        m + (params_ptr->rank - dst_rank) * params_ptr->tile_m_perrank;

    // reduce buffer: [tile_m_perrank * TILE_M, N] row-major
    int M_reduce = params_ptr->tile_m_perrank * get<0>(TileShape{});
    auto mReduce = make_tensor(
        params_ptr->local_reduce_buffer,
        make_ordered_layout(make_shape(M_reduce, (int)N), make_step(_1{}, _0{})));
    Tensor gReduce =
        local_tile(mReduce, take<0, 2>(TileShape{}), make_coord(m_reduce, n));
    Tensor gReduce_epi = flat_divide(gReduce, EpilogueTile{});

    Tensor sReduce_epi = make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});

    constexpr int ThreadLayoutN = size<1>(EpilogueTile{}) / kAlignment;
    constexpr int ThreadLayoutM = ThreadCount / ThreadLayoutN;
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        make_layout(make_shape(Int<ThreadLayoutM>{}, Int<ThreadLayoutN>{}),
                    make_stride(Int<ThreadLayoutN>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kAlignment>{}), make_stride(_0{}, _1{})));

    auto thread_copy = tiled_copy.get_slice(thread_idx);
    Tensor tsReduce = thread_copy.partition_S(sReduce_epi);
    Tensor tgReduce = thread_copy.partition_D(gReduce_epi);

    using BarrierSync =
        cutlass::detail::NamedBarrierSync<ThreadCount, (int)RsNamedBarriers::Reduce>;
    using Barrier = GpuScopeBarrier<BarrierSync>;

    int reduce_tile_idx = params_ptr->tile_layout(m_reduce_in_output, n);
    int *lock_ptr = params_ptr->local_barrier_ptr[params_ptr->rank];
    int flag_idx = reduce_tile_idx * 2 + 1;

    bool is_local_tile_reduce = dst_rank == params_ptr->rank;

    if (not is_local_tile_reduce) {
      // remote adds must wait until our own tile put down the base value
      Barrier::wait_lt(lock_ptr, thread_idx, flag_idx, 1);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gReduce_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gReduce_epi); ++epi_m) {
        auto barrier_token = fetch_pipeline.consumer_try_wait(fetch_read_state);
        fetch_pipeline.consumer_wait(fetch_read_state, barrier_token);
        Tensor tsReduce_epi = tsReduce(_, _, _, fetch_read_state.index());
        Tensor tgReduce_epi = tgReduce(_, _, _, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int copy_m = 0; copy_m < size<1>(tgReduce_epi); ++copy_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int copy_n = 0; copy_n < size<2>(tgReduce_epi); ++copy_n) {
            Tensor trReduce = make_tensor<Element>(size<0>(tgReduce));
            copy(tiled_copy, tsReduce_epi(_, copy_m, copy_n), trReduce);
            if (is_local_tile_reduce) {
              copy(tiled_copy, trReduce, tgReduce_epi(_, copy_m, copy_n));
            } else {
              using VecType = uint_byte_t<sizeof(trReduce)>;
              local_red<VecType, sizeof(Element) * kAlignment, Element>(
                  recast<VecType>(trReduce)(_0{}),
                  (void *)tgReduce_epi(_, copy_m, copy_n).data(), true);
            }
          }
        }
        fetch_pipeline.consumer_release(fetch_read_state);
        ++fetch_read_state;
      }
    }

    int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);
    if (reduce_count == params_ptr->world_size) {
      Barrier::wait_eq_reset(lock_ptr, thread_idx, flag_idx,
                             params_ptr->world_size, 0);
    }
    return fetch_read_state;
  }
};

}  // namespace rsutil
