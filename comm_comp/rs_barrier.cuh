// rs_barrier.cuh
//
// Device-side synchronization for the fused GEMM+ReduceScatter kernel,
// ported from flux (include/flux/cuda/system_barrier.hpp and the fp16
// local_red from include/flux/cuda/memory_utils.hpp), adapted to live
// alongside stock CUTLASS 4.6 instead of inside flux's tree.
//
//   GenericSystemBarrier    system-scope (cross-GPU over NVLink) flag ops:
//                           ld.acquire.sys spin, atomicCAS_system, red.sys
//   GpuScopeBarrier         same shape but gpu-scope, for flags only ever
//                           touched by the local GPU (flux's
//                           CustomizedGenericBarrier)
//   local_red               red.global.add.noftz.v8.f16 -- the vectorized
//                           reduction the DMA reduce warp uses
//
// Flag protocol per output tile (2 ints per tile, on each rank's barrier
// buffer):
//   flag[2*t]   epilogue-done: producer CAS 0->1 after the tile's TMA store
//               retired; the (single) remote fetch warp consumes CAS 1->0
//   flag[2*t+1] reduce count: local-gpu counter, base-copy sets it to 1,
//               remote adds increment, last arriver resets to 0

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"
#include "cute/tensor.hpp"

namespace rsutil {

// Named-barrier ids for the three thread groups that publish/consume flags
// (same layout as flux's FluxNamedBarriers)
enum class RsNamedBarriers : int {
  EpilogueFlag = (int)cutlass::arch::ReservedNamedBarriers::FirstUserBarrier,
  Fetch = EpilogueFlag + 1,
  Reduce = EpilogueFlag + 2,
};

// System-scope version of cutlass::GenericBarrier (flux GenericSystemBarrier)
template <class Sync>
struct GenericSystemBarrier : public cutlass::GenericBarrier<Sync> {
 protected:
  CUTLASS_DEVICE
  static int ld_acquire(int *ptr) {
    int state = 0;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
    return state;
  }

  CUTLASS_DEVICE
  static void red_release(int *ptr, int val) {
    asm volatile("fence.acq_rel.sys;\n");
    asm volatile("red.relaxed.sys.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
  }

 public:
  CUTLASS_DEVICE
  static void wait_lt(void *lock_ptr, int thread_idx, int flag_idx, int count) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    if (thread_idx == 0) {
      #pragma unroll 1
      while (ld_acquire(flag_ptr) < count) {}
    }
    Sync::sync();
  }

  CUTLASS_DEVICE
  static void wait_eq(void *lock_ptr, int thread_idx, int flag_idx, int val) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    if (thread_idx == 0) {
      #pragma unroll 1
      while (ld_acquire(flag_ptr) != val) {}
    }
    Sync::sync();
  }

  // wait until *flag == val, then atomically replace it with reset_val
  CUTLASS_DEVICE
  static void wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx,
                            int val, int reset_val = 0) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    if (thread_idx == 0) {
      #pragma unroll 1
      while (atomicCAS_system(flag_ptr, val, reset_val) != val) {}
    }
    Sync::sync();
  }

  CUTLASS_DEVICE
  static void arrive_inc(void *lock_ptr, int thread_idx, int flag_idx,
                         int val = 1) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    Sync::sync();
    if (thread_idx == 0) {
      red_release(flag_ptr, val);
    }
  }
};

// gpu-scope variant for local-only flags (flux CustomizedGenericBarrier).
// Inherits wait_lt (gpu-scope ld.acquire) from cutlass::GenericBarrier.
template <class Sync>
struct GpuScopeBarrier : public cutlass::GenericBarrier<Sync> {
  CUTLASS_DEVICE
  static void wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx,
                            int val, int reset_val = 0) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    if (thread_idx == 0) {
      #pragma unroll 1
      while (atomicCAS(flag_ptr, val, reset_val) != val) {}
    }
    Sync::sync();
  }

  // warp-synchronous: every lane learns the post-increment value
  CUTLASS_DEVICE
  static int arrive_inc_get(void *lock_ptr, int thread_idx, int flag_idx,
                            int val = 1) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    Sync::sync();
    int old_val = 0;
    if (thread_idx == 0) {
      asm volatile("fence.acq_rel.gpu;\n");
      old_val = atomicAdd(flag_ptr, val);
    }
    return __shfl_sync(0xffffffff, old_val + val, 0);
  }
};

// red.global.add.noftz.v8.f16: adds 8 packed halves (16 bytes) into gmem
// (flux cutlass::arch::local_red, fp16 specialization)
template <typename AccessType, int StoreBytes, typename ElementType>
struct local_red;

template <typename AccessType>
struct local_red<AccessType, 16, cutlass::half_t> {
  CUTLASS_DEVICE
  local_red(AccessType const &D, void *ptr, bool pred_guard) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Registers = uint16_t[8];
    Registers const &data = reinterpret_cast<Registers const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %1, 0;\n"
        "  @p red.global.add.noftz.v8.f16 [%0], {%2, %3, %4, %5, %6, %7, %8, %9};\n"
        "}\n"
        :
        : "l"(ptr), "r"((int)pred_guard),
          "h"(data[0]), "h"(data[1]), "h"(data[2]), "h"(data[3]),
          "h"(data[4]), "h"(data[5]), "h"(data[6]), "h"(data[7]));
#else
    CUTE_INVALID_CONTROL_PATH("local_red needs sm90a");
#endif
  }
};

}  // namespace rsutil
