// ipc_multiproc.cuh
//
// Minimal one-process-per-GPU runtime for the comm_comp experiments, shaped
// like production TP deployments (flux under torchrun): fork() one child per
// rank, exchange cudaIpc memory handles through an anonymous shared mmap,
// and synchronize with a host shm barrier plus flux-style device-side
// barrier-all kernel over IPC-mapped flag buffers. No MPI, no pybind11.
//
// Linux-only (fork + mmap); the target 4xH800 box is Linux.

#pragma once

#if !defined(__linux__)
#error "ipc_multiproc.cuh requires Linux (fork/mmap); build & run on the target machine"
#endif

#include <cuda_runtime.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "common.cuh"

namespace mproc {

constexpr int kMaxWorld = 8;
constexpr int kMaxBufs = 8;    // IPC'd allocations per rank
constexpr int kMaxResults = 16;  // doubles per rank for reporting

// ---------------------------------------------------------------------------
// shared memory blob (mapped before fork, shared by all ranks)
// ---------------------------------------------------------------------------

// CUDA must NOT be initialized in the parent before fork() -- a child of a
// CUDA-initialized process cannot use CUDA (cudaErrorInitializationError).
// Probe the device count in a throwaway child instead.
inline int probe_device_count() {
  int fds[2];
  if (pipe(fds) != 0) {
    std::perror("pipe");
    std::exit(1);
  }
  pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork");
    std::exit(1);
  }
  if (pid == 0) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
    (void)!write(fds[1], &n, sizeof(n));
    _exit(0);
  }
  int n = 0;
  if (read(fds[0], &n, sizeof(n)) != sizeof(n)) n = 0;
  close(fds[0]);
  close(fds[1]);
  int st = 0;
  waitpid(pid, &st, 0);
  return n;
}

struct ShmBarrier {
  std::atomic<int> count{0};
  std::atomic<int> gen{0};

  // spins until all ranks arrive; bails out if another rank flagged failure
  void wait(int world, const std::atomic<int>* failed = nullptr) {
    const int g = gen.load();
    if (count.fetch_add(1) + 1 == world) {
      count.store(0);
      gen.fetch_add(1);
    } else {
      while (gen.load() == g) {
        if (failed && failed->load()) std::exit(1);
        sched_yield();
      }
    }
  }
};

struct Shm {
  int world = 0;
  ShmBarrier barrier;
  std::atomic<int> failed{0};  // any rank sets this on error
  cudaIpcMemHandle_t handles[kMaxWorld][kMaxBufs];
  double results[kMaxWorld][kMaxResults];
};

// Forks world-1 children. Returns the rank (0..world-1) in every process;
// the parent becomes rank 0. wait_children() must be called by rank 0 after
// all work is done.
struct MultiProc {
  Shm* shm = nullptr;
  int world = 0;
  int rank = -1;
  std::vector<pid_t> children;

  void init(int world_) {
    world = world_;
    if (world > kMaxWorld) {
      std::fprintf(stderr, "world %d > kMaxWorld %d\n", world, kMaxWorld);
      std::exit(1);
    }
    void* mem = mmap(nullptr, sizeof(Shm), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
      std::perror("mmap");
      std::exit(1);
    }
    shm = new (mem) Shm();
    shm->world = world;

    rank = 0;
    for (int r = 1; r < world; ++r) {
      pid_t pid = fork();
      if (pid < 0) {
        std::perror("fork");
        std::exit(1);
      }
      if (pid == 0) {  // child
        rank = r;
        children.clear();
        break;
      }
      children.push_back(pid);
    }
  }

  void barrier() { shm->barrier.wait(world, &shm->failed); }

  void fail(const char* msg) {
    std::fprintf(stderr, "[rank %d] FATAL: %s\n", rank, msg);
    shm->failed.store(1);
    std::exit(1);
  }

  // rank 0 only: reap children, return nonzero if any rank failed
  int wait_children() {
    int status_all = 0;
    for (pid_t pid : children) {
      int st = 0;
      waitpid(pid, &st, 0);
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) status_all = 1;
    }
    if (shm->failed.load()) status_all = 1;
    return status_all;
  }

  // ---- IPC handle exchange -------------------------------------------------
  // Register a local allocation under slot `buf_id`, then resolve every
  // rank's pointer: own rank keeps the local pointer, peers are IPC-opened.
  void exchange(int buf_id, void* local_ptr, void* ptrs_out[/*world*/]) {
    CUDA_CHECK(cudaIpcGetMemHandle(&shm->handles[rank][buf_id], local_ptr));
    barrier();
    for (int r = 0; r < world; ++r) {
      if (r == rank) {
        ptrs_out[r] = local_ptr;
      } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptrs_out[r], shm->handles[r][buf_id],
                                        cudaIpcMemLazyEnablePeerAccess));
      }
    }
    barrier();  // nobody frees/reuses handles until everyone opened them
  }
};

// ---------------------------------------------------------------------------
// device-side barrier-all (flux CudaIpcBarrierAllKernel)
// ---------------------------------------------------------------------------

struct SyncPtrs {
  int* p[kMaxWorld];
};

// sync buffer: int[world] per rank, zero-initialized. Thread i handshakes
// with rank i: set my slot on rank i, then consume rank i's slot on me.
static __global__ void barrier_all_kernel(SyncPtrs sync, int rank, int world) {
  const int i = threadIdx.x;
  if (i < world) {
    __threadfence_system();
    int* remote = sync.p[i] + rank;  // my flag in rank i's buffer
    int* local = sync.p[rank] + i;   // rank i's flag in my buffer
    while (atomicCAS_system(remote, 0, 1) != 0) {}
    while (atomicCAS_system(local, 1, 0) != 1) {}
    __threadfence_system();
  }
  __syncthreads();
}

inline void barrier_all_on_stream(const SyncPtrs& sync, int rank, int world,
                                  cudaStream_t stream) {
  barrier_all_kernel<<<1, 32, 0, stream>>>(sync, rank, world);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace mproc
