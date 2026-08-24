# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CUDA microbenchmark suites quantifying GPU communication/computation overlap on
NVLink/PCIe multi-GPU boxes, built to interrogate the design choices of
[flux](https://github.com/bytedance/flux)-style fused GEMM+ReduceScatter (the
paper markdown sits in the repo root). There is no application code, no test
suite, no lint: the deliverables are benchmark binaries, the result bundles
under `reports/`, and the analysis documents inside them.

Two different machines are involved:
- **This machine (Windows)** is for editing, analysis, and plotting only.
- **Benchmarks run on remote Linux GPU boxes** (4×H800 NVLink for `comm_comp`
  and `signalling`; a local sm_120 PCIe pair for `tests`). Result zips come
  back and get analyzed here.

## Build

Everything needs the submodules first: `git submodule update --init`
(CUTLASS v4.6 and ThunderKittens under `third_party/`).

- `comm_comp/`: `make -j3` — hardwired `-arch=sm_90a` (WGMMA/TMA need the `a`
  suffix), CUDA ≥ 12.4. Hopper only.
- `tests/` and `signalling/`: `make` — default `ARCH=sm_120a`; override with
  `make ARCH=sm_90a` for Hopper or `ARCH=sm_120` if the toolkit rejects the
  family-specific arch.
- `l2_peer_cache/`: `make` — default `ARCH=sm_90` (it targets the NVLink box);
  builds the same source twice, the second time with `-Xptxas -dlcm=cg`.

Each `.cu` is a standalone binary with `--help`-style usage in its header
comment; single-process multi-GPU via `cudaDeviceEnablePeerAccess` + UVA (no
MPI/NCCL/NVSHMEM), except `exp3_gemm_rs_fused` which is multi-process
(fork + cudaIPC, Linux-only).

## Running benchmarks (on the GPU box)

```bash
cd comm_comp
./pick_clock.sh                      # find a clock the box can HOLD under load
sudo nvidia-smi -lgc <freq> -i 0,1,2,3
./run_all.sh                         # QUICK=1 for short windows; FORCE=1 to skip preflight
```

`run_all.sh` refuses to run on a busy/unlocked box, verifies the clock holds
UNDER LOAD (`load_clock_check`, aborts otherwise), runs every experiment with
timeouts and `--verify`, and packs CSVs + logs + env snapshot into
`results_<stamp>/` + zip. Correctness checking is the `--verify` flag on each
binary, not a separate test suite.

**Clock lore (bites everyone):** `nvidia-smi -lgc N` is a request, not a
guarantee — this box silently sags from 1830 under load (power cap). The
20260810/20260811 bundles ran on an un-held 1830 lock; the 20260812 bundle
(held 1305 MHz, chosen by `pick_clock.sh`) supersedes their absolute numbers.
Ratios mostly survived recalibration; the exceptions are documented in
`reports/benchmark_4xH800_20260812/`. Never compare absolute times across
bundles with different clocks. The "1980 MHz" printed by binaries is
`cudaDevAttrClockRate` (peak), not the live clock.

## Experiment map

- `comm_comp/` — the core suite (CUTLASS sm90 fp16 GEMM, tile 128×256×64,
  same config family as flux's H800 tuning):
  - `exp1_ce_interference`: does copy-engine traffic cost an independent GEMM?
    Control patterns (`engine-only`, `bystander`, `local`) separate CE engine
    / fabric / local-HBM contention.
  - `exp2_ag_gemm_granularity`: AG+GEMM chunk-granularity sweep (S chunks per
    shard) — comm side is insensitive, cost is compute-side tile quantization.
  - `exp3_epilogue_remote` (v1): GEMM epilogue writing D over NVLink — the
    flux **sm80**-style push design; `scatter-local` control isolates
    segmentation from remoteness.
  - `exp3_gemm_rs_fused` (v2): faithful flux **sm90** design (local TMA store
    + per-tile system flag + fetch/reduce warps pulling over NVLink,
    `rs_*.cuh`). Runs base / ctrl (same kernel, comm off) / fused per rank:
    `struct = ctrl/base`, `comm = fused/ctrl`, `total = struct × comm` — the
    ctrl group is what makes attribution possible; keep it when extending.
- `signalling/` — push vs pull completion-signalling latency (single-process
  CUDA translation of NVSHMEM primitives, MoK-style fan-in).
- `l2_peer_cache/` — is peer memory cached in the *requester's* L2? (No: the
  L2 is memory-side.) Bandwidth + chase-latency sweeps across the L2 capacity
  and a direct NVLink byte count, all with a `local` control whose step is
  what proves the instrument works; `run_all.sh` reduces it to a `verdict.txt`
  that says INCONCLUSIVE when the control does not behave.
- `tests/` — P2P microbench for the local sm_120 pair (CE vs TMA vs register
  paths, interference matrices, ParallelKittens Fig.2/3 reproduction).

Methodology invariants when adding experiments: paired baselines re-checked
for drift, a control group whose physical expectation is known a priori
(≈1.00), locked-and-held clocks, `--verify` on every data-producing mode, CSV
output via `--csv` with self-describing headers.

## Reports

`reports/benchmark_<box>_<date>/` = one run bundle: raw `results_*/`
CSVs/logs/env snapshot (never edit those), an ANALYSIS markdown, and a
`plot_*.py` that regenerates `figs/` from the sibling CSVs (matplotlib; the
H20 comparison bundle is referenced by relative path `../comm_comp_signalling_4xH20/`).
Analysis docs are written in Chinese. Superseded claims are corrected in place
with dated 【修订】 notes pointing at the newer bundle rather than rewritten
silently. Paper-ready figures export PDF with `pdf.fonttype 42`.

`tmp/` is gitignored scratch — anything worth keeping moves into `reports/`
and gets committed.
