#!/usr/bin/env bash
# run_all.sh -- build and run every comm_comp experiment, collect logs + CSVs
# + environment info into results_<timestamp>/, and pack it into a zip (or
# tar.gz if zip is missing).
#
#   cd comm_comp && ./run_all.sh
#   QUICK=1 ./run_all.sh          # shorter windows / fewer iterations
#
# Notes:
#   * lock clocks first or the numbers are noisy:
#       sudo nvidia-smi -lgc <freq> -i 0,1,2,3
#     (the script only RECORDS clocks, it never changes them)
#   * every run has a timeout; a hang (e.g. a broken flag pairing in the
#     fused GEMM+RS) is recorded in manifest.txt as rc=124 instead of
#     stalling the script, and leftover ranks are pkill'ed.

set -u
cd "$(dirname "$0")"

QUICK=${QUICK:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=results_${STAMP}
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"

log() { echo "$@" | tee -a "$MANIFEST"; }

# ---------------------------------------------------------------------------
# environment snapshot
# ---------------------------------------------------------------------------
log "== comm_comp run_all $STAMP (QUICK=$QUICK) =="
{
  echo "--- git ---"
  git rev-parse HEAD 2>/dev/null
  git status --short 2>/dev/null
  echo "--- uname ---"
  uname -a
  echo "--- nvcc ---"
  nvcc --version 2>&1
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1
  echo "--- topology ---"
  nvidia-smi topo -m 2>&1
  echo "--- clocks (before) ---"
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv 2>&1
  echo "--- cpu ---"
  lscpu 2>/dev/null | head -20
} > "$OUT/env.txt" 2>&1

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
log "-- building (make -j4) --"
if ! make -j4 > "$OUT/build.log" 2>&1; then
  log "!! build FAILED for at least one target -- see build.log; continuing"
fi

# ---------------------------------------------------------------------------
# runner: run NAME TIMEOUT_S CMD...
# ---------------------------------------------------------------------------
run() {
  local name=$1 tmo=$2
  shift 2
  if [ ! -x "$1" ]; then
    log ">> $name: SKIPPED ($1 not built)"
    return
  fi
  log ">> $name: $* (timeout ${tmo}s)"
  local t0=$SECONDS
  timeout --signal=INT --kill-after=30 "$tmo" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  log "   $name rc=$rc dur=$((SECONDS - t0))s$([ $rc -eq 124 ] && echo ' (TIMEOUT)')"
}

# extra cleanup for the multi-process runs: a timeout kills only rank 0,
# forked ranks would keep spinning on the GPUs
cleanup_fused() {
  pkill -INT -f exp3_gemm_rs_fused 2>/dev/null
  sleep 2
  pkill -KILL -f exp3_gemm_rs_fused 2>/dev/null
  true
}

if [ "$QUICK" = "1" ]; then
  WIN=100; ITERS_FUSED=20; SIZES1="4096,8192"; SIZES2="8192"
else
  WIN=200; ITERS_FUSED=50; SIZES1="2048,4096,8192"; SIZES2="8192,16384x8192x8192"
fi

# ---------------------------------------------------------------------------
# exp1: CE traffic vs independent GEMM
# ---------------------------------------------------------------------------
run exp1_default 2400 ./exp1_ce_interference \
    --sizes "$SIZES1" --window-ms "$WIN" --csv "$OUT/exp1_default.csv"
# --comm-buf 2G so the 256M point fits a slot even at world=8 (slot = buf/7)
run exp1_msgsweep 2400 ./exp1_ce_interference \
    --sizes 8192 --patterns pull,allgather --msgs 1M,4M,16M,64M,256M \
    --comm-buf 2G --window-ms "$WIN" --csv "$OUT/exp1_msgsweep.csv"

# ---------------------------------------------------------------------------
# exp2: AG+GEMM granularity sweep (verify first, then the sweeps)
# ---------------------------------------------------------------------------
run exp2_verify 900 ./exp2_ag_gemm_granularity \
    --sizes 8192 --chunks 1,4,16 --iters 10 --verify
run exp2_pull 2400 ./exp2_ag_gemm_granularity \
    --sizes "$SIZES2" --window-ms "$WIN" --csv "$OUT/exp2_pull.csv"
run exp2_serial 1500 ./exp2_ag_gemm_granularity \
    --sizes 8192 --comm-streams 1 --window-ms "$WIN" --csv "$OUT/exp2_serial.csv"
run exp2_push 1500 ./exp2_ag_gemm_granularity \
    --sizes 8192 --push --window-ms "$WIN" --csv "$OUT/exp2_push.csv"

# ---------------------------------------------------------------------------
# exp3 v1: epilogue remote write (TMA + nosmem epilogues, all modes)
# ---------------------------------------------------------------------------
run exp3_epilogue 2400 ./exp3_epilogue_remote \
    --verify --window-ms "$WIN" --csv "$OUT/exp3_epilogue.csv"

# ---------------------------------------------------------------------------
# exp3 v2: fused GEMM+RS (multi-process; pkill leftovers after each)
# ---------------------------------------------------------------------------
run exp3_fused_8192 900 ./exp3_gemm_rs_fused \
    --iters "$ITERS_FUSED" --verify --csv "$OUT/exp3_fused.csv"
cleanup_fused
run exp3_fused_k2048 900 ./exp3_gemm_rs_fused \
    --m 8192 --n 8192 --k 2048 --iters "$ITERS_FUSED" --verify \
    --csv "$OUT/exp3_fused.csv"
cleanup_fused
run exp3_fused_k512 900 ./exp3_gemm_rs_fused \
    --m 8192 --n 8192 --k 512 --iters "$ITERS_FUSED" --verify \
    --csv "$OUT/exp3_fused.csv"
cleanup_fused

# ---------------------------------------------------------------------------
# wrap up
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active \
    --format=csv > "$OUT/clocks_after.txt" 2>&1

log "-- done; packing --"
if command -v zip >/dev/null 2>&1; then
  zip -qr "${OUT}.zip" "$OUT" && log "packed: comm_comp/${OUT}.zip"
else
  tar czf "${OUT}.tar.gz" "$OUT" && log "packed: comm_comp/${OUT}.tar.gz (no zip on box)"
fi
log "== all done =="
