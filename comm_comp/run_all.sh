#!/usr/bin/env bash
# run_all.sh -- build and run every comm_comp experiment, collect logs + CSVs
# + environment info into results_<timestamp>/, and pack it into a zip (or
# tar.gz if zip is missing).
#
#   cd comm_comp && ./run_all.sh
#   QUICK=1 ./run_all.sh          # shorter windows / fewer iterations
#   FORCE=1 ./run_all.sh          # skip the environment pre-check abort
#
# Notes:
#   * lock clocks first or the numbers are noisy:
#       nvidia-smi -lgc <freq> -i 0,1,2,3
#     (the script only RECORDS clocks, it never changes them)
#   * the pre-check below refuses to run if any target GPU is busy, has
#     foreign allocations, or looks unlocked -- a benchmark on a shared box
#     produces numbers that are worse than none (they look real).
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
# environment pre-check: refuse to bench on a busy or unlocked box.
# util/memory catch a co-tenant job (its processes may be invisible from
# inside a container, but its load is not); a low idle SM clock means nobody
# ran nvidia-smi -lgc (locked clocks hold their frequency even at idle).
# ---------------------------------------------------------------------------
ids=""
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ids="-i $CUDA_VISIBLE_DEVICES"
bad=0
REF_IDLE_CLOCK=0
while IFS=',' read -r idx util mem sm smmax; do
  idx=${idx// /}; util=${util// /}; mem=${mem// /}
  sm=${sm// /}; smmax=${smmax// /}
  case "$util$mem$sm" in *[!0-9]*) continue ;; esac   # skip [N/A] rows
  if [ "$util" -gt 5 ]; then
    log "!! GPU $idx: utilization ${util}% -- another job is running"; bad=1
  fi
  if [ "$mem" -gt 2048 ]; then
    log "!! GPU $idx: ${mem} MiB already allocated by other processes"; bad=1
  fi
  if [ "$sm" -lt 1000 ]; then
    log "!! GPU $idx: SM clock ${sm} MHz (max ${smmax}) -- looks UNLOCKED;" \
        "run: nvidia-smi -lgc <freq> -i ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"; bad=1
  fi
  # remember what idle looked like; load_clock_check compares against it
  [ "${REF_IDLE_CLOCK:-0}" = 0 ] && REF_IDLE_CLOCK=$sm
done < <(nvidia-smi $ids \
    --query-gpu=index,utilization.gpu,memory.used,clocks.sm,clocks.max.sm \
    --format=csv,noheader,nounits 2>/dev/null)
if [ "$bad" = 1 ]; then
  if [ "${FORCE:-0}" = 1 ]; then
    log "!! pre-check FAILED but FORCE=1 -- continuing; treat results as dirty"
  else
    log "!! environment pre-check FAILED -- fix the above or rerun with FORCE=1"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
log "-- building (make -j4) --"
if ! make -j4 > "$OUT/build.log" 2>&1; then
  log "!! build FAILED for at least one target -- see build.log; continuing"
fi

# ---------------------------------------------------------------------------
# clock / power / temperature sampler
# ---------------------------------------------------------------------------
# `nvidia-smi -lgc` locks the SM clock but NOT the memory clock, and it does not
# stop the hardware from capping on power or temperature. Both would be
# invisible in the result CSVs and would show up only as inexplicable timing
# shifts -- which is exactly what the 20260812 unified-calibre run hit: exp1's
# opening baseline and exp2's overlap window, both measured ~0.5-0.75 s into
# sustained load, came out 4-8% slow while every measurement before and after
# them was fine. Sampling at 100 ms turns that class of question into a lookup.
SAMPLE_FIELDS="index,clocks.sm,clocks.mem,power.draw,temperature.gpu,\
clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_power_cap,\
clocks_event_reasons.hw_thermal_slowdown"

start_sampler() {
  # Probe once first: a bad field name makes nvidia-smi fail, and with the
  # loop's stderr discarded that would leave an empty trace file that looks
  # like "nothing happened" rather than "the query was wrong".
  if ! nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
       --query-gpu="$SAMPLE_FIELDS" --format=csv,noheader,nounits \
       > "$OUT/gpu_trace_probe.txt" 2>&1; then
    log "!! gpu sampler DISABLED: query failed -- see gpu_trace_probe.txt"
    SAMPLER_PID=""
    return
  fi
  ( while :; do
      printf '%s,' "$(date +%s.%N)"
      nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
        --query-gpu="$SAMPLE_FIELDS" \
        --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';'
      printf '\n'
      sleep 0.1
    done ) > "$OUT/gpu_trace.csv" 2>/dev/null &
  SAMPLER_PID=$!
  log "-- gpu sampler started (pid $SAMPLER_PID, 100 ms) --"
}

stop_sampler() {
  [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null
  wait "${SAMPLER_PID:-}" 2>/dev/null
  true
}

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
  log "   $name epoch_start=$(date +%s.%N)"
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

# WARM: wall-clock warmup floor, passed to every experiment. A fixed warmup
# ITERATION count is a shape-dependent amount of wall clock -- 30 iterations is
# 0.68 s at M=32768 but 0.022 s at M=512 -- so small shapes get timed before
# they reach steady state. Every ratio here is doubly exposed, because the
# denominator (alone / local / base) is timed FIRST and absorbs the transient.
# 500 ms warms every shape equally; costs ~4 min across the suite.
if [ "$QUICK" = "1" ]; then
  WIN=100; ITERS_FUSED=40; WARM=50; SIZES1="4096,8192"; SIZES2="8192"
  MSWEEP_V1="512x8192x8192"
  MSWEEP_V2="2048"
else
  WIN=200; ITERS_FUSED=300; WARM=500
  SIZES1="2048,4096,8192"; SIZES2="8192,16384x8192x8192"
  # M-sweep at fixed K=8192: flop/byte-of-D is constant, so any tax growth at
  # small M is a granularity/latency effect (few tiles, shallow waves, fixed
  # sync overheads), not a bandwidth one -- the axis the flux paper sweeps
  # (their decode points m=64/512), orthogonal to our K sweep.
  # (m=8192 comes from the default exp3 runs; above it the sweep checks the
  # large-M plateau -- comm_sd should stay flat once waves are saturated)
  MSWEEP_V1="64x8192x8192,128x8192x8192,256x8192x8192,512x8192x8192,1024x8192x8192,2048x8192x8192,4096x8192x8192,16384x8192x8192,32768x8192x8192"
  MSWEEP_V2="512 1024 2048 4096 8192 16384 32768"
fi

# ---------------------------------------------------------------------------
# under-load clock check
# ---------------------------------------------------------------------------
# The idle check above is necessary but NOT sufficient, and the difference is
# not academic: on this box `nvidia-smi -lgc 1830` holds 1830 MHz at idle
# (97.9% of idle samples) and then collapses to a median of 1380 MHz under
# sustained GEMM, with sw_power_cap active in 79% of loaded samples at ~600 W.
# Idle is precisely when the power cap is NOT engaged, so the check that was
# there could never have caught it. Every "locked clocks" claim in the
# 20260810/11/12 reports rests on that idle reading.
#
# So: generate real load, sample the clock during it, and compare against what
# the idle check saw. Ratios taken back-to-back survive a drooping clock (both
# sides see the same one); absolute us/TFLOP/s and any comparison across
# minutes do not.
load_clock_check() {
  local dev=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  local first=${dev%%,*}
  [ -x ./exp3_epilogue_remote ] || { log "-- load clock check SKIPPED --"; return; }
  log "-- under-load clock check (4 s of 8192^3 GEMM) --"
  ./exp3_epilogue_remote --sizes 8192x8192x8192 --modes local \
      --window-ms 2000 --warmup-ms 2000 > "$OUT/load_check.log" 2>&1 &
  local job=$!
  local samples=""
  for _ in $(seq 1 20); do
    sleep 0.2
    local row
    row=$(nvidia-smi -i "$first" --query-gpu=clocks.sm,power.draw \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    case "$row" in
      *,*) [ "${row#*,}" != "" ] && samples="$samples ${row%%,*}" ;;
    esac
  done
  wait $job 2>/dev/null
  local n=0 sum=0 lo=999999
  for c in $samples; do
    case "$c" in *[!0-9]*) continue ;; esac
    n=$((n + 1)); sum=$((sum + c))
    [ "$c" -lt "$lo" ] && lo=$c
  done
  if [ "$n" -lt 5 ]; then
    log "!! load clock check: only $n usable samples -- inconclusive"
    return
  fi
  local avg=$((sum / n))
  log "   GPU $first under load: mean ${avg} MHz, min ${lo} MHz over $n samples"
  local idle=${REF_IDLE_CLOCK:-0}
  if [ "$idle" -gt 0 ] && [ "$avg" -lt $((idle * 95 / 100)) ]; then
    log "!! SM clock droops to ${avg} MHz under load vs ${idle} MHz at idle" \
        "($((100 * avg / idle))%)."
    log "   The clock is NOT locked where it matters. Ratios measured"
    log "   back-to-back still hold; absolute us/TFLOP/s and anything compared"
    log "   across minutes do not. Re-lock at a clock the box can sustain"
    log "   (try: nvidia-smi -lgc ${avg} -i $dev) or accept ratios only."
    [ "${FORCE:-0}" = 1 ] || log "   (continuing anyway -- this is a warning, not an abort)"
  fi
}
load_clock_check

start_sampler

# ---------------------------------------------------------------------------
# exp1: CE traffic vs independent GEMM
# ---------------------------------------------------------------------------
run exp1_default 2400 ./exp1_ce_interference \
    --sizes "$SIZES1" --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp1_default.csv"
# --comm-buf 2G so the 256M point fits a slot even at world=8 (slot = buf/7)
run exp1_msgsweep 2400 ./exp1_ce_interference \
    --sizes 8192 --patterns pull,allgather --msgs 1M,4M,16M,64M,256M \
    --comm-buf 2G --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp1_msgsweep.csv"

# ---------------------------------------------------------------------------
# exp2: AG+GEMM granularity sweep (verify first, then the sweeps)
# ---------------------------------------------------------------------------
run exp2_verify 900 ./exp2_ag_gemm_granularity \
    --sizes 8192 --chunks 1,4,16 --iters 10 --verify
run exp2_pull 2400 ./exp2_ag_gemm_granularity \
    --sizes "$SIZES2" --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp2_pull.csv"
run exp2_serial 1500 ./exp2_ag_gemm_granularity \
    --sizes 8192 --comm-streams 1 --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp2_serial.csv"
run exp2_push 1500 ./exp2_ag_gemm_granularity \
    --sizes 8192 --push --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp2_push.csv"

# ---------------------------------------------------------------------------
# exp3 v1: epilogue remote write (TMA + nosmem epilogues, all modes)
# ---------------------------------------------------------------------------
run exp3_epilogue 2400 ./exp3_epilogue_remote \
    --verify --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp3_epilogue.csv"
# M-sweep companion (fixed K): remote-write tax should stay flat vs M (the
# intensity K is unchanged); a rise at small M isolates partial-tile / launch
# granularity effects. m=64 mirrors the flux paper's smallest decode point;
# the fused kernel below cannot reach it (M % (tile_M*world) == 0).
run exp3_epilogue_msweep 2400 ./exp3_epilogue_remote \
    --sizes "$MSWEEP_V1" --verify --window-ms "$WIN" --warmup-ms "$WARM" \
    --csv "$OUT/exp3_epilogue_msweep.csv"

# ---------------------------------------------------------------------------
# exp3 v2: fused GEMM+RS (multi-process; pkill leftovers after each)
# ---------------------------------------------------------------------------
run exp3_fused_8192 900 ./exp3_gemm_rs_fused \
    --iters "$ITERS_FUSED" --warmup-ms "$WARM" --verify \
    --csv "$OUT/exp3_fused.csv"
cleanup_fused
run exp3_fused_k2048 900 ./exp3_gemm_rs_fused \
    --m 8192 --n 8192 --k 2048 --iters "$ITERS_FUSED" --warmup-ms "$WARM" \
    --verify \
    --csv "$OUT/exp3_fused.csv"
cleanup_fused
run exp3_fused_k512 900 ./exp3_gemm_rs_fused \
    --m 8192 --n 8192 --k 512 --iters "$ITERS_FUSED" --warmup-ms "$WARM" \
    --verify \
    --csv "$OUT/exp3_fused.csv"
cleanup_fused

# M-sweep at fixed K=8192 (m=8192 point == exp3_fused_8192 above). Comm and
# compute shrink together here, so comm_sd should hold ~constant until the
# per-rank tile count gets too small to fill waves / hide flag latency --
# the paper's small-m failure mode, distinct from the small-K floor above.
for m in $MSWEEP_V2; do
  run "exp3_fused_m${m}" 900 ./exp3_gemm_rs_fused \
      --m "$m" --n 8192 --k 8192 --iters "$ITERS_FUSED" --warmup-ms "$WARM" \
      --verify --dump-iters "$OUT/iters_fused_m${m}" \
      --csv "$OUT/exp3_fused_msweep.csv"
  cleanup_fused
done

# ---------------------------------------------------------------------------
# wrap up
# ---------------------------------------------------------------------------
stop_sampler
nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active \
    --format=csv > "$OUT/clocks_after.txt" 2>&1

log "-- done; packing --"
if command -v zip >/dev/null 2>&1; then
  zip -qr "${OUT}.zip" "$OUT" && log "packed: comm_comp/${OUT}.zip"
else
  tar czf "${OUT}.tar.gz" "$OUT" && log "packed: comm_comp/${OUT}.tar.gz (no zip on box)"
fi
log "== all done =="
