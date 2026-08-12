#!/usr/bin/env bash
# rerun_m32768.sh -- settle the m=32768 comm_sd upturn flagged in
# reports/benchmark_4xH800_20260811/ANALYSIS_MSWEEP_4xH800.md §1.
#
#   cd comm_comp && ./rerun_m32768.sh
#   QUICK=1 ./rerun_m32768.sh      # fewer iters, step 1 only (smoke test)
#   FORCE=1 ./rerun_m32768.sh      # skip the environment pre-check abort
#
# The question: at M=32768 the fused kernel's comm_sd rose 1.09 -> 1.17 and
# only fused grew a fat tail (p95 +14% vs mean; base/ctrl stayed tight). The
# 20260811 run cannot say why, because the CSV records only mean and p95 --
# a distribution shift and three slow iterations produce the same pair.
#
# Three steps, each answering one question:
#
#   step 1  is it a shift or a tail?
#           300 iters + p50 + the raw per-iteration series at the two shapes
#           that bracket the upturn.
#
#   step 2  if it is a shift: does the cost track TILE COUNT (flag protocol,
#           fetch path) or KERNEL DURATION (skew, drift)?
#           At fixed K the two are locked (both scale with M*N), so this
#           sweeps K to break them apart:
#
#             shape                tiles   pull    ~dur   holds
#             32768x8192x8192       8192   384Mi   7.9ms  the anomaly
#             32768x8192x4096       8192   384Mi   4.0ms  tiles, halves dur
#             16384x8192x16384      4096   192Mi   7.4ms  dur, halves tiles
#             16384x8192x8192       4096   192Mi   3.7ms  baseline
#
#           Compare comm_abs_us (fused-ctrl), NOT comm_sd: K moves the
#           denominator, so the ratios are not comparable across these rows.
#
#   step 3  is struct_sd < 1 at large M real, or run-order bias?
#           The default order times base first, so base absorbs any residual
#           ramp-up and struct_sd=ctrl/base is pushed under 1. Reversed order
#           prices that.
#
# Analyse with:  python3 analyze_rerun.py <results dir>

set -u
cd "$(dirname "$0")"

QUICK=${QUICK:-0}
# The whole point of this rerun is comparability with the 20260811 M-sweep,
# which ran at 1830 MHz (env.txt: current 1830, max 1980). A different lock
# would make every absolute us in step 2 incomparable with the numbers it is
# supposed to explain, so the pre-check below insists on this exact value.
REF_CLOCK=${REF_CLOCK:-1830}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=rerun_${STAMP}
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"

log() { echo "$@" | tee -a "$MANIFEST"; }

if [ "$QUICK" = "1" ]; then
  ITERS=40; WARMUP=5
else
  # 300 iters x 3 variants x ~8 ms = ~7 s per point; there is no reason to
  # economise here, and the 20260811 run's 50 iters put only ~2-3 samples in
  # the tail that the whole question is about.
  ITERS=300; WARMUP=30
fi

# ---------------------------------------------------------------------------
# environment snapshot + pre-check (same rules as run_all.sh: a benchmark on a
# busy or unlocked box produces numbers that are worse than none)
# ---------------------------------------------------------------------------
log "== rerun_m32768 $STAMP (QUICK=$QUICK ITERS=$ITERS WARMUP=$WARMUP) =="
{
  echo "--- git ---";       git rev-parse HEAD 2>/dev/null; git status --short 2>/dev/null
  echo "--- nvidia-smi ---"; nvidia-smi 2>&1
  echo "--- topology ---";   nvidia-smi topo -m 2>&1
  echo "--- clocks (before) ---"
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks_event_reasons.active \
      --format=csv 2>&1
} > "$OUT/env.txt" 2>&1

devs=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
bad=0 clock_bad=0
while IFS=',' read -r idx util mem sm smmax; do
  idx=${idx// /}; util=${util// /}; mem=${mem// /}; sm=${sm// /}; smmax=${smmax// /}
  case "$util$mem$sm" in *[!0-9]*) continue ;; esac
  [ "$util" -gt 5 ]   && { log "!! GPU $idx: utilization ${util}%"; bad=1; }
  [ "$mem" -gt 2048 ] && { log "!! GPU $idx: ${mem} MiB held by other processes"; bad=1; }
  if [ "$sm" -lt 1000 ]; then
    log "!! GPU $idx: SM clock ${sm} MHz (max ${smmax}) -- UNLOCKED"
    bad=1 clock_bad=1
  elif [ "$sm" -ne "$REF_CLOCK" ]; then
    log "!! GPU $idx: SM clock ${sm} MHz, but the 20260811 reference run used" \
        "${REF_CLOCK} MHz -- absolute us would not be comparable"
    bad=1 clock_bad=1
  fi
done < <(nvidia-smi -i "$devs" \
    --query-gpu=index,utilization.gpu,memory.used,clocks.sm,clocks.max.sm \
    --format=csv,noheader,nounits 2>/dev/null)
if [ "$clock_bad" = 1 ]; then
  log "   fix with:  nvidia-smi -lgc $REF_CLOCK -i $devs"
  log "   (this script only RECORDS clocks, it never changes them; override the"
  log "    target with REF_CLOCK=<freq> if you deliberately want another lock)"
fi
if [ "$bad" = 1 ]; then
  if [ "${FORCE:-0}" = 1 ]; then
    log "!! pre-check FAILED but FORCE=1 -- results are dirty, do not compare"
  else
    log "!! environment pre-check FAILED -- fix the above or rerun with FORCE=1"
    exit 1
  fi
fi

log "-- building (make -j4) --"
make -j4 exp3_gemm_rs_fused > "$OUT/build.log" 2>&1 || {
  log "!! build FAILED -- see $OUT/build.log"; exit 1; }

# ---------------------------------------------------------------------------
# runner. The fused experiment forks one process per GPU; a timeout kills only
# rank 0, so leftover ranks must be reaped or they poison the next point.
# ---------------------------------------------------------------------------
run_point() {
  local tag=$1 m=$2 n=$3 k=$4 order=$5 csv=$6
  log ">> $tag: ${m}x${n}x${k} order=$order iters=$ITERS"
  local t0=$SECONDS
  timeout --signal=INT --kill-after=30 900 \
    ./exp3_gemm_rs_fused --m "$m" --n "$n" --k "$k" \
      --iters "$ITERS" --warmup "$WARMUP" --verify \
      --order "$order" \
      --dump-iters "$OUT/iters_${tag}" \
      --csv "$OUT/${csv}" > "$OUT/${tag}.log" 2>&1
  local rc=$?
  log "   $tag rc=$rc dur=$((SECONDS - t0))s$([ $rc -eq 124 ] && echo ' (TIMEOUT)')"
  pkill -INT -f exp3_gemm_rs_fused 2>/dev/null; sleep 2
  pkill -KILL -f exp3_gemm_rs_fused 2>/dev/null; sleep 1
  true
}

# ---------------------------------------------------------------------------
# step 1: shift or tail? (the two shapes bracketing the upturn)
# ---------------------------------------------------------------------------
log "-- step 1: statistics (p50 + raw series) --"
run_point s1_m16384 16384 8192 8192 base,ctrl,fused step1_stats.csv
run_point s1_m32768 32768 8192 8192 base,ctrl,fused step1_stats.csv

if [ "$QUICK" = "1" ]; then
  log "-- QUICK: skipping steps 2 and 3 --"
else
  # -------------------------------------------------------------------------
  # step 2: decouple tile count / pull volume from kernel duration
  # -------------------------------------------------------------------------
  log "-- step 2: decouple tiles vs duration (vary K) --"
  # same tiles + same pull volume as the anomaly, half the duration
  run_point s2_m32768_k4096  32768 8192  4096 base,ctrl,fused step2_decouple.csv
  # same duration as the anomaly, half the tiles + half the pull volume
  run_point s2_m16384_k16384 16384 8192 16384 base,ctrl,fused step2_decouple.csv

  # -------------------------------------------------------------------------
  # step 3: run-order control
  # -------------------------------------------------------------------------
  log "-- step 3: run-order control (reverse) --"
  run_point s3_m32768_rev 32768 8192 8192 fused,ctrl,base step3_order.csv
  run_point s3_m16384_rev 16384 8192 8192 fused,ctrl,base step3_order.csv
fi

nvidia-smi --query-gpu=index,clocks.sm,clocks_event_reasons.active \
    --format=csv > "$OUT/clocks_after.txt" 2>&1

log "-- done; analysing --"
python3 analyze_rerun.py "$OUT" 2>&1 | tee -a "$OUT/analysis.txt"

if command -v zip >/dev/null 2>&1; then
  zip -qr "${OUT}.zip" "$OUT" && log "packed: comm_comp/${OUT}.zip"
else
  tar czf "${OUT}.tar.gz" "$OUT" && log "packed: comm_comp/${OUT}.tar.gz"
fi
log "== all done =="
