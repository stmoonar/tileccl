#!/usr/bin/env bash
# run_all.sh -- build both probe binaries, run every instrument on both, ask
# NCU for an independent byte count, reduce it all to a verdict, and pack
# results_<timestamp>/ into a zip (tar.gz if zip is missing).
#
#   cd l2_peer_cache && ./run_all.sh
#   READER=0 OWNER=1 ./run_all.sh   # which pair to test (default 0,1)
#   QUICK=1 ./run_all.sh            # shorter windows
#   FORCE=1 ./run_all.sh            # skip the environment pre-check abort
#   NO_NCU=1 ./run_all.sh           # skip profiling even if ncu is installed
#
# Notes:
#   * lock clocks first if the absolute GB/s matter:
#       nvidia-smi -lgc <freq> -i <READER>,<OWNER>
#     The verdict itself is a set of RATIOS taken inside one sweep, so it
#     survives a drooping clock; the absolute bandwidths do not.
#   * the pre-check refuses to run on a busy box. Everything here is a
#     bandwidth/latency measurement, so a co-tenant does not just add noise,
#     it moves the numbers the conclusion is read from.
#   * `ncu` needs profiling permission (ERR_NVGPUCTRPERM otherwise ->
#     NVreg_RestrictProfilingToAdminUsers=0, or run as root). Its absence is
#     not fatal: the nvidia-smi NVLink counters need no permission at all.

set -u
cd "$(dirname "$0")"

READER=${READER:-0}
OWNER=${OWNER:-1}
QUICK=${QUICK:-0}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=results_${STAMP}
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"

log() { echo "$@" | tee -a "$MANIFEST"; }

log "== l2_peer_cache run_all $STAMP (reader=$READER owner=$OWNER QUICK=$QUICK) =="

# ---------------------------------------------------------------------------
# environment snapshot
# ---------------------------------------------------------------------------
{
  echo "--- git ---"
  git rev-parse HEAD 2>/dev/null
  git status --short 2>/dev/null
  echo "--- uname ---"
  uname -a
  echo "--- nvcc ---"
  nvcc --version 2>&1
  echo "--- ncu ---"
  ncu --version 2>&1 | head -5
  echo "--- nvidia-smi ---"
  nvidia-smi 2>&1
  echo "--- topology ---"
  nvidia-smi topo -m 2>&1
  echo "--- nvlink status ---"
  nvidia-smi nvlink -s 2>&1
  echo "--- nvlink capabilities ---"
  nvidia-smi nvlink -c 2>&1
  echo "--- nvlink data counters (before) ---"
  nvidia-smi nvlink -gt d 2>&1
  echo "--- clocks (before) ---"
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active \
      --format=csv 2>&1
} > "$OUT/env.txt" 2>&1

# ---------------------------------------------------------------------------
# environment pre-check
# ---------------------------------------------------------------------------
bad=0
REF_IDLE_CLOCK=0
while IFS=',' read -r idx util mem sm smmax; do
  idx=${idx// /}; util=${util// /}; mem=${mem// /}
  sm=${sm// /}; smmax=${smmax// /}
  case "$util$mem$sm" in *[!0-9]*) continue ;; esac    # skip [N/A] rows
  case ",$READER,$OWNER," in *",$idx,"*) ;; *) continue ;; esac
  if [ "$util" -gt 5 ]; then
    log "!! GPU $idx: utilization ${util}% -- another job is running"; bad=1
  fi
  if [ "$mem" -gt 2048 ]; then
    log "!! GPU $idx: ${mem} MiB already allocated by other processes"; bad=1
  fi
  if [ "$sm" -lt 1000 ]; then
    log "!! GPU $idx: SM clock ${sm} MHz (max ${smmax}) -- looks UNLOCKED;" \
        "run: nvidia-smi -lgc <freq> -i $READER,$OWNER"; bad=1
  fi
  [ "$REF_IDLE_CLOCK" = 0 ] && REF_IDLE_CLOCK=$sm
done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.sm,clocks.max.sm \
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
ARCH_ARG=""
[ -n "${ARCH:-}" ] && ARCH_ARG="ARCH=$ARCH"
log "-- building (make -j2 $ARCH_ARG) --"
if ! make -j2 $ARCH_ARG > "$OUT/build.log" 2>&1; then
  log "!! build FAILED -- see build.log"
  cat "$OUT/build.log" | tail -20 | tee -a "$MANIFEST"
  exit 1
fi

# ---------------------------------------------------------------------------
# clock / power sampler (100 ms), same trick as comm_comp/run_all.sh
# ---------------------------------------------------------------------------
SAMPLE_FIELDS="index,clocks.sm,clocks.mem,power.draw,temperature.gpu,\
clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown"
SAMPLER_PID=""
if nvidia-smi -i "$READER,$OWNER" --query-gpu="$SAMPLE_FIELDS" \
     --format=csv,noheader,nounits > "$OUT/gpu_trace_probe.txt" 2>&1; then
  ( while :; do
      printf '%s,' "$(date +%s.%N)"
      nvidia-smi -i "$READER,$OWNER" --query-gpu="$SAMPLE_FIELDS" \
        --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';'
      printf '\n'
      sleep 0.1
    done ) > "$OUT/gpu_trace.csv" 2>/dev/null &
  SAMPLER_PID=$!
  log "-- gpu sampler started (pid $SAMPLER_PID, 100 ms) --"
else
  log "!! gpu sampler DISABLED: query failed -- see gpu_trace_probe.txt"
fi
stop_sampler() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null
  true
}
trap stop_sampler EXIT

# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
run() {
  local name=$1 tmo=$2
  shift 2
  if [ ! -x "$1" ] && ! command -v "$1" >/dev/null 2>&1; then
    log ">> $name: SKIPPED ($1 not available)"
    return 1
  fi
  log ">> $name: $* (timeout ${tmo}s)"
  local t0=$SECONDS
  timeout --signal=INT --kill-after=30 "$tmo" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  log "   $name rc=$rc dur=$((SECONDS - t0))s$([ $rc -eq 124 ] && echo ' (TIMEOUT)')"
  return $rc
}

if [ "$QUICK" = "1" ]; then
  TOTAL=1G; REPS=3; HOPS=5000; WIRE_TOTAL=8G; WIRE_SIZES=8M; NCU_TOTAL=512M
else
  TOTAL=4G; REPS=5; HOPS=20000; WIRE_TOTAL=64G; WIRE_SIZES=1M,8M,64M
  NCU_TOTAL=2G
fi
COMMON="--reader $READER --owner $OWNER"

# ---------------------------------------------------------------------------
# under-load clock check: the sweeps compare bandwidths measured seconds apart,
# so a clock that sags under sustained load biases the large-working-set end of
# every curve against the small end -- exactly the axis the verdict reads.
# ---------------------------------------------------------------------------
log "-- under-load clock check (~3 s of local streaming reads) --"
./peer_l2_probe $COMMON --modes bw --targets local --sizes 512M --total 512G \
    --reps 8 --no-verify --no-counters > "$OUT/load_check.log" 2>&1 &
job=$!
samples=""
sleep 0.4                      # let allocation and the fill kernels finish
for _ in $(seq 1 30); do
  kill -0 $job 2>/dev/null || break   # never average idle samples into this
  sleep 0.2
  row=$(nvidia-smi -i "$READER" --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  case "$row" in ''|*[!0-9]*) ;; *) samples="$samples $row" ;; esac
done
wait $job 2>/dev/null
n=0; sum=0; lo=999999
for c in $samples; do
  n=$((n + 1)); sum=$((sum + c)); [ "$c" -lt "$lo" ] && lo=$c
done
if [ "$n" -ge 5 ]; then
  avg=$((sum / n))
  log "   GPU $READER under load: mean ${avg} MHz, min ${lo} MHz over $n samples"
  if [ "$REF_IDLE_CLOCK" -gt 0 ] && [ "$avg" -lt $((REF_IDLE_CLOCK * 95 / 100)) ]; then
    log "!! SM clock droops to ${avg} MHz under load vs ${REF_IDLE_CLOCK} MHz idle"
    log "   ($((100 * avg / REF_IDLE_CLOCK))%). Ratios inside one sweep still"
    log "   hold roughly, but absolute GB/s here are not comparable to any"
    log "   other bundle. Find a holdable clock: ../comm_comp/pick_clock.sh"
  fi
else
  log "   load clock check: only $n usable samples -- inconclusive"
fi

# ---------------------------------------------------------------------------
# the measurements: every instrument, both builds
# ---------------------------------------------------------------------------
for build in ca cg; do
  bin=./peer_l2_probe
  [ "$build" = cg ] && bin=./peer_l2_probe_nol1
  run "probe_$build" 1800 "$bin" $COMMON \
      --modes bw,lat,wire \
      --total "$TOTAL" --reps "$REPS" --hops "$HOPS" \
      --wire-sizes "$WIRE_SIZES" --wire-total "$WIRE_TOTAL" \
      --csv "$OUT/$build"
done

# A second peer-only wire point with a working set far larger than L2. Nobody
# expects caching there, so its RX/asked is the calibration for the small-buffer
# number: if the 1 MB point and the 512 MB point both come back at 1.00, the
# small one is not a counter artefact.
run "wire_bigbuf" 900 ./peer_l2_probe_nol1 $COMMON \
    --modes wire --targets peer --wire-sizes 512M --wire-total "$WIRE_TOTAL" \
    --no-verify --csv "$OUT/bigbuf"

# ---------------------------------------------------------------------------
# NCU: an independent byte count, from the hardware performance counters
# instead of the driver's link counters.
# ---------------------------------------------------------------------------
# NCU's "peer traffic" metrics only cover PCIe-attached GPUs, so NVLink needs
# the Nvlink section explicitly. --kernel-name filters first, then skip/count,
# so this profiles the single timed launch and not the fill or warmup kernels.
NCU_BASE="--target-processes all --clock-control none --kernel-name stream_read_kernel --launch-skip 1 --launch-count 1"
NCU_METRICS="dram__bytes_read.sum,lts__t_sectors.sum,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
NCU_METRICS_MIN="dram__bytes_read.sum,lts__t_sectors.sum"

if [ "${NO_NCU:-0}" = 1 ]; then
  log "-- ncu skipped (NO_NCU=1) --"
elif ! command -v ncu >/dev/null 2>&1; then
  log "-- ncu not installed; skipping the profiler cross-check --"
else
  for tgt in peer local; do
    run "ncu_nvlink_$tgt" 900 ncu $NCU_BASE --section Nvlink \
        --log-file "$OUT/ncu_nvlink_$tgt.txt" \
        ./peer_l2_probe_nol1 $COMMON --modes once --targets "$tgt" \
        --sizes 8M --total "$NCU_TOTAL" --no-verify --no-counters
    rc=$?
    if [ "$rc" != 0 ] && grep -q "ERR_NVGPUCTRPERM" "$OUT/ncu_nvlink_$tgt.log" 2>/dev/null; then
      log "!! ncu lacks profiling permission on this box."
      log "   Fix: run as root, or set NVreg_RestrictProfilingToAdminUsers=0"
      log "   (modprobe.d) and reboot. The nvidia-smi counters above need no"
      log "   permission and answer the same question."
      break
    fi
    if ! run "ncu_metrics_$tgt" 900 ncu $NCU_BASE --metrics "$NCU_METRICS" \
        --log-file "$OUT/ncu_metrics_$tgt.txt" \
        ./peer_l2_probe_nol1 $COMMON --modes once --targets "$tgt" \
        --sizes 8M --total "$NCU_TOTAL" --no-verify --no-counters; then
      log "   retrying with the minimal metric set (some names are chip-specific)"
      run "ncu_metrics_min_$tgt" 900 ncu $NCU_BASE --metrics "$NCU_METRICS_MIN" \
          --log-file "$OUT/ncu_metrics_min_$tgt.txt" \
          ./peer_l2_probe_nol1 $COMMON --modes once --targets "$tgt" \
          --sizes 8M --total "$NCU_TOTAL" --no-verify --no-counters
    fi
  done
  log "   ncu note: for target=local, dram__bytes_read.sum ~= 8 MiB (one cold"
  log "   pass; the other $NCU_TOTAL of reads were L2 hits). For target=peer it"
  log "   should be ~0 while the Nvlink section shows ~$NCU_TOTAL received."
fi

stop_sampler; trap - EXIT
nvidia-smi nvlink -gt d > "$OUT/nvlink_counters_after.txt" 2>&1
nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active \
    --format=csv > "$OUT/clocks_after.txt" 2>&1

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
# Reduces the CSVs to the three numbers the conclusion rests on, and refuses to
# conclude anything if the control did not behave.
{
  echo "=================================================================="
  echo " VERDICT: is peer memory cached in the REQUESTER's L2?"
  echo " (bundle $OUT, reader GPU $READER, owner GPU $OWNER)"
  echo "=================================================================="
  echo
  for build in ca cg; do
    tag="default L1"; [ "$build" = cg ] && tag="L1 bypassed (-dlcm=cg)"
    echo "---- build $build: $tag ----"
    if [ -s "$OUT/${build}_bw.csv" ]; then
      awk -F, '
        $1=="build" {next}
        {t=$2; bytes=$3+0; frac=$4+0; g=$11+0
         if (frac<=0.25 && bytes>inb[t]) {inb[t]=bytes; ing[t]=g}
         if (frac>=2 && (outb[t]==0 || bytes<outb[t])) {outb[t]=bytes; outg[t]=g}}
        END{
          split("local peer", ord, " ")
          for (i=1; i<=2; i++) { t=ord[i]; if (outg[t]>0)
            printf "  bw  %-5s  in-L2 %8.0f GB/s   out-of-L2 %8.0f GB/s   knee %5.2fx\n",
                   t, ing[t], outg[t], ing[t]/outg[t] }
        }' "$OUT/${build}_bw.csv"
    fi
    if [ -s "$OUT/${build}_lat.csv" ]; then
      awk -F, '
        $1=="build" {next}
        {t=$2; bytes=$3+0; frac=$4+0; ns=$8+0
         if (frac<=0.25 && bytes>inb[t]) {inb[t]=bytes; inn[t]=ns}
         if (frac>=2 && (outb[t]==0 || bytes<outb[t])) {outb[t]=bytes; outn[t]=ns}}
        END{
          split("local peer", ord, " ")
          for (i=1; i<=2; i++) { t=ord[i]; if (inn[t]>0)
            printf "  lat %-5s  in-L2 %8.0f ns     out-of-L2 %8.0f ns     ratio %5.2fx\n",
                   t, inn[t], outn[t], outn[t]/inn[t] }
        }' "$OUT/${build}_lat.csv"
    fi
    if [ -s "$OUT/${build}_wire.csv" ]; then
      awk -F, '
        $1=="build" {next}
        $15==1 {printf "  wire %-5s %8.1f MiB working set -> RX/asked %6.3f, TX/asked %6.3f\n",
                       $2, $3/1048576, $11, $12}
        $15==0 {print "  wire " $2 ": NVLink counters unavailable"}
      ' "$OUT/${build}_wire.csv"
    fi
    echo
  done
  if [ -s "$OUT/bigbuf_wire.csv" ]; then
    awk -F, '$1!="build" && $15==1 {
        printf "  wire peer, %.0f MiB (>> L2, caching impossible) -> RX/asked %6.3f\n",
               $3/1048576, $11}' "$OUT/bigbuf_wire.csv"
    echo
  fi

  # The decision, from the L1-bypassed build (the one that isolates L2).
  awk -F, -v wire="$OUT/cg_wire.csv" '
    $1=="build" {next}
    {t=$2; bytes=$3+0; frac=$4+0; g=$11+0
     if (frac<=0.25 && bytes>inb[t]) {inb[t]=bytes; ing[t]=g}
     if (frac>=2 && (outb[t]==0 || bytes<outb[t])) {outb[t]=bytes; outg[t]=g}}
    END{
      lk = (outg["local"]>0) ? ing["local"]/outg["local"] : 0
      pk = (outg["peer"]>0)  ? ing["peer"]/outg["peer"]   : 0
      wr = -1; n = 0
      while ((getline line < wire) > 0) {
        split(line, f, ",")
        if (f[1]=="build" || f[2]!="peer" || f[15]+0 != 1) continue
        wr = (n==0) ? f[11]+0 : ((wr*n + f[11]) / (n+1)); n++
      }
      # awk does not allow a newline inside a ?: -- keep these on one line each.
      m1 = "[!! no step: the instrument is blind]"
      if (lk > 1.5) m1 = "[ok: the instrument sees an L2]"
      m2 = "[!! stepped, like a cache]"
      if (pk < 1.2) m2 = "[flat: nothing cached locally]"
      m3 = "[!! neither ~1 nor ~0]"
      if (wr > 0.85 && wr < 1.15) m3 = "[every byte crossed the link]"
      printf "CONTROL  local L2 knee            = %.2fx  %s\n", lk, m1
      printf "TEST     peer  L2 knee            = %.2fx  %s\n", pk, m2
      if (n>0) printf "TEST     peer  wire RX / asked    = %.3f   %s\n", wr, m3
      else     printf "TEST     peer  wire RX / asked    = n/a     [no NVLink counters]\n"
      print ""
      if (lk <= 1.5) {
        print "=> INCONCLUSIVE. The control did not show an L2 step, so a flat"
        print "   peer curve proves nothing here. Check that the box was idle"
        print "   and that the sweep really spans the L2 capacity."
      } else if (pk < 1.2 && (n==0 || (wr>0.85 && wr<1.15))) {
        print "=> PEER MEMORY IS NOT CACHED IN THE REQUESTER-SIDE L2."
        print "   Same kernel, same GPU, working sets from far below to far"
        print "   above L2: the local buffer shows the cache, the peer buffer"
        print "   does not."
        if (n>0) print "   The link also carried every byte that was asked for."
        else     print "   (Bandwidth/latency evidence only -- no link counters here.)"
      } else {
        print "=> UNEXPECTED: the peer curve moved with working-set size."
        print "   Do not write this up before checking clocks, box idleness,"
        print "   and that --targets peer really used the remote allocation"
        print "   (the probe prints cudaPointerGetAttributes at startup)."
      }
    }' "$OUT/cg_bw.csv"
} > "$OUT/verdict.txt" 2>&1
cat "$OUT/verdict.txt" | tee -a "$MANIFEST"

# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------
log "-- done; packing --"
if command -v zip >/dev/null 2>&1; then
  zip -qr "${OUT}.zip" "$OUT" && log "packed: l2_peer_cache/${OUT}.zip"
else
  tar czf "${OUT}.tar.gz" "$OUT" && log "packed: l2_peer_cache/${OUT}.tar.gz (no zip on box)"
fi
log "== all done =="
