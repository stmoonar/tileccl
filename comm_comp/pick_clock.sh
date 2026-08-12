#!/usr/bin/env bash
# pick_clock.sh -- find the highest SM clock this box can actually HOLD under
# sustained GEMM load, so run_all.sh's absolute numbers mean something.
#
#   cd comm_comp && ./pick_clock.sh                 # try a sensible ladder
#   ./pick_clock.sh 1400 1300 1200 1100             # try your own candidates
#   KEEP=1 ./pick_clock.sh                          # leave the winner locked
#
# UNLIKE run_all.sh, THIS SCRIPT CHANGES GPU CLOCKS. It sets each candidate in
# turn with `nvidia-smi -lgc`, and unless KEEP=1 it resets clocks (-rgc) when
# it is done. Nothing else on the box should be running.
#
# Why this exists: `nvidia-smi -lgc N` is a REQUEST, not a guarantee. On this
# 4xH800 box, -lgc 1830 reads back as 1830 MHz at idle -- in 97.9% of idle
# samples -- and then runs at a median of 1380 MHz under sustained fp16 GEMM,
# with sw_power_cap active in 79% of loaded samples at ~600 W. Idle is exactly
# when the power cap is not engaged, so an idle readback can never detect this.
# Every "locked 1830 MHz" line in reports/ came from an idle readback.
#
# A drooping clock does not invalidate ratios measured back-to-back (both sides
# see the same clock) but it does invalidate absolute us / TFLOP/s / GB/s and
# any comparison spanning minutes -- including sweeps, where longer shapes sit
# under load longer and therefore get capped harder. That is a confound that
# looks exactly like a real effect of the shape.

set -u
cd "$(dirname "$0")"

DEV=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
FIRST=${DEV%%,*}
SOAK_S=${SOAK_S:-20}
CANDIDATES=${*:-"1830 1500 1400 1300 1200 1100"}

if ! [ -x ./exp3_epilogue_remote ]; then
  echo "need ./exp3_epilogue_remote -- run: make -j4" >&2
  exit 1
fi

echo "== pick_clock: ${SOAK_S}s soak per candidate on GPU(s) $DEV =="
echo "   (this CHANGES clocks; KEEP=1 leaves the winner locked, else -rgc at exit)"
echo

cleanup() {
  pkill -KILL -f exp3_epilogue_remote 2>/dev/null
  if [ "${KEEP:-0}" != "1" ]; then
    nvidia-smi -rgc -i "$DEV" >/dev/null 2>&1
    echo "clocks reset (-rgc). Re-lock the winner yourself before benchmarking."
  fi
}
trap cleanup EXIT INT TERM

printf '%6s | %8s %8s %8s %8s | %7s | %s\n' \
       "req" "idle" "mean" "min" "p10" "hold%" "verdict"

best=0
for c in $CANDIDATES; do
  nvidia-smi -lgc "$c" -i "$DEV" >/dev/null 2>&1 || {
    printf '%6s | %s\n' "$c" "rejected by driver"; continue; }
  sleep 1
  idle=$(nvidia-smi -i "$FIRST" --query-gpu=clocks.sm --format=csv,noheader,nounits \
         | tr -d ' ')

  # real load: a long single-shape GEMM window, same kernel the suite uses
  ./exp3_epilogue_remote --sizes 8192x8192x8192 --modes local \
      --window-ms $((SOAK_S * 1000 / 2)) --warmup-ms $((SOAK_S * 1000 / 2)) \
      >/dev/null 2>&1 &
  job=$!

  vals=""
  for _ in $(seq 1 $((SOAK_S * 4))); do
    sleep 0.25
    v=$(nvidia-smi -i "$FIRST" --query-gpu=clocks.sm --format=csv,noheader,nounits \
        2>/dev/null | tr -d ' ')
    case "$v" in ''|*[!0-9]*) continue ;; esac
    vals="$vals $v"
  done
  wait $job 2>/dev/null

  # drop the first quarter: that is the ramp, not the sustained state
  n=$(echo $vals | wc -w)
  skip=$((n / 4))
  kept=$(echo $vals | tr ' ' '\n' | tail -n +$((skip + 1)) | sort -n)
  k=$(echo "$kept" | wc -l)
  if [ "$k" -lt 8 ]; then
    printf '%6s | %s\n' "$c" "inconclusive ($k samples)"; continue
  fi
  sum=0
  for v in $kept; do sum=$((sum + v)); done
  mean=$((sum / k))
  min=$(echo "$kept" | head -1)
  p10=$(echo "$kept" | sed -n "$(( (k / 10) + 1 ))p")
  hold=$((100 * mean / c))

  if [ "$hold" -ge 98 ]; then
    verdict="HOLDS"
    [ "$best" = 0 ] && best=$c
  elif [ "$hold" -ge 95 ]; then
    verdict="marginal"
  else
    verdict="capped -- sustains only ~${mean} MHz"
  fi
  printf '%6s | %8s %8s %8s %8s | %6s%% | %s\n' \
         "$c" "$idle" "$mean" "$min" "$p10" "$hold" "$verdict"
done

echo
if [ "$best" != 0 ]; then
  echo "highest sustainable clock: ${best} MHz"
  echo "  nvidia-smi -lgc ${best} -i $DEV"
  echo "  cd comm_comp && CUDA_VISIBLE_DEVICES=$DEV ./run_all.sh"
  [ "${KEEP:-0}" = "1" ] && nvidia-smi -lgc "$best" -i "$DEV" >/dev/null 2>&1 \
    && echo "  (KEEP=1: ${best} MHz left locked)"
else
  echo "no candidate held within 2%. Use the lowest 'sustains only ~N MHz'"
  echo "figure above as the lock, or lower the ladder:"
  echo "  ./pick_clock.sh 1000 900 800"
fi
