#!/usr/bin/env bash

# Standalone Exp4 Fig2-left runner: only compute-only and fused are measured.
# Deliberately does not enable
# `set -e`: a failed build/case is recorded and existing results are packaged.
# GPU clocks are locked without privilege escalation. A lock/hold failure skips
# the formal cases but still packages diagnostics and never exits the shell.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! cd "$SCRIPT_DIR"; then
  echo "cannot enter $SCRIPT_DIR"
else

EXP4_GPU_IDS=${EXP4_GPU_IDS:-0,1,2,3}
EXP4_CLOCK_MHZ=${EXP4_CLOCK_MHZ:-1300}
EXP4_CLOCK_TOLERANCE_MHZ=${EXP4_CLOCK_TOLERANCE_MHZ:-15}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$EXP4_GPU_IDS}

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT="exp4_fig2_left_${STAMP}"
FAILED=0
SAMPLER_PID=""
CLOCKS_LOCKED=0
CLOCK_VALID=0
LAST_CASE_RC=0

mkdir -p "$OUT"

run_case() {
  NAME=$1
  LIMIT=$2
  shift 2

  echo
  echo "=== $NAME ==="
  echo "command: $*"

  if command -v timeout >/dev/null 2>&1; then
    timeout "$LIMIT" "$@" 2>&1 | tee "$OUT/${NAME}.log"
    RC=${PIPESTATUS[0]}
  else
    "$@" 2>&1 | tee "$OUT/${NAME}.log"
    RC=${PIPESTATUS[0]}
  fi

  echo "$RC" > "$OUT/${NAME}.exitcode"
  LAST_CASE_RC=$RC
  if [ "$RC" -ne 0 ]; then
    echo "!! $NAME failed: rc=$RC"
    FAILED=1
  else
    echo "-- $NAME completed"
  fi
  return 0
}

stop_sampler() {
  if [ -n "${SAMPLER_PID:-}" ]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi
}

restore_clocks() {
  if [ "$CLOCKS_LOCKED" -eq 1 ]; then
    nvidia-smi -rgc -i "$EXP4_GPU_IDS" \
      > "$OUT/clock_restore.log" 2>&1
    RESTORE_RC=$?
    echo "$RESTORE_RC" > "$OUT/clock_restore.exitcode"
    CLOCKS_LOCKED=0
    if [ "$RESTORE_RC" -ne 0 ]; then
      echo "!! clock restore failed: rc=$RESTORE_RC"
      FAILED=1
    fi
  fi
}

lock_clocks() {
  echo "=== lock clocks: ${EXP4_CLOCK_MHZ} MHz on ${EXP4_GPU_IDS} ==="
  nvidia-smi -lgc "$EXP4_CLOCK_MHZ" -i "$EXP4_GPU_IDS" \
    > "$OUT/clock_lock.log" 2>&1
  LOCK_RC=$?
  echo "$LOCK_RC" > "$OUT/clock_lock.exitcode"
  if [ "$LOCK_RC" -ne 0 ]; then
    echo "!! clock lock failed: rc=$LOCK_RC"
    FAILED=1
    CLOCK_VALID=0
    return 0
  fi

  CLOCKS_LOCKED=1
  nvidia-smi -i "$EXP4_GPU_IDS" \
    --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
    --format=csv > "$OUT/clocks_locked.csv" 2>&1

  BAD_CLOCKS=$(nvidia-smi -i "$EXP4_GPU_IDS" \
    --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null |
    awk -v target="$EXP4_CLOCK_MHZ" -v tol="$EXP4_CLOCK_TOLERANCE_MHZ" '
      { d = $1 - target; if (d < 0) d = -d; if (d > tol) bad++ }
      END { print bad + 0 }
    ')
  if [ "${BAD_CLOCKS:-1}" -ne 0 ]; then
    echo "!! requested clock is not held at idle"
    FAILED=1
    CLOCK_VALID=0
  else
    CLOCK_VALID=1
  fi
  return 0
}

validate_active_clocks() {
  ACTIVE_CHECK=$(awk -F, \
    -v target="$EXP4_CLOCK_MHZ" \
    -v tol="$EXP4_CLOCK_TOLERANCE_MHZ" '
      NR > 1 {
        clk = $3 + 0; util = $6 + 0;
        if (util >= 50) {
          active++;
          d = clk - target; if (d < 0) d = -d;
          if (d > tol) bad++;
        }
      }
      END { print active + 0, bad + 0 }
    ' "$OUT/gpu_trace.csv")
  ACTIVE_SAMPLES=${ACTIVE_CHECK%% *}
  BAD_ACTIVE_SAMPLES=${ACTIVE_CHECK##* }
  echo "active clock samples: $ACTIVE_SAMPLES; out of tolerance: $BAD_ACTIVE_SAMPLES" \
    | tee "$OUT/clock_hold_check.txt"
  if [ "$ACTIVE_SAMPLES" -eq 0 ] || [ "$BAD_ACTIVE_SAMPLES" -ne 0 ]; then
    echo "!! clock did not hold under load"
    FAILED=1
    CLOCK_VALID=0
  else
    CLOCK_VALID=1
  fi
  return 0
}

trap 'stop_sampler; restore_clocks' INT TERM

{
  date -u
  git rev-parse HEAD
  git status --short
  nvcc --version
  nvidia-smi -L
  nvidia-smi topo -m
} > "$OUT/env.txt" 2>&1

nvidia-smi -i "$EXP4_GPU_IDS" \
  --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv > "$OUT/clocks_before.csv" 2>&1

echo "=== build ==="
make exp4_ag_tile_transport -j4 2>&1 | tee "$OUT/build.log"
BUILD_RC=${PIPESTATUS[0]}
echo "$BUILD_RC" > "$OUT/build.exitcode"

if [ "$BUILD_RC" -ne 0 ]; then
  echo "!! build failed: rc=$BUILD_RC"
  FAILED=1
else
  lock_clocks
  if [ "$CLOCK_VALID" -ne 1 ]; then
    echo "!! formal experiments skipped because clock lock failed"
  else
    printf 'timestamp,index,clocks_sm,power_w,temp_c,util_pct\n' \
      > "$OUT/gpu_trace.csv"

    (
      while true; do
        TS=$(date +%s.%N)
        nvidia-smi -i "$EXP4_GPU_IDS" \
          --query-gpu=index,clocks.sm,power.draw,temperature.gpu,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null |
          sed "s/^/${TS},/"
        sleep 0.1
      done
    ) >> "$OUT/gpu_trace.csv" &
    SAMPLER_PID=$!

    run_case exp4_fig2_left_verify 900 \
      ./exp4_ag_tile_transport \
      --ndev 4 \
      --k 1024 \
      --panel-h 1,16,64,256 \
      --n-comm 16 \
      --comm-streams 3 \
      --variants ce-aggregate,tma \
      --iters 5 \
      --warmup-ms 10 \
      --verify \
      --modes fused,compute-only

    validate_active_clocks
    if [ "$LAST_CASE_RC" -ne 0 ] || [ "$CLOCK_VALID" -ne 1 ]; then
      echo "!! formal experiments skipped after verify/clock-hold failure"
    else
      run_case exp4_fig2_left 7200 \
        ./exp4_ag_tile_transport \
        --ndev 4 \
        --m 65536 \
        --k 8192 \
        --panel-h 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
        --n-comm 16 \
        --comm-streams 3 \
        --variants ce-aggregate,tma \
        --verify \
        --modes fused,compute-only \
        --warmup-ms 500 \
        --window-ms 5000 \
        --csv "$OUT/exp4_fig2_left.csv"

      validate_active_clocks
    fi
    stop_sampler
  fi
fi

nvidia-smi -i "$EXP4_GPU_IDS" \
  --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv > "$OUT/clocks_after.csv" 2>&1

restore_clocks
nvidia-smi -i "$EXP4_GPU_IDS" \
  --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv > "$OUT/clocks_restored.csv" 2>&1

cp \
  exp4_ag_tile_transport.cu \
  exp4_transport.cuh \
  Makefile \
  README.md \
  plot_exp4_fig2_left.py \
  run_all.sh \
  run_exp4_panel.sh \
  "$OUT/" 2>> "$OUT/package.log" || FAILED=1

if [ -f exp4_ag_tile_transport ]; then
  cp exp4_ag_tile_transport "$OUT/" 2>> "$OUT/package.log" || FAILED=1
fi

git diff --binary > "$OUT/source.diff"
git log -1 --oneline > "$OUT/commit.txt"

find "$OUT" \
  -maxdepth 1 \
  -type f \
  ! -name SHA256SUMS \
  -print0 |
  sort -z |
  xargs -0 sha256sum > "$OUT/SHA256SUMS"

ARCHIVE=""
if command -v zip >/dev/null 2>&1; then
  ARCHIVE="${OUT}.zip"
  zip -qr "$ARCHIVE" "$OUT"
  PACKAGE_RC=$?
else
  ARCHIVE="${OUT}.tar.gz"
  tar czf "$ARCHIVE" "$OUT"
  PACKAGE_RC=$?
fi

if [ "$PACKAGE_RC" -ne 0 ]; then
  echo "!! packaging failed: rc=$PACKAGE_RC"
  FAILED=1
else
  sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
  echo "result: $(pwd)/$ARCHIVE"
fi

echo
if [ "$FAILED" -ne 0 ]; then
  echo "some steps failed; inspect *.exitcode and matching logs"
else
  echo "all experiments and packaging completed"
fi

# No explicit exit: when invoked from an interactive container shell, a case
# failure cannot terminate that shell or the container's long-lived process.
fi
