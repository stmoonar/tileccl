#!/usr/bin/env bash

# Standalone Exp4 panel-granularity runner. Deliberately does not enable
# `set -e`: a failed build/case is recorded and existing results are packaged.
# GPU clocks must already be configured by the host environment.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! cd "$SCRIPT_DIR"; then
  echo "cannot enter $SCRIPT_DIR"
else

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT="exp4_panel_${STAMP}"
FAILED=0
SAMPLER_PID=""

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

trap 'stop_sampler' INT TERM

{
  date -u
  git rev-parse HEAD
  git status --short
  nvcc --version
  nvidia-smi -L
  nvidia-smi topo -m
} > "$OUT/env.txt" 2>&1

nvidia-smi \
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
  printf 'timestamp,index,clocks_sm,power_w,temp_c,util_pct\n' \
    > "$OUT/gpu_trace.csv"

  (
    while true; do
      TS=$(date +%s.%N)
      nvidia-smi \
        --query-gpu=index,clocks.sm,power.draw,temperature.gpu,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null |
        sed "s/^/${TS},/"
      sleep 0.1
    done
  ) >> "$OUT/gpu_trace.csv" &
  SAMPLER_PID=$!

  run_case exp4_panel_verify 900 \
    ./exp4_ag_tile_transport \
    --ndev 4 \
    --k 1024 \
    --panel-h 1,4,16 \
    --n-comm 4 \
    --iters 5 \
    --verify \
    --modes fused,compute-only,comm-only

  run_case exp4_panel 7200 \
    ./exp4_ag_tile_transport \
    --ndev 4 \
    --k 8192 \
    --panel-h 1,2,4,8,16,32,64,128 \
    --n-comm 8 \
    --verify \
    --modes fused,compute-only,comm-only,bystander,local,memop-cost \
    --warmup-ms 500 \
    --window-ms 200 \
    --csv "$OUT/exp4_panel.csv"

  run_case exp4_panel_ce_isosm 3600 \
    ./exp4_ag_tile_transport \
    --ndev 4 \
    --k 8192 \
    --panel-h 1,8,128 \
    --variants ce-aggregate,ce-panelized \
    --ce-reserve-sm 8 \
    --verify \
    --modes fused,compute-only,bystander,memop-cost \
    --warmup-ms 500 \
    --window-ms 200 \
    --csv "$OUT/exp4_panel_ce_isosm.csv"

  run_case exp4_panel_ncomm 3600 \
    ./exp4_ag_tile_transport \
    --ndev 4 \
    --k 8192 \
    --panel-h 1,8,128 \
    --variants tma \
    --n-comm 1,2,4,8,16 \
    --verify \
    --modes fused,compute-only,comm-only,bystander \
    --warmup-ms 500 \
    --window-ms 200 \
    --csv "$OUT/exp4_panel_ncomm.csv"

  stop_sampler
fi

nvidia-smi \
  --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu \
  --format=csv > "$OUT/clocks_after.csv" 2>&1

cp \
  exp4_ag_tile_transport.cu \
  exp4_transport.cuh \
  Makefile \
  README.md \
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
