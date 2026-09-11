#!/usr/bin/env bash
# H23: how far the width model extrapolates its residual features, per family.
#
# QUEUES. GPU 2 of the lease is running a TIMING benchmark, where adding load
# would corrupt the measurement rather than merely slow it, and GPU 3 is running
# this track's clause-3 chain. Waits for its device to fall idle first.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}

# Gate on the TIMING BENCHMARK PROCESS, not on GPU utilisation.
#
# A utilisation gate was the first version and it is unsafe here: bench_fair
# alternates solver work with idle gaps between fields, so three consecutive
# sub-10% samples can land inside a run that is still going. Starting then does
# not merely slow the benchmark, it corrupts the number it exists to produce --
# and this repo has already withdrawn one speedup figure for being measured
# under conditions it did not state. Waiting on the process is exact.
wait_for_lease() {
  local waited=0
  while pgrep -f "scripts/bench_fair.py" >/dev/null 2>&1; do
    if [ $((waited % 300)) -eq 0 ]; then
      echo "waiting: a timing benchmark holds the lease (${waited}s)"
    fi
    sleep 30; waited=$((waited + 30))
  done
  # and do not stack on another of this track's own fits
  while pgrep -f "scripts/fit_scale.py" >/dev/null 2>&1; do
    echo "waiting: a fit holds the lease"
    sleep 30
  done
  echo "lease free; starting on device $DEV"
}

wait_for_lease

for k in 0 1 2 3 4 5 6 7; do
  out=runs/feat_coverage_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/diag_feature_coverage.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --out "$out" \
      || { echo "FAILED seed $k"; exit 1; }
done
echo "H23 done: $(ls runs/feat_coverage_u*_het.json | wc -l)/8"
