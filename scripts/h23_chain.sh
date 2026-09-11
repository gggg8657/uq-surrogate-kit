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

hits=0
while [ "$hits" -lt 3 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$DEV")
  if [ "$u" -lt 10 ]; then hits=$((hits+1)); else hits=0; fi
  sleep 20
done
echo "device $DEV idle; starting"

for k in 0 1 2 3 4 5 6 7; do
  out=runs/feat_coverage_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/diag_feature_coverage.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --out "$out" \
      || { echo "FAILED seed $k"; exit 1; }
done
echo "H23 done: $(ls runs/feat_coverage_u*_het.json | wc -l)/8"
