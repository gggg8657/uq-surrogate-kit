#!/usr/bin/env bash
# H26: the ceiling of the residual-scale family as a function of its exponent.
# Gated on the PROCESS, not on GPU utilisation: bench_fair idles between fields
# and a utilisation dip can land a job inside a run that is still timing.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}
waited=0
while pgrep -f "scripts/bench_fair.py" >/dev/null 2>&1 \
   || pgrep -f "scripts/bench_speedup.py" >/dev/null 2>&1; do
  [ $((waited % 300)) -eq 0 ] && echo "waiting for the lease (${waited}s)"
  sleep 30; waited=$((waited + 30))
done
echo "lease free; starting on device $DEV"
for k in 0 1 2 3 4 5 6 7; do
  out=runs/rgam_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/eval_resid_gamma.py \
      --ckpt runs/u${k}/best.pt --sigma-source het \
      --h25 runs/rscale_u${k}_het.json --out "$out" \
      || { echo "FAILED seed $k"; exit 1; }
done
echo "H26 done: $(ls runs/rgam_u*_het.json | wc -l)/8"
