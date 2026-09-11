#!/usr/bin/env bash
# H24: the H22 residual arm with the two residual coefficients constrained to
# be >= 0. Pairs exactly with runs/scalerf_u*_het.json -- same shards, families,
# dev suite, folds and checkpoints -- differing only in the constraint.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/h24_nn_u${k}_het.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het \
      --restrict-families poisson,helmholtz,darcy --residual-features \
      --nonneg-features log_resid,log_consist --out "$out" || exit 1
done
echo "H24 done: $(ls runs/h24_nn_u*_het.json | wc -l)/8"
