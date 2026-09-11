#!/usr/bin/env bash
# H22 control arm: the same 3-family restriction the residual features force,
# WITHOUT the features. Without this, the residual arm (runs/scalerf_u*) would
# be compared against a 5-family, 32-shard baseline and would differ in two
# things at once -- the features and the fitting set.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/h22_ctl_u${k}_het.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "=== ctl seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het \
      --restrict-families poisson,helmholtz,darcy --out "$out" || exit 1
done
echo "H22 control done: $(ls runs/h22_ctl_u*_het.json | wc -l)/8"
