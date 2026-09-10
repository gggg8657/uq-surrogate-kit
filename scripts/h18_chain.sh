#!/usr/bin/env bash
# H18: compose the H17 equivariant wrapper with the H15 width model, 8 seeds.
# Same checkpoints and same dev-n as the H15 arm, so the two pair exactly and
# the sign-flip test against runs/scale_u*_het.json is a paired test.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/scaleq_u${k}_het.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --equivariant \
      --out "$out" || exit 1
done
echo "H18 chain done"
