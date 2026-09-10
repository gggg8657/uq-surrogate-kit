#!/usr/bin/env bash
# H19: the H15 arm with the two input-amplitude features ablated out of z.
# No equivariant wrapper, same checkpoints and dev suite as H15 and H18, so all
# three arms pair seed by seed and the sign-flip tests are exact.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/scalena_u${k}_het.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het \
      --drop-features a_spec9,a_spec10 --out "$out" || exit 1
done
echo "H19 chain done"
