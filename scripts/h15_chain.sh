#!/usr/bin/env bash
# H15: the width-rescaling reading of clause 1 under shift, 8 seeds.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/scale_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --out "$out" || exit 1
done
echo "H15 chain done"
