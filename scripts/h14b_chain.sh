#!/usr/bin/env bash
# H14b: the abstention price curve, 8 seeds, GPU 3 (the other half of the lease).
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/selp_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} $PY \
      scripts/eval_selective_price.py --ckpt runs/u${k}/best.pt \
      --sigma-source het --out "$out" || exit 1
done
echo "H14b chain done"
