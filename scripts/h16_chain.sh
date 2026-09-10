#!/usr/bin/env bash
# H16: the width tolerance of the score, 8 seeds, GPU 3.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/wtol_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} $PY -u \
      scripts/eval_width_tolerance.py --ckpt runs/u${k}/best.pt \
      --sigma-source het --out "$out" || exit 1
done
echo "H16 chain done"
