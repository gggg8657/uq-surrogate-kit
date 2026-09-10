#!/usr/bin/env bash
# H16/H17: width tolerance and the scale-equivariant arm, 8 seeds each, GPU 3.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  for arm in base eq; do
    out=runs/wtol_${arm}_u${k}_het.json
    [ -s "$out" ] && { echo "skip $out"; continue; }
    flag=""; [ "$arm" = eq ] && flag="--equivariant"
    echo "=== seed $k arm $arm ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} $PY -u \
        scripts/eval_width_tolerance.py --ckpt runs/u${k}/best.pt \
        --sigma-source het $flag --out "$out" || exit 1
  done
done
echo "H16/H17 chain done"
