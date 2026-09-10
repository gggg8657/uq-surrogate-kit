#!/usr/bin/env bash
# H20 (per-family width model): one width model per family, so one family's shift axis cannot drive
# another's. 8 seeds, same checkpoints as H15/H18 so the pairing is exact.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/scalepf_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --per-family-h \
      --out "$out" || exit 1
done
echo "H19 chain done"
