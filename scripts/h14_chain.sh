#!/usr/bin/env bash
# H14: gated certificate, 8 seeds, one GPU. Sequential so the timing of the
# other GPU in the lease is untouched and no two runs share a device.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/sel_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY scripts/eval_selective.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --out "$out" || exit 1
done
$PY scripts/agg_selective.py --out runs/selective.json
echo "H14 chain done"
