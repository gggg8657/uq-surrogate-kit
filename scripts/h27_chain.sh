#!/usr/bin/env bash
# H27: bound the ceiling of every per-sample scalar modulation of sigma.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}
./scripts/lease_wait.sh
echo "starting on device $DEV"
for k in 0 1 2 3 4 5 6 7; do
  out=runs/starget_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/eval_scale_target.py \
      --ckpt runs/u${k}/best.pt --sigma-source het --out "$out" \
      || { echo "FAILED seed $k"; exit 1; }
done
echo "H27 done: $(ls runs/starget_u*_het.json | wc -l)/8"
