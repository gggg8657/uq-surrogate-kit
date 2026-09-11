#!/usr/bin/env bash
# H23: univariate rank correlations that separate suppression from a real
# statement about the sigma head. No fitting; diagnostic only.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for k in 0 1 2 3 4 5 6 7; do
  out=runs/rcorr_u${k}_het.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} $PY -u scripts/resid_corr.py \
      --ckpt runs/u${k}/best.pt --out "$out" || exit 1
done
echo "H23 rcorr done: $(ls runs/rcorr_u*_het.json | wc -l)/8"
