#!/usr/bin/env bash
# H32: the regime gate. Same protocol as H31 plus --gate, so every file carries
# both the c-ladder and the gate decision and the gated method is reconstructed
# by the aggregator without a second run.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PCT=50,75,90,95,99,100
for arm in eq base; do
  flag=""; tag="gtbase"
  if [ "$arm" = "eq" ]; then flag="--equivariant"; tag="gteq"; fi
  for k in 0 1 2 3 4 5 6 7; do
    out=runs/${tag}_u${k}_het.json
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    echo "=== arm $arm seed $k ($(date -u +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/eval_scale_target.py \
        --ckpt runs/u${k}/best.pt --sigma-source het $flag \
        --onesided --gate --dev-n 256 --c-pct "$PCT" \
        --out "$out" || { echo "FAILED $arm seed $k"; exit 1; }
  done
done
echo "H32 done: eq $(ls runs/ | grep -c '^gteq_')/8  base $(ls runs/ | grep -c '^gtbase_')/8"
