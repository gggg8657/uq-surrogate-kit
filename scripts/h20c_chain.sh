#!/usr/bin/env bash
# H20 (clause 3): does the OOD result rest on the same preprocessing bug that
# clause 1's did? Base vs H17-equivariant predictor, same checkpoints, same
# shards, same detectors, same threshold. 8 seeds per arm, because the
# published 47/49 is a ONE-checkpoint number and a verdict needs eight.
#
# Robust against the way the first attempt died (truncated at ~27 of 49 shards
# with no error and no JSON): each (seed, arm) writes its own file, existing
# files are skipped so the chain resumes, and a completion marker is written
# only after every cell exists.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-3}
for k in 0 1 2 3 4 5 6 7; do
  for arm in base eq; do
    out=runs/cons_${arm}_u${k}.json
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    flag=""; [ "$arm" = "eq" ] && flag="--equivariant"
    echo "=== seed $k arm $arm ==="
    CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/eval_consistency.py \
        --ckpts runs/u${k}/best.pt --sigma-source het $flag \
        --out "$out" || { echo "FAILED seed $k arm $arm"; exit 1; }
  done
done
echo "H20c chain done: $(ls runs/cons_base_u*.json runs/cons_eq_u*.json | wc -l)/16 cells"
