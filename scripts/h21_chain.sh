#!/usr/bin/env bash
# H21: clause 3 re-measured on the equivariant predictor, 8 seeds per arm.
# Both arms run SEQUENTIALLY on one GPU: the other half of the lease is busy with
# this track's own per-family H20 run, and the brief says queue rather than
# oversubscribe. The arms share checkpoints, so the comparison is paired.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
for ARM in base eq; do
flag=""; [ "$ARM" = eq ] && flag="--equivariant"
for k in 0 1 2 3 4 5 6 7; do
  out=runs/cons3_${ARM}_u${k}_het.json
  [ -s "$out" ] && { echo "skip $out"; continue; }
  echo "=== seed $k arm $ARM ==="
  $PY -u scripts/eval_consistency.py --ckpts runs/u${k}/best.pt \
      --sigma-source het $flag --out "$out" || exit 1
done
done
echo "H21 chain done (both arms)"
