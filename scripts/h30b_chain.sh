#!/usr/bin/env bash
# H30: price clause 1 under shift in labels, two estimators on the same draws.
#
# NAMED h30b BECAUSE h30_chain.sh COLLIDED. Two instances of this brief run in
# this repository and both numbered their hypotheses from the same critique log,
# so both created scripts/h30_chain.sh; the peer's cleanup deleted it while this
# chain was on seed 6. See COORDINATION.md.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-3}
./scripts/lease_wait.sh
echo "starting on device $DEV"
for k in 0 1 2 3 4 5 6 7; do
  out=runs/label_budget_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/eval_label_budget.py \
      --ckpts runs/u${k}/best.pt --sigma-source het --out "$out" \
      > logs/label_budget_u${k}.log 2>&1 \
      || { echo "FAILED seed $k"; exit 1; }
  echo "  wrote $out"
done
echo "H30 done: $(ls runs/label_budget_u*_het.json | wc -l)/8"
