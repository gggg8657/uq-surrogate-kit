#!/usr/bin/env bash
# Clause 1 as a VERDICT, not a screen: 8 seeds x 3 sigma sources.
#
# The brief's seed-count lesson: `runs/members.json` could only report C(5,M)
# subset spreads off 5 checkpoints and had to label itself "screen, not
# verdict". Coverage is a clause, so it gets 8 independent trainings per arm.
#
# `const` is the control, not a candidate. Split conformal rescales whatever
# sigma it is handed, so *any* sigma reaches ~90% marginal coverage; if `het`
# lands in the band and `const` also lands in the band, the coverage number is
# the rescaling and not the head. What separates them is sharpness and the
# spread-error correlation, which the same run records.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
GPU=${GPU:-3}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
SOURCES=${SOURCES:-"het cqr const"}
mkdir -p logs runs
for s in $SEEDS; do
  [ -f "runs/u$s/best.pt" ] || { echo "skip seed $s: no checkpoint"; continue; }
  for src in $SOURCES; do
    out="runs/conf_u${s}_${src}.json"
    [ -f "$out" ] && { echo "have $out"; continue; }
    echo "== seed $s / $src =="
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_conformal.py \
        --ckpts "runs/u$s/best.pt" --sigma-source "$src" --out "$out" \
        > "logs/conf_u${s}_${src}.log" 2>&1 \
      || echo "FAILED seed $s $src (rc=$?)"
  done
done
echo "UQ_SEEDS_DONE"
