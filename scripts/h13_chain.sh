#!/usr/bin/env bash
# H13: is the weighted-conformal abstention a property of the shifts, or of the
# clip constant we chose?
#
# `tests/test_conformal.py::test_infinite_weighted_quantile_is_a_clip_property`
# proves an infinite quantile is impossible when clip**2 <= n_cal*alpha/(1-alpha)
# -- 10.67 at n_cal=1024, alpha=0.1 -- and the shipped clip is 20.0. So the
# sweep straddles the bound: 2, 4, 8 and 10 are below it (abstention impossible
# by construction), 20 and 50 above it.
#
# EIGHT seeds, not three. The workspace's own seed-count lesson is that three
# seeds get an effect's sign right and its size wrong; a clause verdict needs 8.
# Writes to NEW filenames so no published conf_u*_het.json is touched.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
GPU=${GPU:-2}
cd "$(dirname "$0")/.."
mkdir -p logs runs
CLIPS="20 2 4 8 10 50"     # shipped value first: rec["weighted"] keeps its meaning
for s in 0 1 2 3 4 5 6 7; do
  [ -f "runs/u$s/best.pt" ] || { echo "skip seed $s: no checkpoint"; continue; }
  out="runs/h13_clip_u${s}_het.json"
  [ -f "$out" ] && { echo "have $out"; continue; }
  echo "== H13 seed $s =="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_conformal.py \
      --ckpts "runs/u$s/best.pt" --sigma-source het \
      --probe-clip $CLIPS --out "$out" \
      > "logs/h13_clip_u${s}.log" 2>&1 \
    || echo "FAILED seed $s (rc=$?)"
done
echo "H13_DONE $(date -Is)"
