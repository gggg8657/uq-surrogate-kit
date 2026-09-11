#!/usr/bin/env bash
# H22 CONTROL: the same three families, the same fitting set, WITHOUT the two
# residual features.
#
# Why this arm has to exist. `--residual-features` does two things at once: it
# adds two columns to z AND forces the fit onto the three families that have a
# cheap operator apply. Comparing that against the 5-family H15 arm would move
# two variables, so a gain could be the features or could be the narrower
# fitting set. This arm holds the fitting set fixed and removes only the
# features, so H22 minus H22-control is exactly the two residual columns.
#
# Queues behind the H22 chain on the same device rather than running beside it:
# the lease is two GPUs and both are busy with this track's own jobs.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}
FAMS=poisson,helmholtz,darcy

# wait for the H22 arm to finish all 8 seeds before starting
while [ "$(ls runs/scalerf_u*_het.json 2>/dev/null | wc -l)" -lt 8 ]; do sleep 30; done
echo "H22 arm complete; starting control $(date -Is)"

for k in 0 1 2 3 4 5 6 7; do
  out=runs/scalerc_u${k}_het.json
  if [ -s "$out" ]; then echo "skip $out"; continue; fi
  echo "=== seed $k ==="
  CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/fit_scale.py \
      --ckpt runs/u${k}/best.pt --sigma-source het \
      --restrict-families "$FAMS" --out "$out" || { echo "FAILED $k"; exit 1; }
done
echo "H22 control done: $(ls runs/scalerc_u*_het.json | wc -l)/8"
