#!/usr/bin/env bash
# H31: the one-sided conservative reading. s* on the DEVELOPMENT shift suite
# chooses a ladder of inflation constants c; coverage AND realized median
# interval width are then recorded at each c on the 24 evaluation shards and on
# the three in-distribution families.
#
# Both prediction paths, 8 seeds each, equivariant first because it is the
# headline (H29: the wrapper takes poisson_amp2 83.26 -> 1.014).
#
# Thread discipline: one process at a time, OMP_NUM_THREADS=4, so this chain
# claims 4 of the track's 48-core lease and nothing else. The work is on GPU 2.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
cd "$(dirname "$0")/.."
DEV=${CUDA_VISIBLE_DEVICES:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PCT=50,75,90,95,99,100
for arm in eq base; do
  flag=""; tag="osbase"
  if [ "$arm" = "eq" ]; then flag="--equivariant"; tag="oseq"; fi
  for k in 0 1 2 3 4 5 6 7; do
    out=runs/${tag}_u${k}_het.json
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    echo "=== arm $arm seed $k ($(date -u +%H:%M:%S)) ==="
    CUDA_VISIBLE_DEVICES=$DEV $PY -u scripts/eval_scale_target.py \
        --ckpt runs/u${k}/best.pt --sigma-source het $flag \
        --onesided --dev-n 256 --c-pct "$PCT" \
        --out "$out" || { echo "FAILED $arm seed $k"; exit 1; }
  done
done
echo "H31 done: eq $(ls runs/oseq_u*_het.json 2>/dev/null | wc -l)/8  base $(ls runs/osbase_u*_het.json 2>/dev/null | wc -l)/8"
