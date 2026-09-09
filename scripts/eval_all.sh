#!/usr/bin/env bash
# Every measurement stage, in order, on the one usable leased GPU.
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=3          # GPU 2 is degraded 15.3x, see critique_log
CK=$(ls runs/m*/best.pt | tr '\n' ' ')
echo "=== ckpts: $CK"
for stage in "eval_conformal.py --ckpts $CK" \
             "eval_ood.py --ckpts $CK" \
             "bench_speedup.py --ckpts $CK" \
             "bench_isoaccuracy.py --ckpts $CK" \
             "measure_residual_floor.py"; do
  name=$(echo "$stage" | awk '{print $1}' | sed 's/.py//')
  echo "=== $name $(date -Is)"
  $PY scripts/$stage > "logs/$name.log" 2>&1 || echo "!!! $name FAILED rc=$?"
  tail -3 "logs/$name.log"
done
$PY scripts/report.py
echo "=== all stages done $(date -Is)"
