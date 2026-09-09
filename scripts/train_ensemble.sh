#!/usr/bin/env bash
# Five members, two lanes, one GPU each. Members differ by seed only.
#   bash scripts/train_ensemble.sh [epochs]
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
EPOCHS=${1:-60}
lane () {   # $1 = gpu, rest = seeds
  local gpu=$1; shift
  for s in "$@"; do
    echo "[lane $gpu] member $s start $(date -Is)"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_member.py \
      --seed "$s" --out "runs/m$s" --epochs "$EPOCHS" > "logs/train_m$s.log" 2>&1
    echo "[lane $gpu] member $s done  $(date -Is) rc=$?"
  done
}
# GPU 2 is held at 2.88 TFLOP/s by 91 leaked CUDA contexts that are not ours
# (measured 2026-09-09, see critique_log.md turn 1); everything runs on GPU 3.
lane 3 0 1 2 3 4 &

wait
echo "all members done $(date -Is)"
