#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=3
CK=(runs/m*/best.pt)
$PY scripts/eval_label_probe.py --ckpts "${CK[@]}" > logs/eval_label_probe.log 2>&1 \
  || echo "!!! label_probe failed"
$PY scripts/ablate_members.py --ckpts "${CK[@]}" > logs/ablate_members.log 2>&1 \
  || echo "!!! ablate failed"
echo ALLDONE
