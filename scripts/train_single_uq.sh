#!/usr/bin/env bash
# 8 seeds of the single-network UQ model on ONE GPU, 4 at a time.
#
# 8 and not 3: the brief's seed-count lesson. `runs/members.json` could only
# report C(5,M) subset spreads off 5 checkpoints and had to label itself
# "screen, not verdict"; the coverage clause is a verdict, so it gets 8 arms.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
GPU=${GPU:-3}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
PAR=${PAR:-4}
mkdir -p logs
echo "$SEEDS" | tr ' ' '\n' | xargs -P "$PAR" -I{} sh -c \
  "CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_single_uq.py --seed {} \
     --out runs/u{} > logs/train_u{}.log 2>&1; echo seed {} rc=\$?"
