#!/usr/bin/env bash
# Everything clause 1 and clause 2 need from the single-network model, in order,
# started only once all 8 seeds have written done.json.
#
# Ordering matters for the timing rows: nothing else may be on GPU 3 while
# bench_speedup runs, so the benches go LAST, after the 24 conformal evals.
set -u
PY=${PY:-$HOME/miniforge3/envs/pdeno/bin/python}
GPU=${GPU:-3}
cd "$(dirname "$0")/.."
mkdir -p logs

n=0
while [ "$n" -lt 8 ]; do
  n=$(ls runs/u*/done.json 2>/dev/null | wc -l)
  [ "$n" -lt 8 ] && sleep 30
done
echo "all 8 seeds trained at $(date -Is)"

GPU=$GPU bash scripts/eval_uq_seeds.sh 2>&1 | tail -40
$PY scripts/agg_uq_seeds.py --out runs/uq_seeds.json

# clause 2: the shipped model's own speedup. The 108.5x in runs/members.json is
# a 1-channel ensemble MEMBER and does not transfer -- codex was right about
# that and this is the run that settles it. Same solver, same batches, same
# iteration counts and same clock ramp as runs/bench.json, so the rows are
# directly comparable; only the timed model changes.
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/bench_speedup.py \
    --ckpts runs/u0/best.pt --uq-source het --out runs/bench_uq.json \
    > logs/bench_uq.log 2>&1 || echo "bench_uq FAILED rc=$?"

# clause 3 on the same model, so all three clauses are read off one checkpoint
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_consistency.py \
    --ckpts runs/u0/best.pt --sigma-source het --out runs/consistency_uq.json \
    > logs/eval_cons_uq.log 2>&1 || echo "cons_uq FAILED rc=$?"

echo "CHAIN_DONE $(date -Is)"
