"""Train one ensemble member: a task-conditioned FNO over the 5 trained families.

    CUDA_VISIBLE_DEVICES=2 python scripts/train_member.py --seed 0 --out runs/m0

One member per process, one process per GPU, because the members are
independent by construction -- the whole point of a deep ensemble is that they
differ, so there is nothing to synchronize and DDP would only add an all-reduce
that has to be avoided.

The members differ in *initialization and data order only* (one seed drives
both). They see the same data, which is the standard deep-ensemble recipe and
the honest one: bootstrapping the data as well would conflate epistemic spread
with a smaller effective training set, and the spread would look better for the
wrong reason.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.sims.dataset import GPUCorpus, denorm_u, load_corpus  # noqa: E402
from uqkit.sims.fno2d import FNO2d  # noqa: E402
from uqkit.sims.pde2d import PRETRAIN_TASKS, TASK_ID  # noqa: E402


@torch.no_grad()
def evaluate(model, shards, device, batch=64):
    model.eval()
    out = {}
    for s in shards:
        st = s.stats
        errs = []
        for i in range(0, len(s), batch):
            a = ((s.a[i : i + batch] - st["a_mean"]) / st["a_std"]).to(device)
            u_true = s.u[i : i + batch].to(device)
            tid = torch.full((a.shape[0],), s.tid, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(a, tid)
            errs.append(rel_l2(denorm_u(pred.float(), st), u_true))
        out[s.task] = float(torch.cat(errs).mean())
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default=",".join(PRETRAIN_TASKS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--n-tasks", type=int, default=len(TASK_ID) + 1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")

    tr_shards, stats = load_corpus(args.root, tasks, "train", 64)
    va_shards, _ = load_corpus(args.root, tasks, "cal", 64, stats=stats)
    corpus = GPUCorpus(tr_shards, device)

    model = FNO2d(width=args.width, modes=args.modes, n_layers=args.layers,
                  n_tasks=args.n_tasks).to(device)
    n_par = model.param_count()
    steps_per_epoch = corpus.steps_per_epoch(args.batch)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    print(f"member seed={args.seed} params={n_par/1e6:.2f}M steps={total}", flush=True)
    logf = (out / "log.jsonl").open("w")
    best = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        run, nb = 0.0, 0
        for a, u, tid in corpus.epoch_batches(args.batch, ep, seed=args.seed):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = rel_l2(model(a, tid).float(), u, reduce="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            run += float(loss.detach())
            nb += 1
        rec = {"epoch": ep, "train_rel_l2": run / max(nb, 1),
               "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            val = evaluate(model, va_shards, device)
            rec["cal_rel_l2"] = val
            rec["cal_mean"] = sum(val.values()) / len(val)
            if rec["cal_mean"] < best:
                best = rec["cal_mean"]
                torch.save({"model": model.state_dict(),
                            "args": vars(args) | {"width": args.width,
                                                  "modes": args.modes,
                                                  "layers": args.layers,
                                                  "n_tasks": args.n_tasks},
                            "stats": stats, "tasks": tasks, "epoch": ep,
                            "cal_mean": best, "params": n_par}, out / "best.pt")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)
    logf.close()
    print(f"done in {time.time() - t0:.0f}s  best cal mean rel-L2 {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
