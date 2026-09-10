"""Train ONE network that emits mean + sigma + quantiles in one forward pass.

    CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/pdeno/bin/python \
        scripts/train_single_uq.py --seed 0 --out runs/u0

Same corpus, same architecture trunk, same optimizer, same 60 epochs and the
same `rel_l2` loss on the mean as `scripts/train_member.py`. The only
difference is the 3-channel uncertainty head and its two losses, both of which
take a **detached** residual and a **detached** trunk -- see
`uqkit/sims/fno2d_uq.py` for why that detachment is the point rather than a
convenience.

Hypothesis H5, written into `critique_log.md` before this file ran: this puts
the coverage row and the >=100x speedup row on the same model, because the head
costs three output channels and one extra 1x1 projection and nothing else.
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
from uqkit.sims.fno2d_uq import (FNO2dUQ, nll_loss, pinball_loss,  # noqa: E402
                                 sigma_from, split_heads)
from uqkit.sims.pde2d import PRETRAIN_TASKS, TASK_ID  # noqa: E402


@torch.no_grad()
def evaluate(model, shards, device, alpha, batch=64):
    """Mean rel-L2 per family, plus the head's own diagnostics.

    `pit` is the fraction of pixels inside [q_lo, q_hi] *before* any conformal
    correction. It is not a coverage claim -- the claim is what
    `eval_conformal.py` measures after calibration -- but if the head has
    learned nothing it sits at 0 or 1 and says so several minutes earlier than
    the eval does.
    """
    model.eval()
    out, diag = {}, {}
    for s in shards:
        st = s.stats
        errs, nll, pit, sig = [], [], [], []
        for i in range(0, len(s), batch):
            a = ((s.a[i : i + batch] - st["a_mean"]) / st["a_std"]).to(device)
            u_true = s.u[i : i + batch].to(device)
            u_n = ((s.u[i : i + batch] - st["u_mean"]) / st["u_std"]).to(device)
            tid = torch.full((a.shape[0],), s.tid, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(a, tid)
            y = y.float()
            mean_n, log_s, d_lo, d_hi = split_heads(y)
            errs.append(rel_l2(denorm_u(mean_n, st), u_true))
            r = u_n - mean_n
            nll.append(nll_loss(log_s, r).item())
            pit.append((((r >= d_lo) & (r <= d_hi)).float().mean()).item())
            sig.append(log_s.exp().mean().item())
        out[s.task] = float(torch.cat(errs).mean())
        diag[s.task] = {"nll": sum(nll) / len(nll), "pit_pixel": sum(pit) / len(pit),
                        "sigma_mean_n": sum(sig) / len(sig)}
    model.train()
    return out, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default=",".join(PRETRAIN_TASKS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
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
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--w-nll", type=float, default=1.0)
    p.add_argument("--w-pinball", type=float, default=1.0)
    p.add_argument("--no-detach-uq", action="store_true",
                   help="let the uncertainty loss train the shared trunk too. "
                        "Changes the mean, so an arm run this way is NOT "
                        "comparable to the ensemble baseline without saying so.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")
    q_lo, q_hi = args.alpha / 2, 1.0 - args.alpha / 2

    tr_shards, stats = load_corpus(args.root, tasks, "train", 64)
    va_shards, _ = load_corpus(args.root, tasks, "cal", 64, stats=stats)
    corpus = GPUCorpus(tr_shards, device)

    model = FNO2dUQ(width=args.width, modes=args.modes, n_layers=args.layers,
                    n_tasks=args.n_tasks,
                    detach_uq=not args.no_detach_uq).to(device)
    n_par = model.param_count()
    n_par_uq = sum(q.numel() for q in model.proj_uq.parameters())
    steps_per_epoch = corpus.steps_per_epoch(args.batch)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    print(f"uq seed={args.seed} params={n_par/1e6:.2f}M (head {n_par_uq/1e6:.3f}M) "
          f"steps={total} detach_uq={not args.no_detach_uq}", flush=True)
    logf = (out / "log.jsonl").open("w")
    best = float("inf")
    t0 = time.time()
    ckpt_args = vars(args) | {"width": args.width, "modes": args.modes,
                              "layers": args.layers, "n_tasks": args.n_tasks,
                              "uq": True, "detach_uq": not args.no_detach_uq,
                              "out_ch": 4}
    for ep in range(args.epochs):
        run = {"mean": 0.0, "nll": 0.0, "pin": 0.0}
        nb = 0
        for a, u, tid in corpus.epoch_batches(args.batch, ep, seed=args.seed):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(a, tid)
            y = y.float()
            mean_n, log_s, d_lo, d_hi = split_heads(y)
            l_mean = rel_l2(mean_n, u, reduce="mean")
            # detached: the head is fitted to the mean's error, it does not get
            # to move the mean to make its own job easier
            r = (u - mean_n).detach()
            l_nll = nll_loss(log_s, r)
            l_pin = pinball_loss(d_lo, r, q_lo) + pinball_loss(d_hi, r, q_hi)
            loss = l_mean + args.w_nll * l_nll + args.w_pinball * l_pin
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            run["mean"] += float(l_mean.detach())
            run["nll"] += float(l_nll.detach())
            run["pin"] += float(l_pin.detach())
            nb += 1
        rec = {"epoch": ep, "train_rel_l2": run["mean"] / max(nb, 1),
               "train_nll": run["nll"] / max(nb, 1),
               "train_pinball": run["pin"] / max(nb, 1),
               "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            val, diag = evaluate(model, va_shards, device, args.alpha)
            rec["cal_rel_l2"] = val
            rec["cal_head"] = diag
            rec["cal_mean"] = sum(val.values()) / len(val)
            # Selection is on the MEAN's error only, exactly as the ensemble
            # members were selected. Selecting on a combined score would make
            # the arms differ in two ways instead of one.
            if rec["cal_mean"] < best:
                best = rec["cal_mean"]
                torch.save({"model": model.state_dict(), "args": ckpt_args,
                            "stats": stats, "tasks": tasks, "epoch": ep,
                            "cal_mean": best, "params": n_par,
                            "params_uq_head": n_par_uq}, out / "best.pt")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)
    logf.close()
    (out / "done.json").write_text(json.dumps(
        {"seed": args.seed, "epochs": args.epochs, "best_cal_mean": best,
         "elapsed_s": time.time() - t0, "params": n_par,
         "params_uq_head": n_par_uq,
         "detach_uq": not args.no_detach_uq}, indent=2))
    print(f"done in {time.time() - t0:.0f}s  best cal mean rel-L2 {best:.5f}",
          flush=True)


if __name__ == "__main__":
    main()
