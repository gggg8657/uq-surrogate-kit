"""The H15 development shift suite: shards for *fitting* a difficulty model.

A width predictor has to see shifted data or it cannot know anything about
shift. Fitting it on the 32 evaluation shards would make the resulting coverage
worthless, so this module defines a disjoint suite and the discipline around it:

* **new seed block.** Evaluation shards use `20000 + 37*i`; these use
  `40000 + 91*i`. No sample is shared with any split of any evaluation shard.
* **values the evaluation shards do not use.** The roughness axis is specified
  as a *delta* from each family's base alpha, chosen to miss both the
  `rough`/`smooth` absolutes (1.5, 4.0) and every rung of the graded ladder
  (`GRADED_DALPHA`); `tau` misses 16.0 and `amp` misses 2.0. `assert_disjoint()`
  checks this rather than trusting the comment, and `tests/test_devshift.py`
  runs it.
* **nothing is written to disk.** These shards are generated in memory,
  reduced immediately to (features, score) rows, and dropped. Reproducibility
  is by seed, and the evaluation shards on disk cannot be accidentally
  overwritten by a dev run.

`MECHANISM` groups the specs by the knob they turn, which is what makes the
leave-one-mechanism-out reading of H15 expressible. The graded `dam` ladder is
the *same axis* as `rough`/`smooth`, so it belongs to `alpha`: holding out
`alpha` has to hold out the ladder too, or the headline reading leaks.
"""
from __future__ import annotations

import torch

from .sims import pde2d as P

#: deltas on each family's base alpha. Chosen to avoid the absolute values
#: 1.5 and 4.0 (the `rough`/`smooth` shards) and every graded rung, while
#: bracketing them so the "unseen strength" reading is interpolation.
DEV_DALPHA = (-1.4, -0.85, -0.4, -0.15, 0.4, 1.1, 2.2)
#: absolute correlation lengths; the evaluation shards use 16.0
DEV_TAU = (4.0, 9.0, 12.0, 21.0)
#: input amplitude multipliers; the evaluation shards use 2.0
DEV_AMP = (1.3, 1.6, 2.6, 3.4)

DEV_PARENTS = ("poisson", "helmholtz", "diffusion", "advdiff", "darcy")
SEED_BASE, SEED_STRIDE = 40000, 91


def _tag(v):
    return f"{v:g}".replace(".", "p").replace("-", "m")


def specs():
    """[(task, parent, mechanism, cfg_override)] for the whole dev suite."""
    out = []
    for p in DEV_PARENTS:
        base = P.CFG[p]
        for d in DEV_DALPHA:
            out.append((f"dev_{p}_da{_tag(d)}", p, "alpha",
                        dict(alpha=base["alpha"] + d)))
        for t in DEV_TAU:
            out.append((f"dev_{p}_tau{_tag(t)}", p, "tau", dict(tau=t)))
        for a in DEV_AMP:
            out.append((f"dev_{p}_amp{_tag(a)}", p, "amp", dict(amp=a)))
    return out


#: mechanism -> the evaluation shard suffixes it would leak information about.
#: `dam*` sits under `alpha` because the graded ladder turns the same knob.
MECHANISM_COVERS = {
    "alpha": ("rough", "smooth", "dam"),
    "tau": ("tau",),
    "amp": ("amp2",),
}
MECHANISMS = tuple(MECHANISM_COVERS)


def eval_mechanism(task):
    """Which dev mechanism a given evaluation shard's shift belongs to."""
    for mech, suffixes in MECHANISM_COVERS.items():
        for s in suffixes:
            if task.endswith(f"_{s}") or f"_{s}" in task:
                return mech
    return None


def register():
    """Add the dev tasks to `pde2d`'s tables. Idempotent."""
    for task, parent, _mech, over in specs():
        P.CFG[task] = P.CFG[parent] | over
        P.FAMILY_BASE[task] = P.FAMILY_BASE[parent]
        P.PARENT[task] = parent
    return specs()


def assert_disjoint():
    """Fail loudly if a dev value coincides with an evaluation value.

    The comment above says the suites do not overlap. This checks it, because a
    single coinciding alpha would make the leave-one-mechanism-out reading a
    leak that no downstream number would reveal.
    """
    bad = []
    for p in DEV_PARENTS:
        base = P.CFG[p]["alpha"]
        dev = {round(base + d, 10) for d in DEV_DALPHA}
        # `rough`/`smooth` are absolute; the ladder is relative to the base
        ev = {1.5, 4.0} | {round(base + d, 10) for d in P.GRADED_DALPHA}
        if dev & ev:
            bad.append(f"{p}: alpha {sorted(dev & ev)} used by both suites")
    if 16.0 in DEV_TAU:
        bad.append("tau 16.0 is an evaluation value")
    if 2.0 in DEV_AMP:
        bad.append("amp 2.0 is an evaluation value")
    if bad:
        raise AssertionError("dev suite overlaps the evaluation suite: "
                             + "; ".join(bad))
    return True


@torch.no_grad()
def generate(task, n, N=64, device="cuda", index=0):
    """One dev shard, in memory. Seeded out of the dev block."""
    return P.generate(task, n, N=N, seed=SEED_BASE + SEED_STRIDE * index,
                      device=device)
