"""uqkit -- a surrogate that also says when not to trust it.

Three layers, each usable alone:

* `conformal` -- split / group / covariate-shift-weighted calibration, on any
  nonconformity score.
* `ood` -- shift detection and error detection, kept as two separate questions.
* `bench` -- a speedup harness that will not emit a number without the hardware,
  the thread count, and the accuracy it was measured at.

`api.UQSurrogate` ties them together; `sims/` holds the example simulators.
"""
from .api import Simulator, UQSurrogate           # noqa: F401
from .conformal import (GroupConformal, LikelihoodRatioProbe,  # noqa: F401
                        SplitConformal, WeightedConformal, SCORES,
                        conformal_quantile, get_score)
from .ensemble import DeepEnsemble                # noqa: F401
from .features import Mahalanobis, spectral_features   # noqa: F401
from .metrics import auroc, binom_ci, pearson, rel_l2  # noqa: F401
from .ood import error_auroc, residual_score, shift_auroc, spread_score  # noqa: F401

__version__ = "0.1.0"
