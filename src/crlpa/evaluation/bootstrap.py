from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from crlpa.evaluation.metrics import sharpe


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    p_value: float

    def significant(self, level: float = 0.05) -> bool:
        return self.p_value < level


def paired_bootstrap(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    statistic: Callable[[np.ndarray], float] = sharpe,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    block_size: int = 1,
    seed: int = 42,
) -> BootstrapResult:
    """Paired (optionally block) bootstrap of ``statistic(a) - statistic(b)``.

    Resampling indices are shared between the two series so the comparison stays
    paired. ``block_size > 1`` uses a circular block bootstrap to preserve serial
    dependence. The two-sided p-value tests the null that the difference is zero.
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires equal-length series")
    n = a.size
    rng = np.random.default_rng(seed)
    observed = statistic(a) - statistic(b)

    diffs = np.empty(n_resamples)
    n_blocks = int(np.ceil(n / block_size))
    for i in range(n_resamples):
        if block_size <= 1:
            idx = rng.integers(0, n, size=n)
        else:
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([(s + np.arange(block_size)) % n for s in starts])[:n]
        diffs[i] = statistic(a[idx]) - statistic(b[idx])

    tail = (1 - confidence) / 2
    ci_low, ci_high = np.quantile(diffs, [tail, 1 - tail])
    centred = diffs - diffs.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    return BootstrapResult(float(observed), float(ci_low), float(ci_high), p_value)
