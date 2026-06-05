from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_returns(
    n_steps: int = 260,
    assets: tuple[str, ...] = ("equity_index", "rates", "credit", "commodity", "fx"),
    seed: int = 11,
) -> pd.DataFrame:
    """Create correlated weekly returns with one stress regime."""
    rng = np.random.default_rng(seed)
    n_assets = len(assets)
    base_corr = np.full((n_assets, n_assets), 0.25)
    np.fill_diagonal(base_corr, 1.0)
    vol = np.linspace(0.012, 0.025, n_assets)
    cov = np.outer(vol, vol) * base_corr
    returns = rng.multivariate_normal(np.linspace(0.0004, 0.0012, n_assets), cov, n_steps)
    returns[110:125] += np.linspace(-0.035, 0.005, n_assets)
    return pd.DataFrame(returns, columns=assets)

