from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1 / n_assets)


def inverse_volatility(returns: pd.DataFrame, lookback: int = 52) -> np.ndarray:
    window = returns.tail(lookback)
    vols = window.std().replace(0, np.nan).fillna(window.std().mean())
    inv = 1 / vols.to_numpy()
    return inv / inv.sum()

