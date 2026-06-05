from __future__ import annotations

import numpy as np
import pandas as pd


def cvar(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        raise ValueError("returns cannot be empty")
    cutoff = np.quantile(values, 1 - alpha)
    return float(-values[values <= cutoff].mean())


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1 + values)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1
    return float(-drawdown.min())


def sharpe(returns: pd.Series | np.ndarray, periods_per_year: int = 52) -> float:
    values = np.asarray(returns, dtype=float)
    std = values.std(ddof=1)
    if std == 0:
        return 0.0
    return float(values.mean() / std * np.sqrt(periods_per_year))

