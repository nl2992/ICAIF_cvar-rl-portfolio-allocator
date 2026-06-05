from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AllocationState:
    step: int
    weights: np.ndarray
    wealth: float


class AllocationEnv:
    """Small gym-like long-only allocation environment."""

    def __init__(self, returns: pd.DataFrame, transaction_cost_bps: float = 2.0) -> None:
        if returns.empty:
            raise ValueError("returns cannot be empty")
        self.returns = returns.reset_index(drop=True)
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.n_assets = returns.shape[1]
        self.state = AllocationState(0, np.full(self.n_assets, 1 / self.n_assets), 1.0)

    def reset(self) -> AllocationState:
        self.state = AllocationState(0, np.full(self.n_assets, 1 / self.n_assets), 1.0)
        return self.state

    def step(self, action: np.ndarray) -> tuple[AllocationState, float, bool, dict[str, float]]:
        weights = self._normalise(action)
        old = self.state.weights
        costs = float(np.abs(weights - old).sum() * self.transaction_cost)
        asset_return = float(weights @ self.returns.iloc[self.state.step].to_numpy())
        portfolio_return = asset_return - costs
        wealth = self.state.wealth * (1 + portfolio_return)
        done = self.state.step >= len(self.returns) - 1
        self.state = AllocationState(self.state.step + 1, weights, wealth)
        return self.state, portfolio_return, done, {"costs": costs, "wealth": wealth}

    def _normalise(self, action: np.ndarray) -> np.ndarray:
        weights = np.asarray(action, dtype=float)
        if weights.shape != (self.n_assets,):
            raise ValueError(f"expected action shape {(self.n_assets,)}")
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0:
            return np.full(self.n_assets, 1 / self.n_assets)
        return weights / total

