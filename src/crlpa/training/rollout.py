from __future__ import annotations

import numpy as np
import pandas as pd

from crlpa.envs.allocation import AllocationEnv


def run_static_policy(env: AllocationEnv, weights: np.ndarray) -> pd.Series:
    env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        _, reward, done, _ = env.step(weights)
        rewards.append(reward)
    return pd.Series(rewards, name="portfolio_return")

