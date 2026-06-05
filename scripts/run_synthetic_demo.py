from __future__ import annotations

from crlpa.data.synthetic import make_synthetic_returns
from crlpa.envs.allocation import AllocationEnv
from crlpa.evaluation.metrics import cvar, max_drawdown, sharpe
from crlpa.policies.baselines import inverse_volatility
from crlpa.training.rollout import run_static_policy


def main() -> None:
    returns = make_synthetic_returns()
    env = AllocationEnv(returns)
    weights = inverse_volatility(returns.iloc[:52])
    rewards = run_static_policy(env, weights)
    print(
        {
            "steps": len(rewards),
            "sharpe": round(sharpe(rewards), 4),
            "cvar_95": round(cvar(rewards), 4),
            "max_drawdown": round(max_drawdown(rewards), 4),
        }
    )


if __name__ == "__main__":
    main()

