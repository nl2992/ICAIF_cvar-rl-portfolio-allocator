# CVaR-Constrained RL Portfolio Allocator

Research stack for benchmark-aware portfolio allocation with explicit drawdown and CVaR constraints.

## Objective

Design an RL allocator that optimises return while controlling downside risk. The starting point is a compact, testable environment for multi-asset allocation, with policy and evaluation interfaces that can later be swapped for SAC/PPO or a continuous-time actor-critic implementation.

## Initial scope

- Portfolio dataset contract for returns, factor exposures, costs, benchmark weights, and optional liabilities.
- Gym-like allocation environment with budget normalisation and transaction costs.
- Baseline policies for equal-weight and inverse-volatility allocation.
- Risk metrics: annualised return, Sharpe, Sortino, max drawdown, CVaR, turnover, and constraint violations.
- Synthetic demo for weekly rebalancing over a liquid futures-style universe.

## Repo layout

```text
src/crlpa/
  data/          synthetic returns and future data contracts
  envs/          allocation environment
  policies/      baseline policies
  training/      hooks for RL training loops
  evaluation/    portfolio and risk metrics
configs/         experiment definitions
docs/            research notes and data checklist
scripts/         runnable entry points
tests/           smoke and unit tests
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_synthetic_demo.py
```

## Next decisions

- Universe: futures, ETFs, equities, crypto, or mixed multi-asset basket.
- Rebalance cadence and leverage/shorting constraints.
- Risk target: absolute CVaR, benchmark-relative CVaR, max drawdown, or surplus shortfall.
- Whether the first trained agent is discrete-time SAC/PPO or a continuous-time actor-critic approximation.

