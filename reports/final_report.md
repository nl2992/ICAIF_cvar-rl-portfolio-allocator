# Final Report — cvar_ac_v1
_Generated 2026-06-06._

## Research question
> Can a constrained actor-critic allocator reduce tail risk and constraint breaches versus unconstrained RL while remaining competitive with standard portfolio optimisers after costs?

## Allocator vs. baselines (test split)
|  | sharpe | ann_return | ann_vol | max_drawdown | cvar_95 | cvar_99 | cvar_breach_rate | avg_turnover | final_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | -0.5541 | -0.0387 | 0.0672 | 0.1010 | 0.0217 | 0.0298 | 0.0000 | 0.0000 | 0.9241 |
| min_variance | -0.8579 | -0.0433 | 0.0501 | 0.1116 | 0.0159 | 0.0216 | 0.0000 | 0.0279 | 0.9153 |
| cvar_optimizer | -0.1126 | -0.0080 | 0.0571 | 0.0746 | 0.0179 | 0.0257 | 0.0000 | 0.0359 | 0.9840 |
| rl_constrained | -0.5160 | -0.0361 | 0.0670 | 0.0965 | 0.0215 | 0.0292 | 0.0000 | 0.0010 | 0.9291 |
| rl_unconstrained | -0.4541 | -0.0330 | 0.0687 | 0.0911 | 0.0218 | 0.0301 | 0.0000 | 0.0010 | 0.9350 |

## Statistical comparison (paired block bootstrap, Sharpe)
| comparison | sharpe_diff | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- |
| constrained_minus_rl_unconstrained | -0.1239 | -0.2806 | 0.0357 | 0.1280 |
| constrained_minus_equal_weight | -0.0542 | -0.1498 | 0.0395 | 0.2620 |

## Deterministic baselines (full universe)
|  | sharpe | ann_return | ann_vol | max_drawdown | cvar_95 | cvar_99 | cvar_breach_rate | avg_turnover | final_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | -0.5541 | -0.0387 | 0.0672 | 0.1010 | 0.0217 | 0.0298 | 0.0000 | 0.0000 | 0.9241 |
| inverse_vol | -0.6301 | -0.0367 | 0.0568 | 0.0923 | 0.0181 | 0.0243 | 0.0000 | 0.0144 | 0.9280 |
| min_variance | -0.8579 | -0.0433 | 0.0501 | 0.1116 | 0.0159 | 0.0216 | 0.0000 | 0.0279 | 0.9153 |
| mean_variance | 0.3307 | 0.0285 | 0.0997 | 0.0987 | 0.0235 | 0.0313 | 0.0000 | 0.0908 | 1.0578 |
| risk_parity | -0.7782 | -0.0430 | 0.0546 | 0.1047 | 0.0173 | 0.0224 | 0.0000 | 0.0149 | 0.9159 |
| cvar_optimizer | -0.1126 | -0.0080 | 0.0571 | 0.0746 | 0.0179 | 0.0257 | 0.0000 | 0.0359 | 0.9840 |

## Notes
- Metrics are averaged across training seeds for the RL variants.
- CVaR breach rate is the fraction of weeks the rolling CVaR estimate exceeds the configured limit.
- See `results/training_curves/` for per-episode Lagrange-multiplier and breach-rate paths.