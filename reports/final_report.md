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
| rl_constrained | -0.5750 | -0.0405 | 0.0681 | 0.1053 | 0.0221 | 0.0301 | 0.0000 | 0.0008 | 0.9206 |
| rl_unconstrained | -0.5591 | -0.0396 | 0.0682 | 0.1033 | 0.0221 | 0.0302 | 0.0000 | 0.0008 | 0.9225 |

## Statistical comparison (paired block bootstrap, Sharpe)
| comparison | sharpe_diff | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- |
| constrained_minus_rl_unconstrained | -0.0792 | -0.2254 | 0.0663 | 0.2905 |
| constrained_minus_equal_weight | -0.0351 | -0.1168 | 0.0461 | 0.4165 |

## Deterministic baselines (full universe)
|  | sharpe | ann_return | ann_vol | max_drawdown | cvar_95 | cvar_99 | cvar_breach_rate | avg_turnover | final_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | -0.5541 | -0.0387 | 0.0672 | 0.1010 | 0.0217 | 0.0298 | 0.0000 | 0.0000 | 0.9241 |
| inverse_vol | -0.6301 | -0.0367 | 0.0568 | 0.0923 | 0.0181 | 0.0243 | 0.0000 | 0.0144 | 0.9280 |
| min_variance | -0.8579 | -0.0433 | 0.0501 | 0.1116 | 0.0159 | 0.0216 | 0.0000 | 0.0279 | 0.9153 |
| mean_variance | 0.3307 | 0.0285 | 0.0997 | 0.0987 | 0.0235 | 0.0313 | 0.0000 | 0.0908 | 1.0578 |
| risk_parity | -0.7782 | -0.0430 | 0.0546 | 0.1047 | 0.0173 | 0.0224 | 0.0000 | 0.0149 | 0.9159 |
| cvar_optimizer | -0.1126 | -0.0080 | 0.0571 | 0.0746 | 0.0179 | 0.0257 | 0.0000 | 0.0359 | 0.9840 |

## Real-ETF studies

See `reports/etf_study.md` for the full write-up.

### Differentiable allocator — stress study (constrained vs unconstrained)
|  | ann_return | ann_vol | sharpe | sortino | calmar | max_drawdown | hit_rate | cvar_95 | cvar_99 | avg_turnover | total_costs | cvar_breach_rate | constraint_violations | final_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 0.0624 | 0.0719 | 0.8780 | 0.8367 | 0.5523 | 0.1129 | 0.5918 | 0.0219 | 0.0435 | 0.0000 | 0.0000 | 0.2022 | 0.0000 | 1.3644 |
| inverse_vol | 0.0410 | 0.0534 | 0.7796 | 0.6909 | 0.3634 | 0.1129 | 0.6030 | 0.0156 | 0.0435 | 0.0098 | 0.0013 | 0.2022 | 0.0000 | 1.2294 |
| min_variance | 0.0446 | 0.0499 | 0.9013 | 0.7571 | 0.3953 | 0.1129 | 0.6330 | 0.0141 | 0.0435 | 0.0112 | 0.0015 | 0.2022 | 0.0000 | 1.2514 |
| cvar_optimizer | 0.0419 | 0.0519 | 0.8172 | 0.7043 | 0.3713 | 0.1129 | 0.6292 | 0.0149 | 0.0435 | 0.0157 | 0.0021 | 0.2022 | 0.0000 | 1.2348 |
| rl_unconstrained | 0.0653 | 0.1110 | 0.6261 | 0.6072 | 0.3221 | 0.2058 | 0.5610 | 0.0333 | 0.0647 | 0.0233 | 0.0031 | 0.2345 | 7.2000 | 1.3850 |
| rl_cvar_constrained | 0.0565 | 0.0652 | 0.8758 | 0.8729 | 0.4821 | 0.1176 | 0.5753 | 0.0189 | 0.0327 | 0.0387 | 0.0052 | 0.1888 | 0.0000 | 1.3261 |

### Stress study — CVaR-99 constraint bootstrap test
| comparison | diff | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- |
| cvar99_constrained_minus_unconstrained | -0.0330 | -0.0347 | -0.0076 | 0.0000 |

### Walk-forward — concatenated out-of-sample metrics
|  | ann_return | ann_vol | sharpe | sortino | calmar | max_drawdown | hit_rate | cvar_95 | cvar_99 | n_weeks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rl_unconstrained | 0.0434 | 0.0632 | 0.7036 | 0.6127 | 0.2865 | 0.1514 | 0.6000 | 0.0226 | 0.0416 | 520.0000 |
| rl_cvar_constrained | 0.0408 | 0.0496 | 0.8325 | 0.7304 | 0.3590 | 0.1138 | 0.6077 | 0.0168 | 0.0331 | 520.0000 |
| min_variance | 0.0332 | 0.0227 | 1.4470 | 1.3996 | 0.5804 | 0.0572 | 0.6173 | 0.0071 | 0.0106 | 520.0000 |
| inverse_vol | 0.0344 | 0.0325 | 1.0575 | 0.9827 | 0.4899 | 0.0703 | 0.5942 | 0.0104 | 0.0183 | 520.0000 |

### Walk-forward — fold-level CVaR-99 paired test
| n_folds | mean_cvar99_con | mean_cvar99_unc | mean_diff | folds_improved | wilcoxon_p |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.0142 | 0.0194 | -0.0052 | 5.0000 | 0.0625 |

### Robustness — constraint effect under cost/limit/universe perturbations
| scenario | unc_sharpe | con_sharpe | unc_cvar99 | con_cvar99 | cvar99_reduction | unc_maxdd | con_maxdd | maxdd_reduction | constraint_helps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6319 | 0.8648 | 0.0611 | 0.0301 | 0.0310 | 0.2005 | 0.1108 | 0.0897 | True |
| cost_2x | 0.6329 | 0.8836 | 0.0572 | 0.0265 | 0.0307 | 0.1993 | 0.1019 | 0.0974 | True |
| cost_3x | 0.5172 | 0.8447 | 0.0547 | 0.0251 | 0.0296 | 0.2257 | 0.1030 | 0.1227 | True |
| limit_tight | 0.6319 | 1.1310 | 0.0611 | 0.0172 | 0.0438 | 0.2005 | 0.0552 | 0.1453 | True |
| limit_loose | 0.6319 | 0.5756 | 0.0611 | 0.0505 | 0.0106 | 0.2005 | 0.2168 | -0.0163 | True |
| drop_gold | 0.4653 | 0.9218 | 0.0563 | 0.0197 | 0.0367 | 0.2262 | 0.0623 | 0.1639 | True |
| drop_commodity | 0.5754 | 0.8652 | 0.0627 | 0.0340 | 0.0287 | 0.2115 | 0.1203 | 0.0912 | True |

### Ablations — anchor / CVaR level / risk budget / objective
| ablation | anchor | alpha | limit | objective | sharpe | cvar_95 | cvar_99 | max_drawdown | avg_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | inverse_vol | 0.9500 | 0.0120 | return | 0.8648 | 0.0177 | 0.0301 | 0.1108 | 0.0391 |
| no_anchor | nan | 0.9500 | 0.0120 | return | 0.8265 | 0.0180 | 0.0318 | 0.1076 | 0.0017 |
| alpha_0.90 | inverse_vol | 0.9000 | 0.0120 | return | 0.6843 | 0.0249 | 0.0452 | 0.1725 | 0.0363 |
| alpha_0.99 | inverse_vol | 0.9900 | 0.0120 | return | 1.2508 | 0.0098 | 0.0146 | 0.0533 | 0.0337 |
| budget_tight | inverse_vol | 0.9500 | 0.0072 | return | 1.2723 | 0.0092 | 0.0135 | 0.0411 | 0.0296 |
| budget_loose | inverse_vol | 0.9500 | 0.0192 | return | 0.5146 | 0.0302 | 0.0493 | 0.2300 | 0.0207 |
| objective_sharpe | inverse_vol | 0.9500 | 0.0120 | sharpe | 0.7907 | 0.0166 | 0.0273 | 0.1077 | 0.0435 |

### Feature study — market-only vs factor betas
| variant | ann_return | ann_vol | sharpe | sortino | calmar | max_drawdown | hit_rate | cvar_95 | cvar_99 | avg_turnover | total_costs | cvar_breach_rate | constraint_violations | final_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_only | 0.0552 | 0.0607 | 0.9142 | 0.9156 | 0.5186 | 0.1063 | 0.5805 | 0.0175 | 0.0288 | 0.0387 | 0.0052 | 0.0936 | 0.0000 | 1.3176 |
| market+factor_betas | 0.0563 | 0.0646 | 0.8823 | 0.8652 | 0.4888 | 0.1158 | 0.5768 | 0.0190 | 0.0346 | 0.0406 | 0.0054 | 0.1985 | 0.0000 | 1.3247 |

## Notes
- Metrics are averaged across training seeds for the RL variants.
- CVaR breach rate is the fraction of weeks the rolling CVaR estimate exceeds the configured limit.
- See `results/training_curves/` for per-episode Lagrange-multiplier and breach-rate paths.