# ETF Study — CVaR-Constrained RL Allocator vs. Baselines

_Real-data study on a 7-ETF macro universe. Generated from `configs/experiment_etf.yaml`._

## 1. Data

- **Universe (7 ETFs):** SPY (equity), TLT (rates), HYG (credit), DBC (commodity),
  GLD (gold), UUP (US dollar), BIL (1–3m T-bills = cash leg).
- **Source:** Yahoo chart API adjusted close (dividends/splits folded in → total-return proxy),
  fetched by `src/crlpa/data/load_prices.py` and frozen to parquet.
- **Frequency:** weekly (W-FRI) simple returns from resampled adjusted close.
- **Span:** 2008-01-01 → 2024-12-31, **887 weekly observations**.
- **Chronological split (no shuffle):** train 532 (≈2008–2018), validation 177
  (≈2018–2021), test 178 (≈2021–2024). The 2008 GFC and 2020 COVID crashes fall in
  train/validation; the test window is comparatively calm.
- **Leakage controls:** observations at step `t` use only returns `[:t]`
  (`tests/test_no_lookahead.py`); rolling baselines recompute weights from
  history strictly before each decision.

## 2. Method

A weekly MDP: state = per-asset momentum/vol over a 26-week lookback + current
weights + drawdown + rolling CVaR estimate. The actor emits a Gaussian over
pre-softmax logits → softmax → long-only weights, projected onto the admissible
set (max weight 40%, turnover cap 0.5, BIL as cash leg). Costs are 5 bps on
realised turnover. A reward critic estimates value-to-go; a **safety critic**
estimates the discounted CVaR-breach-to-go; a **Lagrange multiplier** penalises
the policy when the breach rate exceeds the budget. The unconstrained baseline is
the identical agent with the multiplier frozen at zero.

CVaR is the 95% weekly expected shortfall; the breach threshold is a 3% weekly
CVaR limit; the tolerated breach budget is 5%.

## 3. Deterministic baselines (test split, 2021–2024)

| Strategy | Sharpe | Ann. return | Max DD | CVaR-95 | Avg turnover |
| --- | --- | --- | --- | --- | --- |
| equal_weight | 0.82 | 0.049 | 0.097 | 0.0163 | 0.000 |
| inverse_vol | 1.45 | — | 0.041 | 0.0087 | 0.014 |
| min_variance | **2.52** | 0.064 | **0.016** | **0.0064** | 0.014 |
| mean_variance | 0.84 | — | 0.104 | 0.0198 | 0.091 |
| risk_parity | 0.70 | — | 0.134 | 0.0182 | 0.015 |
| cvar_optimizer | 2.24 | 0.064 | 0.016 | 0.0080 | 0.017 |

**Read:** minimum-variance and the min-CVaR optimiser dominate. Both effectively
park in BIL/TLT during the 2022 rate-hike drawdown, earning the positive cash rate
at near-zero volatility — a hard bar to clear on this calm, cash-friendly window.

## 4. RL allocators vs. baselines (test split)

Full run: 150 episodes × 5 seeds × 2 variants, metrics averaged across seeds.

| Strategy | Sharpe | Ann. return | Max DD | CVaR-95 | Breach rate | Avg turnover |
| --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 0.82 | 0.049 | 0.097 | 0.0163 | 0.00 | 0.000 |
| min_variance | **2.52** | 0.064 | **0.016** | **0.0064** | 0.00 | 0.014 |
| cvar_optimizer | 2.24 | 0.064 | 0.016 | 0.0080 | 0.00 | 0.017 |
| rl_constrained | 0.79 | 0.048 | 0.098 | 0.0166 | 0.00 | 0.001 |
| rl_unconstrained | 0.79 | 0.049 | 0.099 | 0.0167 | 0.00 | 0.001 |

Paired block bootstrap (Sharpe, 2000 resamples, block 4):

| Comparison | Sharpe diff | 95% CI | p-value |
| --- | --- | --- | --- |
| constrained − unconstrained | −0.033 | [−0.091, 0.024] | 0.26 |
| constrained − equal_weight | +0.031 | [−0.030, 0.090] | 0.31 |

Neither difference is significant. Both RL variants land near equal weight and well
below the variance/CVaR optimisers.

## 5. The constraint-coupling finding

The first full run produced **constrained ≈ unconstrained** (identical CVaR and
breach rates). Investigation traced this to a scale mismatch in the policy
objective rather than a modelling choice:

1. The reward advantage is standardised to unit scale, but the **cost advantage
   was left on its raw scale** (the per-step CVaR excess is ~1e-3). So
   `λ · cost_advantage` was negligible for any sane multiplier — the constraint
   could never bind.
2. The Lagrange multiplier was driven by the **raw cost magnitude** (~1e-3), which
   is far too small to move the dual; λ stayed ≈0.01 even at large learning rates.

Two fixes (kept as the defaults):
- **Standardise the cost advantage** to the same unit scale as the reward
  advantage (`models/cvar_actor_critic.py`).
- **Drive the dual with the breach indicator (0/1) against a breach-rate budget**
  (`cost_mode: breach`), which is well scaled. With this, λ reaches O(1) and the
  constraint demonstrably binds (λ ≈ 0.7 in a tightened-limit probe).

## 6. Findings

1. **Deterministic optimisers dominate this universe/window.** Minimum-variance
   (Sharpe 2.52, CVaR-95 0.0064) and the min-CVaR LP (2.24) far exceed both RL
   variants and equal weight, by parking in the cash/rates leg through the
   2022 drawdown.
2. **The learned RL allocator is essentially static near-equal-weight**
   (Sharpe ≈ 0.79 for both variants vs 0.82 equal weight). The compact on-policy
   A2C does not learn the state-dependent "rotate into cash under stress" policy.
3. **The CVaR limit is non-binding out-of-sample** (0 breaches for every strategy
   on the calm 2021–2024 test split), so constrained and unconstrained agents are
   statistically indistinguishable (Sharpe diff −0.03, p=0.26) — the constraint is
   a near-free insurance premium here.
4. **The constraint machinery is correct and binds where stress lives.** After the
   coupling fix, the multiplier reaches λ ≈ 7–8 during the crisis-heavy training
   span. Yet the *deployed greedy* policy is unchanged versus unconstrained
   (in-sample full history: CVaR-95 0.0200 vs 0.0201, breach 0.117 vs 0.117) — the
   actor converges to the same near-uniform optimum regardless of dual pressure.
   So the bottleneck is the **policy class / training budget**, not the constraint
   layer.

## 7. Limitations & next steps

- The compact on-policy A2C, trained on a single deterministic price path for
  150 episodes, converges close to a static near-equal-weight allocation; it does
  not learn the "rotate into cash during stress" behaviour that the variance/CVaR
  optimisers exploit. The CVaR constraint therefore acts mainly as an insurance
  premium on this calm out-of-sample window rather than producing a tail-risk
  reduction, because tail events sit in the training span.
- **Next steps:** (i) episode start-point randomisation and return scaling for a
  stronger learning signal; (ii) longer training / larger networks; (iii) evaluate
  on a stress-inclusive walk-forward so the constraint's protective value is
  measured where breaches actually occur; (iv) tighten the CVaR limit so it binds
  out-of-sample; (v) PPO/SAC actor for sample efficiency.
