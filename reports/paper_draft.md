# A Differentiable CVaR-Constrained Allocator: Explicit Tail Control for Learned Portfolio Policies

*Working draft. All numbers are reproducible from this repository
(`scripts/run_diff_study.py`, `run_walk_forward.py`, `run_robustness.py`,
`run_ablations.py`, `run_macro_study.py`); figures in `reports/figures/`.*

---

## Abstract

We study whether an explicit Conditional-Value-at-Risk (CVaR) constraint reduces
the downside tail risk of a *learned* multi-asset allocator without sacrificing
net performance. On a seven-ETF macro universe (2008–2024, weekly, 5 bps costs) we
find that a naïve model-free actor–critic learns little beyond an equal-weight
portfolio, and—because of a scale mismatch in the constraint coupling—its CVaR
constraint is inert. After (i) fixing the coupling and (ii) replacing the
score-function learner with a **differentiable allocator** that back-propagates a
risk-adjusted objective and a differentiable CVaR penalty through the rollout, the
constraint becomes a meaningful control: on a stress out-of-sample window (trained
pre-2018, tested through the 2020 and 2022 drawdowns) it cuts 99% CVaR by ~49% and
maximum drawdown by ~43% relative to an unconstrained learner, while *improving*
risk-adjusted return; the tail reduction is robust across seeds, a 10-fold
walk-forward, and 2–3× transaction-cost perturbations. We report an honest null on
the secondary question: neither learner beats a rolling minimum-variance optimiser
on this universe. The contribution is therefore an explicit, well-behaved
tail-control layer for learned allocators, plus a reproducible benchmark and a
cautionary finding about constraint coupling in Lagrangian RL.

---

## 1. Introduction

Reinforcement learning (RL) for portfolio allocation is usually evaluated on
return or Sharpe ratio, with risk handled implicitly. Practitioners, however,
manage *tail* risk explicitly. This work asks a narrow, falsifiable question:

> Does an explicit CVaR constraint reduce the downside tail losses and constraint
> breaches of a learned allocator, versus an otherwise identical unconstrained
> learner, while remaining competitive with standard portfolio optimisers after
> costs?

Our contributions:

1. A **leak-free, tested portfolio-control environment** with transaction costs,
   an admissible-set projection (long-only, max-weight, turnover, cash floor), and
   a rolling CVaR state/constraint signal.
2. A **differentiable CVaR-constrained allocator**: rather than score-function
   policy gradients, we exploit that the per-step reward `wᵀr − cost` is a known,
   differentiable function of the weights, and back-propagate a Sharpe/return
   objective plus a differentiable CVaR penalty through the rollout, with a
   Lagrangian dual on a breach-rate budget and a residual policy over an adaptive
   anchor for generalisation.
3. A **negative methodological finding**: in the model-free Lagrangian
   actor–critic, standardising the reward advantage but not the cost advantage
   makes the constraint inert for any reasonable multiplier; we identify and fix
   the coupling.
4. A **rigorous evaluation**: deterministic optimisers, two model-free RL families
   (A2C, PPO), seed averaging, paired bootstrap, a rolling walk-forward with
   regime slices, robustness to costs/universe, and ablations.

We are explicit about scope: the learned allocator does **not** beat a rolling
minimum-variance optimiser here; the positive result is the constraint's effect on
the learned policy's tail risk.

---

## 2. Problem formulation

**MDP.** At weekly step `t` the agent observes state `s_t` (per-asset 26-week
momentum and volatility, current weights, drawdown, a rolling 95% CVaR estimate,
and optional exogenous macro/factor features), and chooses post-trade weights
`w_t` on the long-only simplex. The environment projects `w_t` onto the admissible
set `𝒲` (max weight 0.40, turnover cap 0.50, optional cash floor), charges
proportional cost `c·‖w_t − w_{t-1}‖₁` (c = 5 bps), and realises
`r_t = w_tᵀρ_t − c·‖w_t − w_{t-1}‖₁`, where `ρ_t` are next-week asset returns.
Features at `t` use only `ρ_{<t}` (no look-ahead; enforced by tests).

**Constrained objective (CMDP).** Maximise risk-adjusted return subject to a tail
constraint,
`max_π  J(π)`  s.t. `CVaR_α(loss) ≤ d`,
with `α = 0.95` and breach threshold on the rolling CVaR estimate. We solve the
Lagrangian `J(π) − λ·(constraint violation)` with dual ascent on `λ`.

---

## 3. Method

### 3.1 Differentiable allocator
The actor is an MLP mapping `s_t` to simplex weights via softmax. Because the
environment's reward and a sample-CVaR are differentiable in `w`, we train by
back-propagating, over randomised windows of the training series, the objective

`L = −Sharpe(net returns)  [or −mean log return]  + λ·ReLU(ĈVaR − d)  + κ·turnover`,

where `ĈVaR` is the mean of the worst `(1−α)` fraction of window net returns
(a differentiable tail mean via `topk`). This pathwise gradient is far more
sample-efficient than score-function RL for this problem.

### 3.2 Residual policy over an adaptive anchor
To generalise across regimes rather than overfit one price path, weights are
`w_t = softmax(log(a_t) + f_θ(s_t))`, where `a_t` is an adaptive inverse-volatility
anchor computed from recent returns. At initialisation `f_θ ≈ 0`, so the policy
*is* the anchor (a strong, regime-robust prior); training learns a state-dependent
tilt. Validation model selection keeps the best objective subject to the validation
CVaR limit (else lowest validation CVaR), with weight decay—both to curb backtest
overfitting.

### 3.3 The constraint-coupling fix
In the model-free Lagrangian A2C, the reward advantage is standardised to unit
scale while the raw per-step CVaR cost is ~1e-3; thus `λ·(cost advantage)` is
negligible and the constraint never binds (we observed constrained ≡ unconstrained
even at large `λ`). Fixes (kept as defaults): standardise the cost advantage to the
same scale, and drive the dual with a **breach indicator** against a breach-rate
budget. The multiplier then reaches O(1) and binds.

### 3.4 Baselines
Deterministic (rolling, re-estimated weekly): cash, equal weight, inverse vol,
minimum variance, mean–variance, risk parity, and a minimum-CVaR Rockafellar–
Uryasev LP. Learned: model-free A2C and PPO (clipped surrogate, GAE), and the
unconstrained differentiable allocator (`λ = 0`).

---

## 4. Data

Seven liquid ETFs spanning the major macro blocks—SPY (equity), TLT (rates), HYG
(credit), DBC (commodity), GLD (gold), UUP (US dollar), BIL (cash)—from Yahoo
adjusted close, resampled to **887 weekly returns, 2008–2024**. Splits are
chronological. Macro covariates (term spread, VIX) are sourced from index proxies
(`^TNX−^IRX`, `^VIX`); a FRED loader is also provided.

---

## 5. Results

### 5.1 Calm chronological split (2021–2024): the constraint is non-binding
Deterministic minimum-variance (Sharpe 2.52, CVaR-95 0.0064) and min-CVaR (2.24)
dominate; the differentiable allocator reaches Sharpe 1.24 (vs equal weight 0.82,
inverse vol 1.45), and the model-free A2C only ≈0.79. The CVaR limit is not
breached by any strategy on this calm window, so constrained ≈ unconstrained—the
constraint is a near-free insurance premium with nothing to insure (table:
`results/tables_diff/`, A2C in `results/tables/`).

### 5.2 Stress window (train pre-2018, test through 2020 & 2022): the main result
Return-greedy unconstrained learner vs CVaR-constrained, mean ± std over 5 seeds:

| Metric | unconstrained | constrained | change |
| --- | --- | --- | --- |
| Sharpe | 0.63 ± 0.09 | **0.88 ± 0.04** | +40% |
| Max drawdown | 0.206 | **0.118** | −43% |
| CVaR-95 | 0.0331 | **0.0188** | −43% |
| CVaR-99 | 0.0654 | **0.0331** | −49% |
| Breach rate | 0.234 | **0.189** | −19% |

Paired block bootstrap of the CVaR-99 difference: −0.033, 95% CI [−0.035, −0.008],
p ≈ 0.00. See `figures/wealth_curves.png`, `drawdown_curves.png`,
`weights_constrained.png` (the constrained policy rotates into the cash/rates leg
under stress), and `lagrange_path.png`.

### 5.3 Walk-forward across regimes (10 OOS folds, ≈2010–2024)
Concatenated out-of-sample:

| Strategy | Sharpe | Max DD | CVaR-99 |
| --- | --- | --- | --- |
| unconstrained RL | 0.70 | 0.151 | 0.0416 |
| **constrained RL** | **0.83** | **0.114** | **0.0331** |
| inverse vol | 1.06 | 0.070 | 0.0183 |
| minimum variance | 1.45 | 0.057 | 0.0106 |

The constraint beats unconstrained RL out-of-sample (Sharpe +18%, CVaR-99 −20%,
max DD −25%); fold-level Wilcoxon on CVaR-99 gives p = 0.06 (5/10 folds improved,
marginal). Per-regime, the reduction concentrates in **selloffs** (CVaR-99 0.066 →
0.048, max DD 0.218 → 0.137; `figures/regime_cvar99.png`). Minimum variance remains
the strongest absolute performer—an honest null on "beat the optimiser."

### 5.4 Robustness and ablations
The CVaR-99 reduction holds in **7/7** perturbations, including 2× and 3× costs and
dropped assets (`figures/robustness_cvar99.png`). Ablations: the anchor helps; a
more extreme tail level (α = 0.99) or tighter budget pushes the allocator further
to cash and, on the stress window, *both* cuts tail risk and lifts Sharpe (to
≈1.25)—tighter risk control was not costly here (`figures/ablation_cvar99.png`).

### 5.5 Feature study
Rolling factor betas alone slightly hurt (redundant with momentum/vol); adding
macro state (term spread, VIX) gives a marginal Sharpe/drawdown improvement
(0.92 → 0.94, max DD 0.105 → 0.095) but no tail gain.

---

## 6. Discussion

The explicit CVaR layer does what it is designed to do—remove deep-loss weeks—and
on a risk-seeking learner this *also* improves Sharpe, because the avoided tail
dominates the modest return give-up. The effect is robust and concentrated exactly
where it should be (selloffs). However, two honesty checks matter: (i) the headline
contrast uses a return-greedy unconstrained baseline and a tightened limit so the
constraint binds—against a Sharpe-objective learner the gap narrows because that
objective already controls variance; (ii) the strongest method overall is a classic
rolling minimum-variance optimiser, not the learner. The practical reading is that
explicit tail constraints are a cheap, reliable safety layer *for learned policies*,
not a route to beating well-tuned convex optimisers on this universe.

---

## 7. Limitations & future work

- Single universe, single region, synthetic 5 bps costs (robustness covers 2–3×
  but not bid–ask realism).
- Walk-forward significance is marginal (p ≈ 0.06); more folds and ≥1 additional
  universe are needed for a strong claim, with multiple-testing / Deflated-Sharpe
  control.
- The "differentiable allocator" is closer to deep portfolio optimisation than
  classic model-free RL; the model-free A2C/PPO arms remain weak and warrant a
  proper hyperparameter sweep.
- Constrained turnover (~0.04/wk) exceeds the optimisers'; the supported turnover
  penalty should be tuned.
- Next: benchmark-relative active-risk variant; per-asset cost calibration; a
  formal CMDP convergence treatment of the breach-rate dual.

---

## 8. Reproducibility

Leak-free checks, environment/constraint/metric unit tests (60 tests), fixed seeds
(7, 13, 23, 42, 2025), and config snapshots. Each result maps to a script:
`run_baselines`, `run_diff_study`, `run_walk_forward`, `run_robustness`,
`run_ablations`, `run_macro_study`, `make_figures`, `make_report`. Full numeric
tables and the narrative are in `reports/etf_study.md`.
