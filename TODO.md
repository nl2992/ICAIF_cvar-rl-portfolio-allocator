# Repo TODO — CVaR-Constrained RL Portfolio Allocator

## Project objective

Build a dynamic portfolio allocation agent that directly controls downside tail risk through a CVaR constraint. The agent should allocate across a liquid multi-asset universe, account for transaction costs, enforce portfolio constraints, and compare against deterministic and unconstrained RL baselines.

The final system should answer:

> Can a constrained actor-critic allocator reduce tail risk and constraint breaches versus unconstrained RL while remaining competitive with standard portfolio optimisers after costs?

---

## Target repo name

`cvar-rl-portfolio-allocator`

---

## Expected repo structure

```text
cvar-rl-portfolio-allocator/
├── README.md
├── TODO.md
├── configs/
│   ├── data.yaml
│   ├── universe.yaml
│   ├── env.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── backtest.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_baseline_portfolios.ipynb
│   ├── 02_env_sanity_checks.ipynb
│   ├── 03_train_allocator.ipynb
│   └── 04_results_and_figures.ipynb
├── src/
│   ├── data/
│   │   ├── load_prices.py
│   │   ├── load_factors.py
│   │   ├── load_macro.py
│   │   └── build_dataset.py
│   ├── features/
│   │   ├── returns.py
│   │   ├── risk_features.py
│   │   ├── factor_features.py
│   │   ├── macro_features.py
│   │   └── cost_features.py
│   ├── envs/
│   │   ├── portfolio_env.py
│   │   └── constraints.py
│   ├── models/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── safety_critic.py
│   │   └── cvar_actor_critic.py
│   ├── baselines/
│   │   ├── equal_weight.py
│   │   ├── inverse_vol.py
│   │   ├── risk_parity.py
│   │   ├── mean_variance.py
│   │   ├── min_variance.py
│   │   └── cvar_optimizer.py
│   ├── training/
│   │   ├── train_allocator.py
│   │   └── lagrangian.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── backtest.py
│   │   ├── walk_forward.py
│   │   ├── bootstrap.py
│   │   └── plots.py
│   └── utils/
│       ├── dates.py
│       ├── splits.py
│       └── io.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
├── scripts/
│   ├── build_dataset.py
│   ├── run_baselines.py
│   ├── train_allocator.py
│   ├── evaluate_allocator.py
│   └── make_report.py
└── tests/
    ├── test_no_lookahead.py
    ├── test_env.py
    ├── test_constraints.py
    ├── test_metrics.py
    └── test_baselines.py
```

---

## Phase 0 — Project setup

- [ ] Create GitHub repo: `cvar-rl-portfolio-allocator`.
- [ ] Add Python environment file: `environment.yml` or `pyproject.toml`.
- [ ] Add `.gitignore` for data, checkpoints, logs, and generated reports.
- [ ] Add `README.md`.
- [ ] Add this `TODO.md`.
- [ ] Add YAML config system.
- [ ] Add experiment tracking convention:
  - [ ] `experiment_id`
  - [ ] `universe_id`
  - [ ] `dataset_version`
  - [ ] `seed`
  - [ ] `split_id`
- [ ] Add deterministic seeds: `7`, `13`, `23`, `42`, `2025`.
- [ ] Add `pytest` skeleton.

**Done when:** `pytest` runs and placeholder tests pass.

---

## Phase 1 — Define research scope

- [ ] Choose first universe:
  - [ ] ETF proxy version: SPY, TLT, GLD, DBC, HYG, UUP, cash.
  - [ ] Alternative: liquid futures universe.
- [ ] Choose rebalance frequency:
  - [ ] weekly for version 1.
  - [ ] daily later.
- [ ] Choose portfolio type:
  - [ ] liability-free first.
  - [ ] benchmark-relative second.
  - [ ] liability-aware extension later.
- [ ] Choose constraints:
  - [ ] long-only.
  - [ ] max individual weight.
  - [ ] cash floor.
  - [ ] max turnover.
  - [ ] gross exposure cap.
- [ ] Choose primary risk constraint:
  - [ ] rolling historical CVaR limit.
  - [ ] model-estimated CVaR limit.
- [ ] Choose success claim:
  - [ ] reduce CVaR and drawdown versus unconstrained RL.
  - [ ] stay competitive with deterministic optimisers after costs.

**Deliverable:** `reports/project_scope.md`.

---

## Phase 2 — Raw data acquisition

Required raw datasets:

- [ ] Asset prices:
  - [ ] adjusted close
  - [ ] total return
  - [ ] volume
  - [ ] dividends/distributions if ETF data
- [ ] Cash/risk-free rate.
- [ ] Factor data:
  - [ ] market factor
  - [ ] size/value/profitability/investment or internal equivalents
  - [ ] momentum factor if available
- [ ] Macro data:
  - [ ] term spread
  - [ ] credit spread
  - [ ] inflation proxy
  - [ ] volatility index
  - [ ] policy rate
  - [ ] dollar index proxy if relevant
- [ ] Benchmark data if benchmark-relative:
  - [ ] benchmark returns
  - [ ] benchmark weights if available
- [ ] Transaction cost assumptions:
  - [ ] spread proxy
  - [ ] commission
  - [ ] slippage
  - [ ] borrow/funding cost if shorting later

Implementation tasks:

- [ ] Implement `src/data/load_prices.py`.
- [ ] Implement `src/data/load_factors.py`.
- [ ] Implement `src/data/load_macro.py`.
- [ ] Store raw data in `/data/raw`.

**Deliverable:** raw datasets with schema report.

---

## Phase 3 — Data cleaning and alignment

- [ ] Convert all assets to total-return series.
- [ ] Align all prices to the same trading calendar.
- [ ] Remove or flag assets with insufficient history.
- [ ] Handle missing data:
  - [ ] forward-fill only when economically valid.
  - [ ] otherwise mark missing and exclude.
- [ ] Lag macro data by release date.
- [ ] Use vintage macro data if available.
- [ ] Resample to weekly decision dates.
- [ ] Compute next-period returns for environment stepping.
- [ ] Validate no look-ahead:
  - [ ] features at time `t` cannot use returns after `t`.
  - [ ] macro values must be available by `t`.
  - [ ] benchmark weights must be known by `t`.

Implementation tasks:

- [ ] Implement `src/data/build_dataset.py`.
- [ ] Implement `tests/test_no_lookahead.py`.

**Deliverable:** `/data/processed/aligned_portfolio_panel.parquet`.

---

## Phase 4 — Feature engineering

Return features:

- [ ] 1-week return.
- [ ] 1-month return.
- [ ] 3-month return.
- [ ] 6-month return.
- [ ] 12-month return.

Risk features:

- [ ] realised volatility windows.
- [ ] downside semideviation.
- [ ] rolling covariance matrix.
- [ ] rolling correlation matrix.
- [ ] historical CVaR proxy.
- [ ] max drawdown state.
- [ ] realised skewness.
- [ ] realised kurtosis.

Factor features:

- [ ] rolling factor betas.
- [ ] rolling factor returns.
- [ ] factor volatility.
- [ ] active factor exposure if benchmark-relative.

Macro features:

- [ ] term spread.
- [ ] credit spread.
- [ ] inflation surprise proxy if available.
- [ ] volatility index level/change.
- [ ] policy-rate change.
- [ ] macro regime indicators.

Portfolio state features:

- [ ] current portfolio weights.
- [ ] previous turnover.
- [ ] current wealth.
- [ ] current drawdown.
- [ ] current CVaR estimate.
- [ ] constraint slack variables.
- [ ] benchmark-relative wealth if applicable.
- [ ] active weights if benchmark-relative.

Implementation tasks:

- [ ] Implement `src/features/returns.py`.
- [ ] Implement `src/features/risk_features.py`.
- [ ] Implement `src/features/factor_features.py`.
- [ ] Implement `src/features/macro_features.py`.
- [ ] Implement `src/features/cost_features.py`.

**Deliverable:** `/data/processed/allocator_state_dataset.parquet`.

---

## Phase 5 — Portfolio environment

Environment mechanics:

- [ ] Observation at rebalance date.
- [ ] Actor proposes post-trade weights.
- [ ] Project proposed weights into admissible set.
- [ ] Compute trades: `new_weights - old_weights`.
- [ ] Apply transaction costs.
- [ ] Apply next-period asset returns.
- [ ] Update wealth.
- [ ] Update drawdown.
- [ ] Update rolling CVaR estimate.
- [ ] Compute reward.
- [ ] Compute constraint violation.
- [ ] Store transition.

Constraints:

- [ ] long-only.
- [ ] max weight per asset.
- [ ] gross exposure limit.
- [ ] cash floor.
- [ ] turnover cap.
- [ ] optional benchmark active-weight cap.
- [ ] optional volatility cap.
- [ ] optional CVaR cap.

Implementation tasks:

- [ ] Implement `src/envs/portfolio_env.py`.
- [ ] Implement `src/envs/constraints.py`.
- [ ] Add tests:
  - [ ] weights sum correctly.
  - [ ] projection respects constraints.
  - [ ] transaction costs reduce wealth.
  - [ ] no-trade path matches buy-and-hold.
  - [ ] CVaR constraint is computed consistently.

**Deliverable:** working `PortfolioEnv`.

---

## Phase 6 — Deterministic baselines

Implement:

- [ ] Cash.
- [ ] Buy-and-hold benchmark.
- [ ] Equal weight.
- [ ] Inverse volatility.
- [ ] Risk parity.
- [ ] Minimum variance.
- [ ] Mean-variance optimiser.
- [ ] Dynamic CVaR optimiser.
- [ ] Benchmark-relative optimiser if applicable.

Implementation tasks:

- [ ] Implement `src/baselines/equal_weight.py`.
- [ ] Implement `src/baselines/inverse_vol.py`.
- [ ] Implement `src/baselines/risk_parity.py`.
- [ ] Implement `src/baselines/mean_variance.py`.
- [ ] Implement `src/baselines/min_variance.py`.
- [ ] Implement `src/baselines/cvar_optimizer.py`.
- [ ] Add `scripts/run_baselines.py`.

**Deliverable:** `reports/tables/baseline_metrics.csv`.

---

## Phase 7 — Unconstrained RL baselines

Implement RL baselines:

- [ ] PPO allocator.
- [ ] SAC allocator.
- [ ] DDPG or TD3 allocator if continuous-action implementation is stable.
- [ ] Same state space.
- [ ] Same action projection.
- [ ] Same transaction costs.
- [ ] No CVaR constraint.

**Deliverable:** unconstrained RL baseline metrics.

---

## Phase 8 — CVaR-constrained actor-critic

Architecture:

- [ ] Actor:
  - [ ] input state.
  - [ ] output portfolio-weight distribution or unconstrained logits.
  - [ ] apply softmax/projection.
- [ ] Return critic:
  - [ ] estimate expected value.
- [ ] Safety critic:
  - [ ] estimate tail loss / CVaR risk.
- [ ] Lagrange multiplier:
  - [ ] increase penalty when CVaR constraint breached.
  - [ ] decrease or stabilise when constraint satisfied.
- [ ] Exploration schedule:
  - [ ] start high.
  - [ ] decay over training.
- [ ] Objective:
  - [ ] maximise expected reward.
  - [ ] penalise CVaR violation.
  - [ ] penalise turnover.
  - [ ] penalise drawdown if needed.

Implementation tasks:

- [ ] Implement `src/models/actor.py`.
- [ ] Implement `src/models/critic.py`.
- [ ] Implement `src/models/safety_critic.py`.
- [ ] Implement `src/models/cvar_actor_critic.py`.
- [ ] Implement `src/training/lagrangian.py`.

**Deliverable:** trainable CVaR-constrained allocator.

---

## Phase 9 — Training loop

Training design:

- [ ] Chronological train/validation/test.
- [ ] Rolling episodes.
- [ ] Weekly rebalancing.
- [ ] Multiple random seeds.
- [ ] Save validation-best model.
- [ ] Log:
  - [ ] reward
  - [ ] return
  - [ ] CVaR estimate
  - [ ] CVaR breach rate
  - [ ] turnover
  - [ ] costs
  - [ ] Lagrange multiplier
  - [ ] entropy/exploration level
  - [ ] max drawdown

Training variants:

- [ ] Without macro features.
- [ ] With factor features only.
- [ ] With macro + factor features.
- [ ] Liability-free version.
- [ ] Benchmark-relative version.

Implementation tasks:

- [ ] Implement `src/training/train_allocator.py`.
- [ ] Add `scripts/train_allocator.py`.

**Deliverable:** model checkpoints and training curves.

---

## Phase 10 — Walk-forward validation

Walk-forward design:

- [ ] Define initial training window.
- [ ] Define validation window.
- [ ] Define test window.
- [ ] Roll forward by fixed step.
- [ ] Refit/retrain where required.
- [ ] Save fold-level predictions, weights, and metrics.

Regime slices:

- [ ] calm equity bull market.
- [ ] equity selloff.
- [ ] inflation/rates shock.
- [ ] credit stress.
- [ ] recovery/risk-on.
- [ ] high-volatility period.
- [ ] low-volatility period.

Implementation tasks:

- [ ] Implement `src/evaluation/walk_forward.py`.

**Deliverable:** `reports/walk_forward_validation.md`.

---

## Phase 11 — Final evaluation

Core metrics:

- [ ] annualised return.
- [ ] annualised volatility.
- [ ] Sharpe.
- [ ] Sortino.
- [ ] Calmar.
- [ ] max drawdown.
- [ ] 1% CVaR.
- [ ] 5% CVaR.
- [ ] expected shortfall.
- [ ] turnover.
- [ ] transaction costs.
- [ ] leverage usage.
- [ ] constraint violation count.
- [ ] CVaR breach rate.
- [ ] hit rate.
- [ ] information ratio if benchmark-relative.
- [ ] tracking error if benchmark-relative.
- [ ] active drawdown if benchmark-relative.

Statistical testing:

- [ ] Paired bootstrap confidence intervals.
- [ ] Fold-level paired tests.
- [ ] Seed-level stability.
- [ ] Regime-level performance comparison.

Implementation tasks:

- [ ] Implement `src/evaluation/metrics.py`.
- [ ] Implement `src/evaluation/backtest.py`.
- [ ] Implement `src/evaluation/bootstrap.py`.

**Deliverable:** `reports/final_report.md`.

---

## Phase 12 — Robustness tests

Run robustness suite:

- [ ] Double transaction costs.
- [ ] Triple transaction costs.
- [ ] Remove macro features.
- [ ] Remove factor features.
- [ ] Remove CVaR constraint.
- [ ] Tighten CVaR threshold.
- [ ] Loosen CVaR threshold.
- [ ] Reduce universe.
- [ ] Expand universe.
- [ ] Change rebalance frequency.
- [ ] Stress one asset class.
- [ ] Train pre-crisis, test crisis.
- [ ] Train crisis-inclusive, test post-crisis.
- [ ] Compare random seeds.
- [ ] Compare against unconstrained PPO/SAC.

**Deliverable:** `reports/robustness_report.md`.

---

## Phase 13 — Diagnostics

Diagnostic charts:

- [ ] Portfolio weights through time.
- [ ] Asset-class weights through time.
- [ ] Turnover through time.
- [ ] Transaction costs through time.
- [ ] Wealth curve.
- [ ] Drawdown curve.
- [ ] CVaR estimate through time.
- [ ] CVaR breach timeline.
- [ ] Lagrange multiplier path.
- [ ] Risk contribution by asset.
- [ ] Factor exposure through time.
- [ ] Macro-regime conditional performance.
- [ ] Action distribution.
- [ ] Policy entropy.
- [ ] Constraint slack variables.

Failure checks:

- [ ] Is the policy just leverage timing?
- [ ] Is the policy just trend following?
- [ ] Is the policy overusing one asset?
- [ ] Is the policy exploiting macro look-ahead?
- [ ] Does performance disappear after costs?
- [ ] Does performance collapse under higher spreads?
- [ ] Does the safety critic actually reduce breaches?

**Deliverable:** `reports/diagnostics_dashboard.html`.

---

## Phase 14 — Ablations

Required ablations:

- [ ] No CVaR constraint.
- [ ] No safety critic.
- [ ] No Lagrange update.
- [ ] No transaction costs.
- [ ] No macro features.
- [ ] No factor features.
- [ ] No rolling covariance features.
- [ ] No drawdown state.
- [ ] Equal reward but different action projection.
- [ ] Actor-only versus actor-critic.
- [ ] Different CVaR confidence levels:
  - [ ] 1%
  - [ ] 5%
  - [ ] 10%
- [ ] Different risk budgets.

**Deliverable:** `reports/ablation_report.md`.

---

## Phase 15 — Paper/demo package

Paper sections:

- [ ] Introduction:
  - [ ] tail-risk-controlled allocation problem.
  - [ ] why unconstrained RL is insufficient.
- [ ] Data:
  - [ ] universe.
  - [ ] features.
  - [ ] splits.
  - [ ] leakage controls.
- [ ] Method:
  - [ ] portfolio MDP.
  - [ ] CVaR constraint.
  - [ ] actor-critic.
  - [ ] safety critic.
  - [ ] Lagrange update.
- [ ] Baselines:
  - [ ] deterministic portfolios.
  - [ ] optimisers.
  - [ ] unconstrained RL.
- [ ] Results:
  - [ ] performance table.
  - [ ] risk table.
  - [ ] CVaR breach table.
  - [ ] regime table.
- [ ] Robustness:
  - [ ] costs.
  - [ ] features.
  - [ ] regimes.
  - [ ] seed stability.
- [ ] Limitations:
  - [ ] backtest overfitting.
  - [ ] reward misspecification.
  - [ ] macro leakage risk.
  - [ ] transaction-cost sensitivity.

**Deliverable:** `reports/paper_draft.md`.

---

## Phase 16 — Final run commands

The repo should support:

```bash
python scripts/build_dataset.py --config configs/data.yaml
python scripts/run_baselines.py --config configs/backtest.yaml
python scripts/train_allocator.py --config configs/training.yaml
python scripts/evaluate_allocator.py --config configs/backtest.yaml
python scripts/make_report.py --experiment_id <ID>
```

**Project complete when:** the above commands rebuild data, run baselines, train constrained RL, evaluate results, and generate final reports without manual notebook intervention.

---

# Acceptance criteria

The repo is acceptable when:

- [ ] Data is leak-free.
- [ ] Weekly decision clock is reproducible.
- [ ] Baselines run.
- [ ] Portfolio environment is tested.
- [ ] Constraints are enforced.
- [ ] Transaction costs are included.
- [ ] Unconstrained RL baseline runs.
- [ ] CVaR-constrained RL model runs.
- [ ] Walk-forward validation runs.
- [ ] CVaR and drawdown metrics are reported.
- [ ] Constraint breaches are reported.
- [ ] Robustness and ablation reports exist.
- [ ] Final report has all key figures and tables.
