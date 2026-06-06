# CLAUDE.md — CVaR-Constrained RL Portfolio Allocator

## Project identity

Repository: `cvar-rl-portfolio-allocator`

This project builds a dynamic portfolio allocation agent that explicitly controls downside tail risk through a CVaR constraint. The agent should allocate across a liquid multi-asset universe, enforce portfolio constraints, account for transaction costs, and compare against deterministic portfolio optimisers and unconstrained RL baselines.

This file is written for coding/research agents working in this repo. Treat it as the project constitution. Follow it unless a later human instruction explicitly overrides it.

---

## High-level objective

The project seeks to answer:

> Can a CVaR-constrained actor-critic portfolio allocator reduce tail risk and constraint breaches versus unconstrained RL while remaining competitive with standard portfolio optimisers after transaction costs?

The intended contribution is not “RL gets the highest return.” The contribution should be:

1. A leak-free portfolio-control environment.
2. A constrained RL allocator that treats CVaR as an explicit risk constraint.
3. A benchmark-relative or target-relative state design inspired by continuous-time ALM.
4. A comparison against deterministic portfolio construction methods and unconstrained RL.
5. A rigorous evaluation of tail risk, turnover, drawdown, and constraint violations.

The objective is allowed to evolve. If the constrained RL policy does not outperform strong baselines, the project should pivot toward one of these still-valid contributions:

- a robust benchmark suite for tail-risk-aware portfolio RL;
- a study showing where unconstrained RL fails under costs and tail-risk constraints;
- a hybrid dynamic CVaR optimiser with learned regime forecasts;
- a benchmark-relative portfolio environment for future RL/control research.

Do not force a positive result. Document failures clearly.

---

## Research hypothesis

Primary hypothesis:

> An explicit CVaR constraint, implemented through a safety critic or Lagrangian penalty, will reduce downside tail losses and constraint violations relative to unconstrained RL, while preserving competitive net performance versus deterministic portfolio baselines.

Secondary hypotheses:

- Benchmark-relative or target-relative state variables improve stability versus raw return-only states.
- Tail-risk-aware policies should trade less aggressively in high-volatility or high-correlation regimes.
- A constrained policy should have fewer catastrophic drawdown periods than unconstrained PPO/SAC-style baselines.
- A strong dynamic CVaR optimiser may be hard to beat; beating it is not required for the project to be useful if the RL policy provides better adaptability or lower violation rates.

---

## Non-negotiable principles

1. No look-ahead leakage.
2. All train/validation/test splits must be chronological.
3. Macro data must be lagged by availability date.
4. Use vintage macro data where feasible.
5. Transaction costs must be included.
6. Baselines must be implemented before complex RL.
7. The action must respect portfolio constraints.
8. Report tail risk and constraint breaches, not only return.
9. A model is not better if it only wins by using more leverage.
10. Preserve failed experiments.

---

## First viable product

The first version should be deliberately narrow:

- Universe: liquid ETF proxy universe.
- Frequency: weekly rebalancing.
- Constraint set: long-only, fully invested or cash-enabled, max weight cap, turnover cap.
- Objective: maximise cost-adjusted return while controlling CVaR.
- Baselines: cash, equal weight, inverse vol, risk parity, minimum variance, mean-variance, dynamic CVaR optimiser.
- RL models: unconstrained PPO/SAC baseline, then CVaR-constrained actor-critic.

Suggested pilot universe:

```text
SPY  US equities
TLT  long-duration Treasuries
GLD  gold
DBC  commodities
HYG  high yield credit
UUP  US dollar proxy
BIL  cash / T-bill proxy
```

This universe may change if better data is available.

---

## Data requirements

Required raw data:

- Asset prices:
  - adjusted close;
  - total return;
  - dividends/distributions if ETF data;
  - volume;
  - bid-ask proxy if available.
- Cash/risk-free rate.
- Factor data:
  - equity market factor;
  - size/value/profitability/investment factors if relevant;
  - momentum factor if available;
  - internal factor library if available.
- Macro data:
  - term spread;
  - credit spread;
  - inflation proxy;
  - policy rate;
  - volatility index;
  - dollar index or FX proxy if relevant.
- Transaction costs:
  - spread estimate;
  - proportional trading cost;
  - slippage;
  - borrow/funding costs if shorting is later enabled.
- Benchmark data if benchmark-relative:
  - benchmark returns;
  - benchmark weights if available.

Optional:

- regime labels;
- recession indicators;
- liquidity stress indicators;
- implied volatility features;
- drawdown event flags.

---

## Data processing standards

The data pipeline should create three layers:

```text
data/raw/        immutable vendor/API/raw files
data/interim/    cleaned aligned panels
data/processed/  model-ready state/action/backtest data
```

Cleaning and alignment requirements:

- align all assets to the same calendar;
- convert prices to total returns;
- resample to weekly decision dates;
- lag macro features by release availability;
- avoid using revised macro data unless it is explicitly labelled as revised;
- compute next-period returns only after feature construction;
- test for leakage with automated checks;
- save a data dictionary.

Do not use future returns, future volatility, future macro releases, or future benchmark weights in the state.

---

## Feature design

Return features:

- 1-week return;
- 1-month return;
- 3-month return;
- 6-month return;
- 12-month return.

Risk features:

- realised volatility;
- rolling covariance matrix;
- rolling correlation matrix;
- downside semideviation;
- historical CVaR proxy;
- drawdown state;
- skewness and kurtosis if stable.

Factor features:

- rolling factor betas;
- recent factor returns;
- factor volatility;
- active factor exposure if benchmark-relative.

Macro features:

- term spread;
- credit spread;
- policy-rate change;
- inflation proxy;
- volatility index level and change;
- regime indicators.

Portfolio state:

- current weights;
- previous weights;
- current wealth;
- current drawdown;
- current CVaR estimate;
- constraint slack variables;
- benchmark-relative wealth if applicable;
- active weights if benchmark-relative.

---

## Portfolio environment

The environment must simulate allocation sequentially:

1. Observe state at rebalance time.
2. Actor proposes post-trade weights.
3. Project proposed weights into admissible set.
4. Compute trades.
5. Apply transaction costs.
6. Apply next-period returns.
7. Update wealth.
8. Update drawdown.
9. Update rolling CVaR estimate.
10. Compute reward and constraint violations.
11. Store transition.

Action design:

- actor outputs post-trade portfolio weights;
- environment projects invalid weights into feasible set;
- projection must respect max weight, cash floor, gross exposure, and turnover limits;
- transaction costs are charged on trade increments.

Required environment tests:

- weights sum correctly;
- constraints are enforced;
- transaction costs reduce wealth;
- no-trade path matches buy-and-hold;
- CVaR calculation is consistent;
- no look-ahead leakage.

---

## Constraint design

Initial constraints:

- long-only;
- max individual asset weight;
- max turnover per rebalance;
- optional cash floor;
- gross exposure cap of 100% unless otherwise specified.

Risk constraint:

```text
CVaR_alpha(portfolio loss) <= risk_budget
```

Default alpha values to test:

- 1%;
- 5%;
- 10%.

The CVaR constraint may be implemented through:

- Lagrangian penalty;
- safety critic;
- constrained policy optimisation;
- post-action risk filter.

The first implementation should use a Lagrangian penalty plus safety critic if feasible.

---

## Reward design

Default reward:

```text
net portfolio return
minus transaction cost
minus turnover penalty
minus CVaR violation penalty
minus drawdown penalty if needed
```

Do not overfit reward weights. Store every reward weight in config.

Alternative objectives to test:

- return with hard CVaR constraint;
- return minus expected shortfall;
- benchmark-relative excess return minus active CVaR;
- utility based on downside semideviation.

If reward misspecification appears severe, stop and simplify.

---

## Model architecture

Preferred architecture:

```text
portfolio/macro/factor state
        ↓
temporal encoder
        ↓
actor proposes portfolio weights
        ↓
constraint projection
        ↓
portfolio environment
        ↓
return critic + safety critic
        ↓
Lagrangian CVaR update
```

Core components:

- Actor:
  - maps state to portfolio allocation;
  - outputs logits or distribution over weights.
- Return critic:
  - estimates expected value.
- Safety critic:
  - estimates tail loss or CVaR breach risk.
- Lagrange multiplier:
  - increases penalty when risk budget is breached;
  - stabilises when constraint is satisfied.
- Exploration schedule:
  - high early;
  - lower later.

Unconstrained RL baselines must use the same environment and action projection, except without the CVaR penalty/safety critic.

---

## Baselines

Implement these before training the constrained RL policy:

- cash;
- buy-and-hold benchmark;
- equal weight;
- inverse volatility;
- risk parity;
- minimum variance;
- mean-variance optimiser;
- dynamic CVaR optimiser;
- unconstrained PPO/SAC allocator.

If deterministic baselines fail or behave strangely, do not proceed to RL training.

---

## Training standards

Training must use chronological data only.

Required training setup:

- train/validation/test split;
- rolling episodes;
- multiple random seeds;
- validation-best checkpointing;
- fixed transaction costs;
- fixed constraints;
- explicit config snapshot.

Log during training:

- reward;
- net return;
- volatility;
- CVaR estimate;
- CVaR breach rate;
- turnover;
- transaction costs;
- drawdown;
- Lagrange multiplier;
- policy entropy;
- constraint slack.

Training variants:

- no macro features;
- factor features only;
- macro plus factor features;
- liability-free version;
- benchmark-relative version if data allows.

---

## Evaluation standards

Required metrics:

- annualised return;
- annualised volatility;
- Sharpe;
- Sortino;
- Calmar;
- max drawdown;
- 1% CVaR;
- 5% CVaR;
- expected shortfall;
- turnover;
- transaction costs;
- leverage usage;
- constraint violation count;
- CVaR breach rate;
- information ratio if benchmark-relative;
- tracking error if benchmark-relative;
- active drawdown if benchmark-relative.

Required plots:

- wealth curve;
- drawdown curve;
- portfolio weights over time;
- turnover over time;
- transaction costs over time;
- CVaR estimate over time;
- Lagrange multiplier path;
- risk contribution by asset;
- factor exposure over time;
- regime-sliced performance.

Success condition:

> The constrained model should materially reduce tail risk and constraint breaches versus unconstrained RL while remaining competitive against deterministic baselines after costs.

Do not claim success based only on higher return.

---

## Walk-forward validation

Use walk-forward evaluation:

1. Train on historical window.
2. Validate on next chronological window.
3. Test on following out-of-sample window.
4. Roll forward.
5. Aggregate fold-level results.

Evaluate by regime:

- calm markets;
- equity drawdowns;
- rates shock;
- credit stress;
- inflation shock;
- recovery/risk-on;
- high-volatility periods;
- low-volatility periods.

Every final result must include out-of-sample and regime-sliced performance.

---

## Robustness and ablation

Run at least these checks:

- no CVaR constraint;
- no safety critic;
- no Lagrange update;
- no transaction costs;
- doubled transaction costs;
- tripled transaction costs;
- no macro features;
- no factor features;
- no rolling covariance features;
- different rebalance frequency;
- different universe;
- different CVaR levels;
- different risk budgets;
- different random seeds;
- crisis-only test;
- post-crisis test.

If the model collapses under higher costs, report that clearly.

---

## Diagnostics

Always check:

- Is the model just trend following?
- Is the model just leverage timing?
- Is the model overallocating to one asset?
- Is performance driven by one crisis window?
- Does performance disappear after costs?
- Does the policy violate constraints frequently?
- Does the safety critic actually reduce tail events?
- Are macro features leaking future information?
- Are results stable across seeds?

Add diagnostics before making claims.

---

## Expected contribution if successful

A successful project should contribute:

- a leak-free portfolio RL environment;
- a constrained actor-critic implementation for portfolio allocation;
- an explicit CVaR safety layer;
- evidence on whether explicit tail constraints improve RL allocation;
- robust comparison against deterministic and unconstrained RL baselines;
- regime-level diagnostics of when the policy helps or fails.

---

## Acceptable pivots

If the initial result fails, pivot in the following order:

1. Replace RL with a dynamic CVaR optimiser plus learned return/risk forecasts.
2. Keep RL but use it only for regime switching between deterministic allocators.
3. Convert the project into a benchmark suite for constrained portfolio RL.
4. Focus on benchmark-relative active risk rather than absolute wealth.
5. Remove macro features and test whether simpler risk states are more robust.
6. Reduce the universe and action space until the environment is stable.

A negative result is acceptable if it is clean, reproducible, and explains why constrained RL underperformed.

---

## Implementation priorities

Work in this order:

1. Data cleaning.
2. Leak-free feature construction.
3. Portfolio environment.
4. Deterministic baselines.
5. Unconstrained RL baselines.
6. CVaR-constrained RL.
7. Walk-forward validation.
8. Robustness and ablation.
9. Diagnostics dashboard.
10. Paper/demo write-up.

Do not train constrained RL before deterministic baselines and environment tests are complete.

---

## Required commands

The repo should eventually support:

```bash
python scripts/build_dataset.py --config configs/data.yaml
python scripts/run_baselines.py --config configs/backtest.yaml
python scripts/train_allocator.py --config configs/training.yaml
python scripts/evaluate_allocator.py --config configs/backtest.yaml
python scripts/make_report.py --experiment_id <ID>
```

---

## Coding standards

- Prefer clear, typed Python.
- Keep configs outside code.
- Use deterministic seeds.
- Write unit tests for portfolio accounting.
- Avoid notebook-only workflows.
- Keep raw data immutable.
- Save all experiment outputs with config snapshots.
- Log every model comparison in a structured table.
- Use explicit date handling.
- Never assume calendar alignment without checking.
- Never use future macro data or revised values unless explicitly labelled.

---

## Definition of done

The project is done when:

- data can be rebuilt from raw sources;
- no-lookahead tests pass;
- deterministic baselines run;
- portfolio environment is tested;
- unconstrained RL baselines run;
- CVaR-constrained RL trains and evaluates;
- transaction costs are included;
- walk-forward validation exists;
- CVaR and expected shortfall are reported;
- constraint breaches are reported;
- robustness and ablation reports exist;
- final report can be generated from scripts.
