# CVaR-Constrained RL Portfolio Allocator

[![CI](https://github.com/nl2992/ICAIF_cvar-rl-portfolio-allocator/actions/workflows/ci.yml/badge.svg)](https://github.com/nl2992/ICAIF_cvar-rl-portfolio-allocator/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](environment.yml)

<p align="center">
  <img src="paper/figures/figure_cvar_hero.png" width="760" alt="An explicit CVaR constraint cuts the tail of a *learned* allocator (constrained "/>
</p>

<p align="center"><em>An explicit CVaR constraint cuts the tail of a *learned* allocator (constrained vs unconstrained) without sacrificing risk-adjusted return.</em></p>

Research stack for dynamic multi-asset allocation that directly controls downside
tail risk through a **CVaR constraint**. A constrained actor-critic allocator is
compared against deterministic optimisers and an unconstrained RL baseline, after
transaction costs and under portfolio constraints.

## Research question

> Can a constrained actor-critic allocator reduce tail risk and constraint
> breaches versus unconstrained RL while remaining competitive with standard
> portfolio optimisers after costs?

**Headline result** (real 7-ETF universe, trained pre-2018, tested through the
2020 COVID crash + 2022 selloff, mean over 5 seeds): the CVaR constraint cuts
**CVaR-99 by ~49% and max drawdown by ~43%** versus unconstrained RL, while
*raising* Sharpe (0.63→0.88) — competitive with minimum-variance (0.90) at far
lower tail risk. See [`reports/etf_study.md`](reports/etf_study.md).

## What is implemented

| Component | Module |
| --- | --- |
| Regime-switching synthetic data (+ parquet dataset build) | `crlpa/data/synthetic.py`, `scripts/build_dataset.py` |
| **Real ETF data loaders** (Yahoo adjusted close → weekly returns) | `crlpa/data/load_prices.py`, `crlpa/data/build_dataset.py` |
| Allocation environment with no-look-ahead observations, costs, rolling CVaR, drawdown | `crlpa/envs/allocation.py` |
| Admissible-set projection: long-only, max weight, cash floor, turnover & gross caps | `crlpa/envs/constraints.py` |
| Deterministic baselines: equal weight, inverse vol, min-variance, mean-variance, risk parity, min-CVaR (Rockafellar–Uryasev LP) | `crlpa/policies/baselines.py` |
| Model-free CVaR-constrained actor-critic: actor, return critic, **safety critic**, Lagrange dual | `crlpa/models/`, `crlpa/training/lagrangian.py` |
| **Differentiable allocator** (residual policy over adaptive anchor; differentiable CVaR penalty) — the strong learner | `crlpa/training/differentiable.py` |
| Metrics, backtest harness, paired block-bootstrap significance tests | `crlpa/evaluation/` |

The **unconstrained RL baseline** is the same agent with `constrained=False`
(Lagrange multiplier frozen at zero, safety critic ignored).

## Method in brief

At each weekly decision step the actor maps the state to a Gaussian over
pre-softmax logits; a softmax yields long-only weights, which are projected onto
the admissible set. Transaction costs are charged on realised trades and
next-period returns are applied. A rolling historical CVaR of net returns drives
the per-step constraint cost `max(0, CVaR − limit)`. A safety critic estimates the
discounted cost-to-go, and a Lagrange multiplier (projected dual ascent) penalises
the policy objective when the CVaR budget is breached.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"     # rl extra pulls in torch
pytest
```

## End-to-end pipeline

```bash
python scripts/build_dataset.py     --config configs/experiment.yaml
python scripts/run_baselines.py     --config configs/experiment.yaml
python scripts/train_allocator.py   --config configs/experiment.yaml      # or: --episodes 50 --seeds 7 13
python scripts/evaluate_allocator.py --config configs/experiment.yaml
python scripts/make_report.py       --experiment_id cvar_ac_v1
```

Outputs land in `results/tables/` (metrics, statistical tests),
`results/checkpoints/`, `results/training_curves/`, and `reports/final_report.md`.

For the real-ETF differentiable-allocator stress study (the headline result):

```bash
python scripts/build_dataset.py    --config configs/experiment_etf.yaml --out data/processed/aligned_portfolio_panel_etf.parquet
python scripts/run_diff_study.py   --config configs/experiment_etf.yaml   # headline stress study -> results/tables_diff/
python scripts/run_walk_forward.py --config configs/experiment_etf.yaml   # rolling OOS + regime slices + fold tests
python scripts/run_robustness.py   --config configs/experiment_etf.yaml   # costs / CVaR limit / universe perturbations
python scripts/run_ablations.py    --config configs/experiment_etf.yaml   # anchor / alpha / risk budget / objective
python scripts/run_macro_study.py  --config configs/experiment_etf.yaml   # market-only vs macro+factor state
```

Additional model-free baseline: `crlpa/training/ppo.py` (clipped PPO with GAE and
an optional CVaR Lagrangian). The strong learner is the differentiable allocator
(`crlpa/training/differentiable.py`).

## Configuration

`configs/experiment.yaml` is the master config the scripts read; the per-phase
files (`data`, `universe`, `env`, `model`, `training`, `backtest`.yaml) mirror its
sections for readability. Switch `data.source` from `synthetic` to `parquet` to run
off a frozen dataset built by `build_dataset.py`. Canonical seeds: `7, 13, 23, 42, 2025`.

## Repo layout

```text
src/crlpa/
  data/        synthetic regime data + real ETF/macro loaders
  features/    macro features, rolling factor betas, exog builder
  envs/        allocation environment (+ optional exog) + constraint projection
  models/      actor, critic, safety critic, CVaR actor-critic agent
  policies/    deterministic baselines
  training/    A2C + Lagrangian dual, PPO, differentiable allocator
  evaluation/  metrics, backtest, bootstrap, walk-forward, regimes, stress
  experiment.py  config -> data/env/agent builders shared by scripts
configs/       master + per-phase YAML configs (+ experiment_etf.yaml)
scripts/       build_dataset / run_baselines / train_allocator / evaluate_allocator /
               run_diff_study / run_walk_forward / run_robustness / run_ablations /
               run_macro_study / make_report
tests/         env, constraints, metrics, baselines, models, PPO, differentiable,
               features, walk-forward, stress, no-look-ahead
reports/       generated reports + project scope + ETF study
```

## Two data paths

The same environment, baselines, and evaluation harness drive two interchangeable
data sources (switch with `data.source` in the config):

- **Real 7-ETF panel** — the paper's headline numbers. Built from Yahoo adjusted
  close (`crlpa/data/load_prices.py` → `build_dataset.py`), trained pre-2018 and
  tested through the 2020 COVID crash and 2022 selloff. The frozen panel ships under
  `data/processed/`, so the headline study reproduces offline (see *Reproduce* below).
- **Synthetic regime-switching data** (`crlpa/data/synthetic.py`) — a self-contained
  smoke path that runs the full pipeline end-to-end with no external download, used
  for tests and quick iteration.

Walk-forward validation, regime slicing, robustness, and ablation suites run on
either source.


<!-- readme-enhanced -->
## Figures

<img src="paper/figures/eval_protocol_reversal.png" width="480" alt="figure"/>

*Evaluation-protocol reversal: a single stress window ranks the learner #1, while rolling walk-forward inverts the ranking — one of the two failure modes this paper isolates.*

<img src="paper/figures/regime_cvar99.png" width="480" alt="figure"/>

*Regime-sliced CVaR-99: the constraint binds hardest in high-volatility weeks, the domain where the hybrid beats min-variance.*

## Reproduce (data → analysis → paper)

**Prerequisites.** Python 3.11. For the exact pinned environment use conda — `conda env create -f environment.yml && conda activate crlpa` — or with pip:
```bash
pip install -e .
```

### Reproduce the paper's headline numbers

Every number in the paper is regenerated by one committed script writing one committed
artifact. The processed ETF panel ships under `data/processed/`, so each step runs
**offline** and deterministically under the seeds in `configs/` (`7, 13, 23, 42, 2025`).

| Paper claim | Command | Output artifact |
| --- | --- | --- |
| CVaR-99 −49%, max-DD −43%, Sharpe +40%, **0** hard breaches (2020+2022 stress, 5-seed mean) — abstract, Tab. 1 | `python scripts/run_diff_study.py --config configs/experiment_etf.yaml` | `results/tables_diff/stress_metrics.csv`, `stress_constraint_test.csv` |
| Constrained CVaR-99 lower in **26/40** folds, Wilcoxon **p=8.8×10⁻⁶** (40-fold two-universe walk-forward) — Tab. `stats` | `python scripts/run_stats_study.py` | `results/tables_stats/pooled_fold_test.csv`, `per_universe_stats.csv` |
| Regime-switching hybrid **+0.107** Sharpe (1.316 vs 1.209 min-var) — §hybrid | `python scripts/run_hybrid_threshold_sweep.py` | `results/tables/hybrid_threshold_sweep.csv` |
| Constraint-coupling bug ablation (inert multiplier from advantage-scale mismatch) — Tab. `coupling` | `python scripts/coupling_fix_ablation.py --config configs/experiment_etf.yaml` | `results/tables_ablations/coupling_fix_summary.csv` |
| Regime-sliced Sharpe/CVaR (calm / high-vol / selloff) — Tab. `regime` | `python scripts/run_walk_forward.py --config configs/experiment_etf.yaml` | `results/tables_wf/walkforward_regime_*.csv` |
| All paper figures | `python scripts/make_figures.py` | `paper/figures/*.png` |

The compiled paper is **`paper/main.tex → paper/main.pdf`** (build: `cd paper && latexmk -pdf main.tex`).

**Canonicity.** The committed `results/` tables are the exact published numbers. Prices
originate from Yahoo Finance adjusted close (`crlpa/data/load_prices.py`); vendor history
can be revised, so the committed panel and tables — not a fresh download — are canonical
for the published figures. Verified offline: `run_diff_study.py` regenerates
`results/tables_diff/stress_metrics.csv` from the committed panel under the fixed seeds.


---

## Claims and Evidence

What the paper argues, and for every headline number, the committed file it comes from. Artifacts live
under `results/tables/`, `results/tables_ablations/`, and `results/tables_stats/`.

### The narrative

A constrained reinforcement-learning allocator is supposed to do one thing: hold tail risk to a stated
budget while it learns. We build a differentiable CVaR-constrained allocator on a seven-ETF macro
universe (2008–2024, weekly, 5 bps costs), and use it less as a horse-race entry than as a microscope on
two failure modes that affect any constrained financial RL study.

First, the constraint works as a tail safety layer where it matters. Through the 2020 crash and the 2022
selloff it cuts 99% CVaR by 49% and max drawdown by 43% while improving Sharpe by 40% and eliminating
hard constraint violations, and a 40-fold two-universe walk-forward confirms the constrained allocator's
99% CVaR is lower in 26 of 40 folds.

Second, building it surfaces two transferable lessons. The Lagrangian *constraint-coupling failure*:
when you standardise the reward advantage but not the cost advantage, the dual variable never grows —
the constraint is silently inert for any reasonable multiplier — and yet the safety-critic architecture
still regularises tail risk on its own. The *evaluation-protocol reversal*: on a 31-asset universe a
single stress window ranks the learner above every optimiser, while a rolling walk-forward inverts the
ranking, so "beat-the-optimiser" claims can be protocol artifacts.

We are explicit about scope. The pure learned allocator does not beat rolling minimum-variance in
aggregate; a regime-switching hybrid does (+0.107 Sharpe), an edge robust across the volatility-quantile
axis but lost at a deeper −6% selloff-threshold definition. The contribution is the constraint mechanism
and the two evaluation failure modes, not a horse-race win.

### Where each number lives

| Claim | Number | File | Field / row |
|---|---|---|---|
| Constraint cuts crisis-window risk (constrained vs unconstrained) | CVaR99 −49%, MaxDD −43%, Sharpe +40% | per-model CVaR/Sharpe/MaxDD in `results/tables/allocator_metrics.csv`, `results/tables/baseline_metrics.csv`; unconstrained CVaR99 0.0647 in `results/tables/cvar_permutation_test.json` | constrained vs `unconstrained` |
| 40-fold two-universe walk-forward | 26 of 40 folds, p=8.8×10⁻⁶ | `results/tables_stats/pooled_fold_test.csv` | `n_folds`=40, `folds_improved`=26, p=8.804538e-06 |
| Two-universe walk-forward design | test_window=26, etf7 + sector10 | `results/stats_etf.log`, `results/stats_sector.log` | header line |
| Constraint-coupling failure (λ silently inert) | buggy λ=0.022 vs fixed λ=0.101 | `results/tables_ablations/coupling_fix_summary.csv` | `final_lam_mean` (buggy / fixed) |
| Architecture regularises even with λ silenced | constrained 0.025 vs unconstrained 0.065 CVaR99 (≈61%) | `results/tables/cvar_permutation_test.json` | `constrained_cvar99_mean`, `unconstrained_cvar99_mean`, `p_one_sided`=0.0044 |
| Regime-switching hybrid beats min-variance | +0.107 Sharpe (1.316 vs 1.209) | `results/tables/hybrid_overlay_results.json` | `sharpe_gain_vs_minvar`=0.10697 |
| Threshold sensitivity sweep | 10/15 configs win, mean gain +0.077, default (0.75,−0.05)=+0.107 | `results/tables/hybrid_threshold_sweep.csv` | `hybrid_beats_minvar`, `sharpe_gain` (all 5 selloff −0.06 rows erase the gain) |
| Per-regime performance (constraint most active in high-vol) | calm Sharpe 1.44 / high-vol 2.17 / selloff | `results/tables/regime_comparison.json` | `data[]` by `regime`, `model_key=rl_cvar_constrained` |

All numbers regenerate from `scripts/` and `src/crlpa/`.
