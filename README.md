# Making Tail Constraints Bind: Two Silent Failure Modes in CVaR-Constrained Portfolio RL

[![CI](https://github.com/nl2992/ICAIF_cvar-rl-portfolio-allocator/actions/workflows/ci.yml/badge.svg)](https://github.com/nl2992/ICAIF_cvar-rl-portfolio-allocator/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Nigel Li** (University of New South Wales; Columbia University) and **Yutong Han** (Columbia University)

Accepted at **ICAIF '26**, the 7th ACM International Conference on AI in Finance, Milan, 14–17 November 2026.
[Paper (PDF)](paper/main.pdf) · [Citation](#citation)

<p align="center">
  <img src="paper/figures/figure_cvar_hero.png" width="820" alt="Left: on the stress window, the CVaR-constrained allocator sits above and to the left of the unconstrained learner (higher Sharpe, lower CVaR-99). Right: tightening the tail budget improves CVaR-99 and Sharpe together."/>
</p>

<p align="center"><em>Left: stress-window risk–return plane, where the CVaR-constrained allocator (diamond) improves on the unconstrained learner (circle). Right: tightening the tail budget improves CVaR-99 and Sharpe together.</em></p>

This repository contains the code, data and results behind the paper. It implements a
leak-free weekly portfolio-control environment, a differentiable CVaR-constrained
allocator, a decision-time CVaR filter, model-free RL baselines (A2C, PPO, SAC), and the
classical optimisers they are compared against, together with the evaluation protocol
that produces every number in the paper.

## Overview

Two failure modes can leave CVaR-constrained financial reinforcement learning unenforced
without any visible symptom.

1. **Lagrangian constraint-coupling failure.** A scale mismatch between the reward and
   constraint channels (an unstandardised cost advantage, or a dual step mis-scaled to the
   CVaR excess) leaves the dual inert, so the CVaR budget is nominal rather than enforced.
   Tail metrics still improve through CVaR-feasible checkpoint selection, which is why the
   failure is silent.
2. **Evaluation-protocol reversal.** On a 31-asset universe, a single stress window ranks
   the learned allocator above every classical optimiser, while a 20-fold rolling
   walk-forward inverts the ranking, so "beat-the-optimiser" claims can be artefacts of the
   protocol rather than of the policy.

Scaling the dual correctly makes it bind in training, but the budget can still be breached
out of sample. A **decision-time projection** that estimates forward CVaR from trailing
statistics and shrinks non-compliant proposals toward a trailing minimum-variance anchor
restores compliance. The claim is tail-risk control rather than alpha, and it holds under
multiplicity control.

## Key results

Seven-ETF macro universe (SPY, TLT, HYG, DBC, GLD, UUP, BIL), weekly, 2008–2024, 5 bps
costs. The stress window trains before 2018 and tests through the 2020 crash and the 2022
selloff.

| Result | Value |
| --- | --- |
| Stress window, constrained vs unconstrained (5 seeds) | CVaR-99 −49%, max drawdown −43%, Sharpe 0.63 → 0.88 (+40%), Sortino +44%, Calmar +50%, portfolio-constraint violations 7.2 → 0 |
| Stress-window CVaR-99 difference (paired block bootstrap, BCa) | −0.033, 95% interval [−0.035, −0.022] (sector universe: −0.044, [−0.050, −0.032]) |
| 40-fold two-universe walk-forward | 26 of the 27 untied folds improve (13 ties, where the budget never binds); pooled Wilcoxon p = 8.8×10⁻⁶, or 3.3×10⁻⁴ with folds clustered by calendar; Benjamini–Hochberg and Deflated-Sharpe controlled |
| Coupling failure | multiplier stays at λ = 0.022 under the mis-scaled dual (0.101 when scaled); the constrained pipeline still cuts CVaR-99 by 61% (permutation p = 0.0044) |
| Why the scaled dual breaches | 27 of 30 checkpoints meet the validation budget (8 of 30 when mis-scaled), so selection keeps a later, more return-seeking checkpoint that breaches in 75% of test weeks |
| Decision-time filter (A6, matched dual settings) | breach 75.1% → 7.4%, CVaR-99 0.016, Sharpe 0.85, against the classical optimisers' 7.9% |
| Protocol reversal, 31 ETFs | single stress split: learner 0.85 vs minimum variance 0.23 (Ledoit–Wolf 0.25); rolling walk-forward: minimum variance 1.40 vs learner 0.74; the tail reduction survives both (p = 6×10⁻⁴) |
| Regime-switching hybrid | Sharpe 1.316 vs 1.209 for minimum variance, with overlapping bootstrap intervals; ahead in 10 of 15 threshold configurations |

The pure learned allocator does not beat rolling minimum variance in aggregate, and the
paper does not claim that it should. Its contribution is the constraint mechanism and the
two evaluation lessons.

<p align="center">
  <img src="paper/figures/figure_filter_frontier.png" width="460" alt="Compliance frontier: classical optimisers and decision-time filter arms reach breach rates of 7.4 to 7.9 percent at high Sharpe, while soft Lagrangians and the filter-off adaptive dual sit far to the right."/>
</p>

<p align="center"><em>Compliance frontier on the stress window. Filter arms (diamonds) and classical optimisers (squares) reach 7.4–7.9% breach, while soft Lagrangians (circles) and the filter-off adaptive dual (triangle) breach in most weeks.</em></p>

## Method in brief

- **Differentiable allocator.** An MLP proposes a residual tilt over an adaptive
  inverse-volatility anchor, `w_t = softmax(log a_t + f_θ(s_t))`, so the policy equals the
  anchor at initialisation. Training back-propagates the return objective, a differentiable
  CVaR penalty (`topk` over the worst weeks) weighted by a dual variable, and a turnover
  term through the rollout. Model selection keeps the best validation checkpoint among
  those that satisfy the validation CVaR limit.
- **Decision-time CVaR filter.** Between the actor's proposal and the environment, forward
  CVaR is estimated as `c_α σ_p − μ_p` from the trailing 104 weeks (Gaussian CVaR multiplier
  `c_0.95 = 2.06`). If it exceeds the budget, the proposal is shrunk toward the trailing
  minimum-variance anchor by the smallest amount that clears it. The decision at `t` uses
  only returns before `t`, which a dedicated look-ahead test checks.
- **Environment.** Weights are projected onto the admissible set (long-only, maximum weight
  0.40, turnover cap 0.50, gross exposure 1.0), costs are charged on traded notional, and
  features use returns strictly before the decision date.
- **Baselines.** Cash, equal weight, inverse volatility, minimum variance, mean–variance,
  risk parity and a min-CVaR Rockafellar–Uryasev LP, all re-estimated weekly. The learned
  baselines are a model-free A2C actor–critic with a safety critic and Lagrange dual, PPO,
  SAC, and the unconstrained differentiable allocator.

## What is implemented

| Component | Module |
| --- | --- |
| Real ETF loaders (Yahoo adjusted close to weekly returns) and synthetic regime data | `crlpa/data/` |
| Allocation environment with no-look-ahead observations, costs, rolling CVaR and drawdown | `crlpa/envs/allocation.py` |
| Admissible-set projection: long-only, maximum weight, cash floor, turnover and gross caps | `crlpa/envs/constraints.py` |
| Deterministic baselines, including Ledoit–Wolf minimum variance and the min-CVaR LP | `crlpa/policies/baselines.py` |
| Differentiable CVaR-constrained allocator (the paper's learner) | `crlpa/training/differentiable.py` |
| Decision-time CVaR filter and its training loop | `crlpa/training/risk_filter.py` |
| Model-free A2C actor–critic with safety critic and Lagrange dual; PPO; SAC | `crlpa/models/`, `crlpa/training/` |
| Metrics, backtests, block bootstrap (percentile, basic, BCa), walk-forward, regimes, stress splits | `crlpa/evaluation/` |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"     # the rl extra installs torch
pytest
```

The package supports Python 3.10 and later, and CI runs on 3.11. The synthetic pipeline
(`scripts/build_dataset.py`, `run_baselines.py`, `train_allocator.py`,
`evaluate_allocator.py`, `make_report.py` with `configs/experiment.yaml`) runs end to end
with no download and is used for tests and quick iteration.

## Reproducing the paper

### Exact environment

All published results were produced under Python 3.12.2 with the versions pinned in
`requirements.lock.txt`. Rebuilding that environment reproduces every re-run arm exactly
(verified 4 October 2026: 83 of 83 packages matched and all 81 tests pass).

```bash
uv venv --python python3.12 .venv-lock
uv pip install --python .venv-lock/bin/python -r requirements.lock.txt
PYTHONPATH=src .venv-lock/bin/python -m pytest -q
```

Use the lock file rather than newer releases. Under numpy 2.5 and torch 2.14, every
comparison is unchanged except the scaled-dual filter arm (A6), where the η = 5 dual
amplifies differences of about 1e-7 and the seed-42 Sharpe moves from 0.991 to 0.824.
Breach rate and CVaR-99 are unaffected.

The processed panels ship under `data/processed/`, so every step runs offline and
deterministically under the canonical seeds `7, 13, 23, 42, 2025`. To verify data
integrity:

```bash
shasum -a 256 -c DATA_MANIFEST.sha256
```

### Where each result comes from

Each result is regenerated by one script, and its output is committed under `results/`.
Commands assume `PYTHONPATH=src` and `--config configs/experiment_etf.yaml` where the
script takes a config.

| Paper result | Script | Committed output |
| --- | --- | --- |
| Stress window, constrained vs unconstrained | `scripts/run_diff_study.py` | `results/tables_diff/stress_metrics.csv`, `stress_constraint_test.csv` |
| BCa intervals for the stress-window CVaR-99 difference | `scripts/cr_bootstrap_intervals.py --universe etf` (and `sector`) | `results/tables_camera_ready/bootstrap_intervals_*.csv` |
| Coupling ablation (Table "coupling") | `scripts/coupling_fix_ablation.py` | `results/tables_ablations/coupling_fix_summary.csv` |
| 61% CVaR-99 reduction with the multiplier near zero | `scripts/cvar_permutation_test.py` | `results/tables/cvar_permutation_test.json` |
| Checkpoint selection and filter firing per arm and seed | `scripts/cr_selection_log.py` | `results/tables_camera_ready/selection_log_summary.csv` |
| Decision-time filter arms A5–A8 | `scripts/task4_risk_filter.py` | `results/tables_reanalysis/task4_filter_arms_summary.csv` |
| Classical rules re-scored against the same budget | `scripts/task12_analysis.py` | `results/tables_reanalysis/breach_vs_budget_table1.csv` |
| Two-universe walk-forward and fold tests | `scripts/run_stats_study.py`, then `scripts/cr_fold_stats.py` | `results/tables_stats/pooled_fold_test.csv`, `results/tables_camera_ready/fold_stats.json` |
| Ten-fold walk-forward and regime slices | `scripts/run_walk_forward.py`, `scripts/compile_regime_table.py` | `results/tables_wf/`, `results/tables/regime_comparison.json` |
| Regime-switching hybrid and threshold sweep | `scripts/run_hybrid_overlay.py`, `scripts/run_hybrid_threshold_sweep.py` | `results/tables/hybrid_overlay_results.json`, `hybrid_threshold_sweep.csv` |
| Protocol reversal and the Ledoit–Wolf check (31 ETFs) | `configs/experiment_large.yaml` (both protocols, recorded in `reports/research_log.md`), `scripts/make_reversal_figure.py`, `scripts/cr_shrinkage_large31.py` | `results/tables_protocol/protocol_sharpe.csv`, `results/tables_camera_ready/shrinkage_large31_stress.csv` (the 31-asset fold outputs were not retained; their recorded values are in the research log) |
| Robustness and ablations | `scripts/run_robustness.py`, `scripts/run_ablations.py` | `results/tables_robustness/robustness_metrics.csv`, `results/tables_ablations/ablation_metrics.csv` |
| Paper figures | `scripts/make_hero_figure.py`, `scripts/make_filter_frontier_figure.py` | `paper/figures/*.pdf` |

`results/tables_camera_ready/README.md` documents the analyses added for the camera-ready
version. The paper builds with `cd paper && latexmk -pdf main.tex`.

Prices come from Yahoo Finance adjusted closes, and vendor history can be revised, so the
committed panels and tables, not a fresh download, are canonical for the published
numbers.

## Repository layout

```text
src/crlpa/
  data/          ETF and macro loaders, synthetic regime data
  features/      macro features, rolling factor betas
  envs/          allocation environment and constraint projection
  models/        actor, critics, safety critic, CVaR actor-critic agent
  policies/      deterministic baselines
  training/      differentiable allocator, decision-time filter, A2C dual, PPO, SAC
  evaluation/    metrics, backtest, bootstrap, walk-forward, regimes, stress
configs/         experiment configs (synthetic, macro ETF, sector, 31-asset)
scripts/         study, analysis and figure scripts
tests/           environment, constraints, no-look-ahead, filter, statistics and model tests
data/processed/  frozen weekly return panels
results/         committed result tables behind every number in the paper
paper/           LaTeX source, figures and compiled PDF
reports/         research log and study reports
```

## Citation

```bibtex
@inproceedings{li2026tailconstraints,
  title     = {Making Tail Constraints Bind: Two Silent Failure Modes in
               {CVaR}-Constrained Portfolio {RL}},
  author    = {Li, Nigel and Han, Yutong},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM}
}
```

The DOI will be added once the proceedings are published.

## License

MIT. See [LICENSE](LICENSE).
