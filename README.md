# Making Tail Constraints Bind: Two Silent Failure Modes in CVaR-Constrained Portfolio RL

[![CI](https://github.com/nl2992/ICAIF_cvar-rl-portfolio-allocator/actions/workflows/ci.yml/badge.svg)](https://github.com/nl2992/ICAIF_cvar-rl-portfolio-allocator/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Nigel Li** (University of New South Wales; Columbia University) and **Yutong Han** (Columbia University)

Accepted at **ICAIF '26**, the 7th ACM International Conference on AI in Finance, Milan, 14-17 November 2026.

[Paper (PDF)](paper/main.pdf) · [Reproduce the results](#reproducing-the-paper) · [Citation](#citation)

This repository contains the paper's implementation, frozen datasets and result tables. We study a differentiable portfolio allocator subject to a Conditional Value-at-Risk (CVaR) budget and show that the standard Lagrangian approach can appear successful even when its constraint is inactive.

The paper makes a narrow claim: **the proposed constraint pipeline and decision-time filter reduce tail risk and improve out-of-sample budget compliance.** It does not claim that the learned allocator consistently generates alpha or beats rolling minimum variance.

## The problem

A Lagrangian CVaR constraint can fail silently. The return objective is order-one, while the weekly CVaR excess is often around `10⁻²`. If the dual step is not scaled to that gap, the multiplier barely moves and the budget is not enforced. Tail metrics may still look good because the anchor, portfolio projection and checkpoint-selection rule also reduce risk.

```mermaid
flowchart LR
    A[Return objective<br/>order 1] --> C[Policy update]
    B[CVaR excess<br/>order 0.01] --> D[Mis-scaled dual update]
    D --> E[Multiplier stays near zero]
    E --> F[Budget exists on paper<br/>but does not shape the policy]
    C --> G[Tail metrics may still improve<br/>through the surrounding pipeline]
    G --> H[Silent failure]
    F --> H
```

The second problem is evaluation: a single favorable test window can reverse the ranking obtained under repeated out-of-sample refits.

<p align="center">
  <img src="paper/figures/figure_protocol.png" width="520" alt="Slope chart showing that the constrained learner leads minimum variance on a single stress split, but minimum variance leads under rolling walk-forward evaluation."/>
</p>

<p align="center"><em>Same 31-asset universe, different protocol: the apparent advantage over minimum variance reverses under rolling walk-forward evaluation.</em></p>

## What we seek to solve

- Make the constraint's effect visible and auditable during training.
- Keep the portfolio within its tail-risk budget after the return distribution shifts.
- Separate genuine tail-risk control from performance claims caused by one favorable backtest.

## Our solution

The learner uses a differentiable CVaR penalty and CVaR-feasible checkpoint selection. At each decision, a safety filter independently estimates forward CVaR from the previous 104 weeks and minimally moves an over-budget proposal toward a trailing minimum-variance portfolio.

```mermaid
flowchart LR
    A[State using data before t] --> B[Residual neural allocator]
    B --> C[Proposed weights]
    C --> D[Estimate forward CVaR<br/>from trailing data]
    D --> E{Within budget?}
    E -- Yes --> G[Admissible-set projection]
    E -- No --> F[Shrink toward trailing<br/>minimum variance]
    F --> G
    G --> H[Long-only portfolio<br/>with weight and turnover caps]
```

The filter is a test-time enforcement layer, not another soft penalty: it changes the portfolio itself before execution. A dedicated no-look-ahead test verifies that the decision at week `t` is unchanged when future returns are shifted.

## Main findings

1. **A CVaR constraint can silently fail.** If the reward and constraint channels are on different scales, the Lagrange multiplier remains nearly inert. In the coupling study, the final multiplier is `0.022` with the mis-scaled update and `0.101` after scaling.

2. **Good tail metrics do not prove that the dual worked.** CVaR-aware checkpoint selection, the inverse-volatility anchor and admissible-set projection can improve tail risk even when the multiplier is effectively inactive. This makes the failure hard to detect without inspecting the multiplier trace.

3. **A binding training penalty does not guarantee test-time compliance.** The scaled dual meets the validation budget at most checkpoints, but the selected policy breaches the budget in `75.1%` of test weeks after the return distribution shifts.

4. **Decision-time projection restores compliance.** Before each trade, the filter estimates forward CVaR from trailing data and shrinks an over-budget proposal toward a trailing minimum-variance portfolio. At matched dual settings, the breach rate falls from `75.1%` to `7.4%`, close to the classical optimizers' `7.9%`.

5. **Strategy rankings depend on the evaluation protocol.** On 31 ETFs, a single stress split favors the constrained learner (Sharpe `0.85` versus `0.23` for minimum variance), while a 20-fold rolling walk-forward reverses the result (`0.74` versus `1.40`). The tail-risk reduction survives both protocols; the apparent optimizer outperformance does not.

## Results at a glance

| Experiment | Result |
| --- | --- |
| Seven-ETF stress window, constrained vs. unconstrained | CVaR-99 `0.0647 → 0.0327` (-49%); max drawdown `20.6% → 11.8%`; Sharpe `0.63 → 0.88` |
| Stress-window CVaR-99 difference | `-0.033`, 95% BCa interval `[-0.035, -0.022]` |
| 40-fold walk-forward across macro and sector universes | 26 of 27 untied folds improve; pooled Wilcoxon `p = 8.8×10⁻⁶`; calendar-clustered `p = 3.3×10⁻⁴` |
| Decision-time filter with scaled dual | Breach rate `75.1% → 7.4%`; CVaR-99 `0.0165`; Sharpe `0.851` |
| 31-ETF protocol comparison | Stress split: learner `0.85`, min-var `0.23`; walk-forward: learner `0.74`, min-var `1.40` |

The study uses weekly data from 2008-2024, 5 bps transaction costs and three universes: seven macro ETFs, ten sector ETFs and a 31-asset cross-asset universe. All features and filter inputs use information available strictly before each decision.

<p align="center">
  <img src="paper/figures/figure_cvar_hero.png" width="820" alt="Stress-window CVaR-99 and Sharpe results for the constrained allocator, unconstrained learner and classical baselines."/>
</p>

<p align="center"><em>Left: the constrained allocator improves stress-window Sharpe while lowering CVaR-99. Right: tighter tail budgets improve both tail risk and Sharpe in the ablation.</em></p>

<p align="center">
  <img src="paper/figures/figure_filter_frontier.png" width="520" alt="Breach rate versus Sharpe for classical optimizers, soft Lagrangian policies, adaptive dual policies and decision-time filter arms."/>
</p>

<p align="center"><em>The decision-time filter moves the scaled-dual allocator from a 75.1% breach rate to 7.4%, near the classical optimizers' 7.9%.</em></p>

## Method

The actor learns a residual tilt over an adaptive inverse-volatility anchor,

```text
w_t = softmax(log(a_t) + f_theta(s_t)).
```

Training back-propagates through portfolio returns using a Sharpe objective, a differentiable worst-tail loss and a turnover penalty. Model selection is restricted to validation checkpoints that meet the CVaR budget.

At decision time, the filter estimates forward CVaR from the trailing 104 weeks. If a proposed portfolio exceeds the budget, it is moved toward a trailing minimum-variance anchor by the smallest amount required to clear the estimate. The final weights remain long-only and obey the maximum-weight, turnover and gross-exposure limits.

The repository also includes cash, equal-weight, inverse-volatility, minimum-variance, mean-variance, risk-parity and min-CVaR baselines, plus model-free A2C, PPO and SAC comparisons.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"
pytest
```

Python 3.10 or later is supported. The committed processed panels allow the paper workflow to run offline.

## Reproducing the paper

The training studies are computationally intensive. Use the pinned environment and committed panels for the published numbers; fresh vendor downloads may differ because historical adjusted prices can be revised.

### 1. Create the exact environment

Published results use Python 3.12.2 and [`requirements.lock.txt`](requirements.lock.txt):

```bash
uv venv --python python3.12 .venv-lock
uv pip install --python .venv-lock/bin/python -r requirements.lock.txt
source .venv-lock/bin/activate
export PYTHONPATH=src
```

All commands below assume the environment is active and are run from the repository root.

### 2. Verify the frozen data and test the pipeline

```bash
shasum -a 256 -c DATA_MANIFEST.sha256
python -m pytest -q
```

The manifest checks the three committed panels under [`data/processed/`](data/processed/). The test suite covers look-ahead prevention, portfolio constraints, the decision-time filter, model training and evaluation statistics.

### 3. Reproduce the stress-window result

Train the constrained and unconstrained allocators over the five canonical seeds and evaluate the 2020-2024 stress window:

```bash
python scripts/run_diff_study.py --config configs/experiment_etf.yaml
python scripts/cr_bootstrap_intervals.py --universe etf
python scripts/cr_bootstrap_intervals.py --universe sector
```

- Training script: [`scripts/run_diff_study.py`](scripts/run_diff_study.py)
- Stress metrics: [`results/tables_diff/stress_metrics.csv`](results/tables_diff/stress_metrics.csv)
- Paired test: [`results/tables_diff/stress_constraint_test.csv`](results/tables_diff/stress_constraint_test.csv)
- BCa intervals: [`results/tables_camera_ready/bootstrap_intervals_etf.csv`](results/tables_camera_ready/bootstrap_intervals_etf.csv) and [`bootstrap_intervals_sector.csv`](results/tables_camera_ready/bootstrap_intervals_sector.csv)

### 4. Reproduce the constraint-coupling diagnosis

Compare the mis-scaled dual, scaled dual, unconstrained learner and loose-budget arm; then test the CVaR-99 difference and record checkpoint-selection diagnostics:

```bash
python scripts/coupling_fix_ablation.py --config configs/experiment_etf.yaml
python scripts/cvar_permutation_test.py
python scripts/cr_selection_log.py
python scripts/cr_selection_log.py --summarise
```

- Coupling study: [`scripts/coupling_fix_ablation.py`](scripts/coupling_fix_ablation.py)
- Summary: [`results/tables_ablations/coupling_fix_summary.csv`](results/tables_ablations/coupling_fix_summary.csv)
- Permutation test: [`results/tables/cvar_permutation_test.json`](results/tables/cvar_permutation_test.json)
- Checkpoint diagnostics: [`results/tables_camera_ready/selection_log_summary.csv`](results/tables_camera_ready/selection_log_summary.csv)

### 5. Reproduce the decision-time filter result

First score the scaled-dual learner and classical strategies against the same budget, then run the four filter/adaptive-dual arms:

```bash
python scripts/task12_analysis.py
python scripts/task4_risk_filter.py
python scripts/make_filter_frontier_figure.py
```

- Classical comparison: [`results/tables_reanalysis/breach_vs_budget_table1.csv`](results/tables_reanalysis/breach_vs_budget_table1.csv)
- Filter-arm summary: [`results/tables_reanalysis/task4_filter_arms_summary.csv`](results/tables_reanalysis/task4_filter_arms_summary.csv)
- Weight-distance diagnostics: [`results/tables_reanalysis/task4_filter_weight_distance_summary.csv`](results/tables_reanalysis/task4_filter_weight_distance_summary.csv)
- Rebuilt figure: [`paper/figures/figure_filter_frontier.png`](paper/figures/figure_filter_frontier.png)

### 6. Reproduce the rolling walk-forward evidence

The first command refits both the seven-ETF and ten-sector universes in 26-week test folds. The second performs the dependence-aware analysis on the committed per-universe fold files, including calendar clustering, exact sign flips and moving-block bootstrap intervals.

```bash
python scripts/run_stats_study.py
python scripts/cr_fold_stats.py
```

- Walk-forward driver: [`scripts/run_stats_study.py`](scripts/run_stats_study.py)
- Pooled test: [`results/tables_stats/pooled_fold_test.csv`](results/tables_stats/pooled_fold_test.csv)
- Per-universe statistics: [`results/tables_stats/per_universe_stats.csv`](results/tables_stats/per_universe_stats.csv)
- Dependence-aware results: [`results/tables_camera_ready/fold_stats.json`](results/tables_camera_ready/fold_stats.json)
- Fold-by-fold calendar table: [`results/tables_camera_ready/fold_stats_per_fold.csv`](results/tables_camera_ready/fold_stats_per_fold.csv)

For the ten-fold macro walk-forward and regime table used elsewhere in the paper:

```bash
python scripts/run_walk_forward.py --config configs/experiment_etf.yaml
python scripts/compile_regime_table.py
```

### 7. Rebuild the protocol-reversal comparison

```bash
python scripts/make_reversal_figure.py
python scripts/make_protocol_figure.py
```

This recreates [`results/tables_protocol/protocol_sharpe.csv`](results/tables_protocol/protocol_sharpe.csv) and [`paper/figures/figure_protocol.png`](paper/figures/figure_protocol.png). The raw 31-asset walk-forward folds were not retained, so this step rebuilds the published comparison from the authoritative values in the paper rather than retraining those folds.

### 8. Rebuild the remaining figures and paper

```bash
python scripts/make_stress_paths_figure.py
python scripts/make_lambda_trace_figure.py
python scripts/make_filter_frontier_figure.py
python scripts/make_hero_figure.py
cd paper
latexmk -pdf main.tex
```

The figure scripts write PDF and PNG versions to [`paper/figures/`](paper/figures/); the paper builds to [`paper/main.pdf`](paper/main.pdf).

### Result map

| Result | Script | Output |
| --- | --- | --- |
| Stress-window comparison | [`run_diff_study.py`](scripts/run_diff_study.py) | [`results/tables_diff/`](results/tables_diff/) |
| Coupling failure | [`coupling_fix_ablation.py`](scripts/coupling_fix_ablation.py) | [`coupling_fix_summary.csv`](results/tables_ablations/coupling_fix_summary.csv) |
| Decision-time filter | [`task4_risk_filter.py`](scripts/task4_risk_filter.py) | [`task4_filter_arms_summary.csv`](results/tables_reanalysis/task4_filter_arms_summary.csv) |
| Two-universe fold statistics | [`run_stats_study.py`](scripts/run_stats_study.py), [`cr_fold_stats.py`](scripts/cr_fold_stats.py) | [`results/tables_stats/`](results/tables_stats/), [`fold_stats.json`](results/tables_camera_ready/fold_stats.json) |
| Protocol reversal | [`make_reversal_figure.py`](scripts/make_reversal_figure.py) | [`protocol_sharpe.csv`](results/tables_protocol/protocol_sharpe.csv) |
| Camera-ready analyses | [documentation](results/tables_camera_ready/README.md) | [`results/tables_camera_ready/`](results/tables_camera_ready/) |

## Repository layout

```text
src/crlpa/       data, environments, policies, training and evaluation
configs/         macro ETF, sector ETF and 31-asset experiment configs
scripts/         experiments, statistical analyses and figure generation
tests/           no-look-ahead, constraints, models and evaluation tests
data/processed/  frozen weekly return panels
results/         committed tables behind the paper's reported numbers
paper/           camera-ready LaTeX source, figures and PDF
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

The DOI will be added after publication.

## License

MIT. See [LICENSE](LICENSE).
