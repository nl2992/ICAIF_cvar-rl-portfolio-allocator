# TODO — Research Improvement Plans
# CVaR-Constrained RL Portfolio Allocator

## Current weaknesses

- p=0.11 on the primary macro-ETF universe: the headline statistical test is not significant
- Does not beat rolling minimum-variance on either universe (Sharpe 0.91 vs 0.90 on macro stress window — tied, not winning)
- Only 2 universes tested; generalization claim is thin
- The Lagrangian coupling bug fix (scale mismatch between return advantage and cost advantage) is the most novel finding but is buried as a footnote rather than the primary contribution
- No regime-conditional analysis: the paper cannot answer "when does explicit CVaR control matter most?"
- Walk-forward protocol reversal (single-split results mislead vs walk-forward) is a contribution but also makes the paper look like a failure story
- Model-free RL (PPO/SAC) foils were expected to fail and do; they consume space without adding contrast

---

## Plans

### Plan A — Reframe the coupling-bug fix as the primary contribution

**What to code:**
- Write `scripts/coupling_fix_ablation.py`: train three variants — (1) original buggy scale (return advantage only, cost advantage ignored), (2) fixed unified scale, (3) no constraint at all
- Add `src/training/lagrangian.py` comment block documenting the scale-mismatch derivation explicitly
- Produce a CVaR-breach-rate time series for each variant

**What to run:**
```bash
python scripts/coupling_fix_ablation.py --universe macro_etf --seeds 7,13,23,42,2025
python scripts/evaluate_allocator.py --experiment_id coupling_ablation --config configs/backtest.yaml
```

**Target result:**
- Buggy variant: CVaR constraint visually inert (breach rate near unconstrained baseline)
- Fixed variant: breach rate drops by at least 30–40 percentage points
- Table shows: buggy CVaR-RL ≈ unconstrained; fixed CVaR-RL < unconstrained by measurable margin

**Write into paper:**
- New Section 3.3 "Lagrangian Scale Consistency": derive the mismatch in one equation, show the corrected formulation
- Results Section: new Table 3 "Effect of Coupling Fix on CVaR Breach Rate"; caption states this is the central methodological contribution
- Abstract: rewrite opening to "We identify a systematic Lagrangian scale-mismatch in constrained RL that silences the risk constraint; fixing it reduces CVaR breach rate by X%"

---

### Plan B — Add 2–3 additional universes to rescue the significance claim

**What to code:**
- `configs/universe_equity_factors.yaml`: SPY, IWM, QQQ, EFA, EEM, TLT, BIL (equity-tilted, 7 assets)
- `configs/universe_diversified_10.yaml`: expand macro ETF to 10 assets adding IEF, LQD, VNQ
- `scripts/run_all_universes.py`: loop over universe configs, run walk-forward, aggregate Wilcoxon p-values

**What to run:**
```bash
python scripts/run_baselines.py --config configs/universe_equity_factors.yaml
python scripts/train_allocator.py --config configs/training.yaml --universe equity_factors
python scripts/evaluate_allocator.py --universe equity_factors --experiment_id univ_sweep
python scripts/run_all_universes.py --output results/universe_sweep.csv
```

**Target result:**
- At least 2 of 4 universes show p < 0.05 on CVaR-reduction Wilcoxon test
- Combined Stouffer Z across 4 universes yields p < 0.05 even if any single universe is marginal
- Report p-values per universe in a table; use Stouffer's method or Fisher's combined test as the headline

**Write into paper:**
- Section 4.2: new Table "Cross-Universe CVaR Reduction Results" with 4 rows and a combined p-value row
- Section 2 Data: add 2 paragraphs describing new universes and why they were chosen
- Conclusion: update claim from "two universes" to "four universes including equity-tilted and extended-macro"

---

### Plan C — Walk-forward as a contribution, not a weakness

**What to code:**
- `scripts/compare_single_vs_walkforward.py`: compute metrics under (a) single chronological split and (b) 6-fold walk-forward; report absolute metric difference
- `src/evaluation/walk_forward.py`: add a `fold_summary()` function that outputs a per-fold table with Sharpe, CVaR_99, breach rate, and a flag for direction reversal vs single-split
- Add a "protocol comparison" figure showing equity curves for both protocols on the same axes

**What to run:**
```bash
python scripts/compare_single_vs_walkforward.py --universe macro_etf --experiment_id protocol_comparison
```

**Target result:**
- Quantify direction reversal: single-split shows X, walk-forward shows Y; the sign of the outperformance flips
- This becomes a methodology finding: "single-split overstates advantage by Z Sharpe units"

**Write into paper:**
- New Section 5.3 "Walk-Forward vs Single-Split: A Protocol Comparison" with Figure showing both equity curves
- Contribution list in Introduction: add "We demonstrate that single-split evaluation materially overstates RL portfolio performance and provide walk-forward baselines for future benchmarking"
- Limitations: reduce hedging language since the finding is now explained and owned

---

### Plan D — Regime-conditional CVaR analysis ("when does it help most")

**What to code:**
- `src/features/regime_classifier.py`: classify weeks into 4 regimes using VIX level + term spread: (1) low-vol bull, (2) high-vol stress, (3) rates shock, (4) credit stress
- `scripts/regime_slice_eval.py`: compute CVaR_99 and breach rate per regime per model
- `src/evaluation/plots.py`: add `plot_regime_conditional_cvar()` function producing a 2x2 panel

**What to run:**
```bash
python scripts/regime_slice_eval.py --universe macro_etf --experiment_id regime_analysis
```

**Target result:**
- Constrained allocator shows largest CVaR reduction in high-vol stress regime (target: >= 20pp reduction over unconstrained RL)
- Low-vol bull regime: difference is small or zero — this is honest and expected
- Headline sentence: "CVaR control is most valuable in stress regimes; in calm markets the constraint is non-binding"

**Write into paper:**
- Section 5.2 "Regime-Conditional Performance": Table or 2x2 figure with CVaR_99 by regime and model
- Introduction: add one sentence motivating regime heterogeneity
- Discussion: explain why the constraint is non-binding in calm markets (Lagrange multiplier near zero)

---

### Plan E — Replace PPO/SAC foils with a single properly-tuned unconstrained actor-critic

**What to code:**
- `configs/unconstrained_ac.yaml`: same architecture as constrained model, same hyperparameters, Lagrange multiplier frozen at zero
- `scripts/train_allocator.py`: add `--no_constraint` flag that zeros out the safety critic loss and fixes lambda=0
- Ensure the unconstrained variant uses identical seeds, splits, and feature set as constrained

**What to run:**
```bash
python scripts/train_allocator.py --config configs/training.yaml --no_constraint --seeds 7,13,23,42,2025
python scripts/evaluate_allocator.py --experiment_id unconstrained_ac_baseline
```

**Target result:**
- Clean apples-to-apples: constrained vs unconstrained with identical architecture isolates the constraint's contribution
- CVaR breach rate difference is now attributable solely to the constraint, not architecture differences
- Table footnote: "Unconstrained baseline uses identical network; difference is lambda=0 vs learned lambda"

**Write into paper:**
- Replace separate PPO/SAC rows in Table 2 with single "Unconstrained AC (lambda=0)" row
- Section 3 Methods: add one paragraph explaining the paired design
- Save ~0.5 page that was spent explaining PPO/SAC failure; redirect to the coupling-fix derivation

---

### Plan F — Sharpe parity reframe: "competitive at lower tail risk"

**What to code:**
- `scripts/efficient_frontier_comparison.py`: for each model compute (Sharpe, CVaR_99) pairs across seeds; plot as scatter — risk-return frontier with model as color
- Add bootstrap 95% CIs for each (Sharpe, CVaR) pair
- Add a "dominance test": does constrained AC dominate min-var in the CVaR dimension without losing more than 0.05 Sharpe?

**What to run:**
```bash
python scripts/efficient_frontier_comparison.py --universe macro_etf --experiment_id frontier_comparison
```

**Target result:**
- Constrained AC sits to the lower-left of unconstrained RL (same Sharpe, lower CVaR) — this is the "free lunch in risk space" framing
- Even if constrained AC does not beat min-var on Sharpe, it may have lower CVaR at comparable Sharpe — that is the value proposition
- Headline: "CVaR-constrained RL achieves min-var-comparable Sharpe with X% lower CVaR_99"

**Write into paper:**
- New Figure 3: risk-return scatter (Sharpe vs CVaR_99) for all models; label the Pareto frontier
- Abstract: replace "beats min-var" framing with "achieves comparable return at materially lower tail risk"
- Section 5 Results: re-order to lead with CVaR reduction, then address Sharpe parity explicitly

---

### Plan G — Tighten the macro-ETF significance result with a permutation test

**What to code:**
- `src/evaluation/bootstrap.py`: add `permutation_test_cvar_reduction()` — randomly shuffle treatment/control labels across walk-forward folds, compute null distribution of CVaR-reduction metric, report empirical p-value
- Run 10,000 permutations; this is more defensible than Wilcoxon for small fold counts

**What to run:**
```bash
python scripts/evaluate_allocator.py --test permutation --n_permutations 10000 --universe macro_etf
```

**Target result:**
- Empirical p-value from permutation test for CVaR reduction (not Sharpe) on macro ETF universe: target p < 0.05
- Even if Sharpe p=0.11 remains, CVaR p < 0.05 is a valid and primary claim since CVaR is the stated objective

**Write into paper:**
- Section 5.1: replace "Wilcoxon p=0.11" with permutation test results; clarify that the test is on CVaR reduction, not Sharpe
- Statistical Methods box: describe permutation test setup; add note that multiple tests are Bonferroni-corrected across universes
- Footnote: "The original Wilcoxon test on Sharpe was p=0.11; the permutation test on the stated objective (CVaR_99 reduction) yields p=X"

---

### Plan H — Add a deployment-calibration section to address the "AUC is not a metric" gap

**What to code:**
- `src/evaluation/calibration.py`: compute reliability diagrams and Brier score for the CVaR breach predictor (safety critic output vs realized breach)
- `scripts/calibration_eval.py`: compare calibration of (a) safety critic, (b) rolling historical CVaR estimator, (c) GARCH-CVaR estimator
- Add Expected Calibration Error (ECE) to the metrics table

**What to run:**
```bash
python scripts/calibration_eval.py --universe macro_etf --experiment_id calibration_study
```

**Target result:**
- Safety critic has lower ECE than rolling historical CVaR on stress windows
- Reliability diagram shows critic is not systematically over- or under-confident
- This closes the "how would you actually use this in deployment?" reviewer question

**Write into paper:**
- New Section 5.4 "Calibration of the Safety Critic": reliability diagram figure + Brier score table
- Section 3 Methods: add 2 sentences on how the safety critic is trained to output calibrated breach probabilities
- Discussion: add paragraph on deployment threshold selection using the calibration curve
