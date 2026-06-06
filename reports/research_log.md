# Research Log & Records — CVaR-Constrained RL Portfolio Allocator

Durable record of experiments, the seminality judgment, the pivot decision, and
sources. Kept so results, diagrams, and references survive even if a direction is
abandoned. All numbers below are reproduced from the committed pipeline
(`scripts/`, results in `results/`, figures in `reports/figures/`).

---

## 0. Seminality bar (decision criterion)

A **seminal** ICAIF contribution must clear at least one of:

1. The learned method **clearly and robustly beats strong optimiser baselines**
   (minimum-variance, min-CVaR LP) on a meaningful, economically relevant axis, on
   **held-out** data, surviving **multiple-testing / Deflated-Sharpe** correction; or
2. A **broadly general, surprising finding** with implications beyond this universe.

"Reduces tail vs an *unconstrained learner*" is **not** sufficient — the unconstrained
learner is itself weak, so beating it is a low bar.

---

## 1. Phase 1 — initial plan (CVaR-constrained differentiable allocator)

**Setup.** Weekly, 2008–2024, 5 bps costs. Two universes, identical provenance
(Yahoo adjusted close): macro ETF-7 {SPY,TLT,HYG,DBC,GLD,UUP,BIL} and sector-10
{XLK,XLF,XLE,XLV,XLY,XLP,XLU,XLI,XLB,BIL}. Differentiable allocator (pathwise
gradient through a Sharpe objective + topk-CVaR penalty + Lagrangian breach-rate
dual), residual policy over an inverse-vol anchor. Baselines: equal-weight,
inverse-vol, minimum-variance, mean-variance, risk-parity, min-CVaR LP; model-free
A2C/PPO/SAC.

### Verified results

| Result | Number | Artifact |
|---|---|---|
| Stress window (macro), CVaR-99 con vs unc | 0.0331 vs 0.0654 (−49%), Sharpe 0.88 vs 0.63 | `results/tables_diff/`, fig `wealth_curves`,`drawdown_curves` |
| Stress window (sector), CVaR-99 | 0.076 vs 0.109 (−30%), Sharpe ~flat 0.75 | `results/tables_diff_sector/stress_metrics.csv`, `tab:stress_sector` |
| Walk-forward pooled (40 folds) CVaR-99 paired test | Wilcoxon **p=6.7e-5**, bootstrap CI [−0.0082,−0.0033] | `results/tables_stats/pooled_fold_test.csv`, fig `fold_forest_cvar99` |
| Per-universe | sector p=0.0003 (BH reject ✓); macro p=0.11 (marginal) | `results/tables_stats/per_universe_stats.csv`, fig `multiuniverse_cvar99` |
| Deflated Sharpe (constrained OOS) | **DSR ≈ 0** on both universes | `per_universe_stats.csv` |
| Model-free PPO sweep (36 cfg) best, stress | val Sharpe 1.70 → **OOS 0.67**, CVaR-99 0.070 (worst tail) | `results/tables_ppo/`, fig `risk_return_frontier`,`modelfree_generalization` |
| Model-free SAC sweep (8 cfg) best, stress | val Sharpe 1.84 → **OOS 0.39** (collapses) | `results/tables_sac/` |
| Best strategy overall (stress Sharpe) | **minimum variance 0.90** ≈ diff-constrained 0.91 | `tab:modelfree` |

### Diagrams (records)
`reports/figures/`: wealth_curves, drawdown_curves, weights_constrained,
weights_unconstrained, turnover, rolling_cvar, lagrange_path, regime_cvar99,
robustness_cvar99, ablation_cvar99, risk_return_frontier, modelfree_comparison,
modelfree_generalization, multiuniverse_cvar99, fold_forest_cvar99.

---

## 2. JUDGMENT — Phase 1 is sound but NOT seminal → PIVOT

**Decision (agent, per delegated mandate): PIVOT.**

Phase 1 is a clean, honest, *publishable* positive result, but it does **not** clear
the seminal bar:

- **It does not beat the optimisers.** Minimum variance (Sharpe 0.90) matches or
  edges the learned allocator on every axis; the headline contrast is only against an
  unconstrained learner.
- **The Sharpe gain is not deflation-robust** (DSR ≈ 0): after correcting for trials,
  the risk-adjusted-return improvement is indistinguishable from zero.
- **The significant effect is narrow**: tail reduction, driven by *one* of two
  universes (sector p=0.0003; macro p=0.11).

### Why the initial plan was sub-par (retrospective — itself a contribution)
1. **Small, low-correlation macro universe** → rolling minimum-variance is near-optimal
   and barely estimation-fragile; almost no headroom for a learner.
2. **887 weekly points is tiny** for model-free RL → PPO/SAC overfit validation
   (Sharpe 1.7/1.8) and collapse OOS (0.67/0.39). Demonstrated, not asserted.
3. **Tail control ≠ Sharpe dominance** when the unconstrained optimiser already
   controls variance: the constraint removes deep-loss weeks but cannot manufacture
   alpha the optimiser is not already capturing.

This retrospective frames the pivot and becomes the "why it failed" comparison once a
seminal result is found.

---

## 3. Pivot plan (Phase 2) — toward a seminal result

Focus (user-selected): **H5/H6**, GPU-enabled (`.venv-cuda`, torch cu124, RTX 4060).

| # | Hypothesis (win claim) | Grid axes |
|---|---|---|
| H5 | **Scale/dimensionality**: on a large, high-dimensional universe (e.g. DOW-30 / S&P-100 constituents) where MVO estimation error explodes, the learned allocator beats the optimiser net of cost. | universe size, lookback, net capacity |
| H6 | **Temporal encoder**: a GRU/Transformer state encoder extracts timing signal MLP/optimisers cannot → Sharpe + tail win. | encoder∈{MLP,GRU,Transformer}, features, capacity |
| H2 | (secondary) min-variance anchor + learned tail tilt → beat optimiser by construction. | anchor, tilt L2, CVaR budget |
| H4 | (secondary) crisis-alpha metric (Calmar/Sortino/max-DD). | objective, α |
| H1 | (secondary) benchmark-relative active-tail vs 60/40. | benchmark, active-CVaR budget |

### Anti-snooping protocol (mandatory — searching hard for a win invites p-hacking)
- Sweep configs × seeds; **select on validation only**.
- Evaluate the selected config on a **hold-out universe AND period that no config ever
  touches**.
- Report **Deflated-Sharpe deflated by the full number of configs tried**, and
  **Benjamini–Hochberg** across hypotheses.
- A result is "winning" **only** if it beats the optimiser on the untouched hold-out
  and survives DSR>0 + FDR control. Otherwise it joins this log as another documented
  non-result (per repo `project notes`: "Do not force a positive result").

---

## 4. Sources / references (kept for later)

**Read (ICAIF '25, full text available):**
- Lee, Jeon, Bae, Lee. *Return Prediction for Mean-Variance Portfolio Selection: How
  Decision-Focused Learning Shapes Forecasting Models.* ICAIF '25.
  arXiv:2409.09684v4. DOI 10.1145/3768292.3770423. (DFL tilts MSE by Σ⁻¹; two
  universes DOW30/S&P100; intermediate loss-mix wins.)
- Faloughi, Guo, Luk. *ProtoHedge: Interpretable Hedging with Market Prototypes.*
  ICAIF '25. DOI 10.1145/3768292.3770347. (Deep hedging + CVaR utility; interpretable
  prototype policy at <0.40% perf cost.)

**Provided but PASSWORD-PROTECTED (could not parse — request unprotected if needed):**
- `u_2507.20957v4.pdf` (arXiv 2507.20957v4)
- `u_2306.06590v3.pdf` (arXiv 2306.06590v3)

**Already cited in paper bib:** Markowitz 1952; Rockafellar–Uryasev 2000; Schulman
PPO 2017; Haarnoja SAC 2018; Achiam CPO 2017; Buehler Deep Hedging 2019; lee2024dfl;
chung2023mvecf.

---

## 5. Status timeline
- Phase 1 complete; paper at 7 pages (acmart sigconf), compiles 0 overfull/0 undefined
  (TeX Live 2024). Pushed to `feat/cvar-allocator-pipeline`.
- Pivot infra ready: `.venv-cuda` (GPU).

---

## 6. Phase 2 — H5 (scale/dimensionality): STRONG positive signal

Built a 31-asset cross-asset/sector/industry ETF universe (`large31`, full 2008–2024
history, `configs/experiment_large.yaml`). Added a **Ledoit-Wolf shrinkage**
minimum-variance baseline (`baselines.min_variance_shrunk`) so the optimiser
comparison is fair (raw sample covariance is the weak baseline on large universes).

**Stress-window (train pre-2018, test 2018–2024), 3 seeds — dimensionality scaling
of Sharpe (RL constrained minus best optimiser incl. shrinkage & equal-weight):**

| universe | dim | best optimiser | RL constrained | margin |
|---|---|---|---|---|
| macro7   | 7  | min-var 0.90 | 0.88 | **−0.02** (optimiser wins) |
| sector10 | 10 | eq-wt 0.72  | 0.75 | **+0.03** |
| large31  | 31 | eq-wt 0.68  | 0.85 | **+0.17** (learner wins) |

On large31 the learner also cuts **CVaR-99 to 0.027 vs ~0.10 for every optimiser
(−73%)** and max-DD to 0.10 vs 0.27. The advantage over classical optimisers grows
monotonically with dimensionality — the candidate **seminal** claim:
> *Explicit tail-constrained learned allocation beats classical optimisers on
> high-dimensional universes where estimation error dominates, and the margin scales
> with dimensionality.*

**Hardening in progress (anti-snooping):** walk-forward on large31 (20 folds, OOS) to
confirm the win is consistent and not one lucky stress split; then Deflated-Sharpe +
fold-level paired tests, and equal-weight + shrinkage optimisers inside the
walk-forward. A "win" counts only if it survives OOS + DSR.

### CUDA engineering note
The differentiable trainer rolls out **sequentially per timestep with scalar host
syncs** (`.item()`/`.tolist()` each step on tiny tensors). Moving that to CUDA as-is
forces a device↔host sync per step and runs *slower* than 16-thread CPU; GPU only
pays off for a *vectorised/batched* rollout. Phase-2 H5 therefore runs on CPU (the
correct tool). `.venv-cuda` (torch cu124, RTX 4060) is reserved for a future batched
encoder (H6) where it genuinely helps. Forcing a GPU port here would be a
pessimisation, not an optimisation.
