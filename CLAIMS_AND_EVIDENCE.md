# Claims and Evidence

What the paper argues, and for every headline number, the committed file it comes from. Artifacts live
under `results/tables/`, `results/tables_ablations/`, and `results/tables_stats/`.

## The narrative

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

## Where each number lives

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
