"""Generate the new model-free / statistics figures from result tables.

On-theme companion to make_figures.py: reuses crlpa.evaluation.plots (shared
palette, Agg backend, dpi 130) so the new figures match the existing set. Reads
the PPO sweep, SAC sweep, and multi-universe stats CSVs; whatever is present is
drawn, missing tables are skipped with a note.

Usage:
    python scripts/make_modelfree_figures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crlpa.evaluation import plots

FIG = Path("reports/figures")
# strategies to surface in the model-free comparison, in display order
ORDER = ["min_variance", "inverse_vol", "rl_cvar_constrained", "rl_unconstrained",
         "ppo_best_unconstrained", "ppo_best_cvar_constrained",
         "sac_best_unconstrained", "sac_best_cvar_constrained"]


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  skip (missing): {path}")
        return None
    return pd.read_csv(path, index_col=0)


def main() -> None:
    made = []
    ppo = _load(Path("results/tables_ppo/ppo_stress_comparison.csv"))
    sac = _load(Path("results/tables_sac/sac_stress_comparison.csv"))

    # combine model-free arms + shared references into one frame (PPO table carries
    # the differentiable arms + baselines; SAC table adds its two arms)
    frames = [f for f in (ppo, sac) if f is not None]
    if frames:
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="first")]
        rows = [s for s in ORDER if s in combined.index]
        comp = combined.loc[rows]

        made.append(plots.plot_risk_return(
            comp[["sharpe", "cvar_99"]], FIG / "risk_return_frontier.png",
            x="sharpe", y="cvar_99",
            title="Risk vs. return on the stress window: model-free arms chase return into the tail"))

        bars = comp.reset_index(names="strategy")
        made.append(plots.plot_grouped_bars(
            bars, "strategy", {"cvar_99": "CVaR-99", "sharpe": "Sharpe"},
            FIG / "modelfree_comparison.png", "value",
            "Model-free (PPO/SAC) vs. differentiable allocator and optimisers"))

    stats = _load(Path("results/tables_stats/per_universe_stats.csv"))
    if stats is not None:
        s = stats.reset_index() if "universe" not in stats.columns else stats
        made.append(plots.plot_grouped_bars(
            s, "universe",
            {"mean_cvar99_unc": "unconstrained", "mean_cvar99_con": "constrained"},
            FIG / "multiuniverse_cvar99.png", "mean CVaR-99",
            "Walk-forward CVaR-99 by universe (constrained vs. unconstrained)",
            colors={"mean_cvar99_unc": "#d62728", "mean_cvar99_con": "#1f77b4"}))

    for p in made:
        print("wrote", p)
    print(f"\n{len(made)} figures in {FIG}/")


if __name__ == "__main__":
    main()
