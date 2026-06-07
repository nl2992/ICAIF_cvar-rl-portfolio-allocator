"""Render the 'robust positive' companion to the reversal slopegraph.

A dumbbell chart of out-of-sample (walk-forward) CVaR-99 per universe: the
unconstrained learner (red) vs. the CVaR-constrained learner (blue). Unlike the
learner-beats-optimiser claim (which reverses under walk-forward), the constraint's
tail-risk reduction *survives* OOS across universes. Overwrites
``multiuniverse_cvar99.png`` (same paper float) with the upgraded visual.

Numbers are the canonical walk-forward per-universe means
(reports/research_log.md / results/tables_stats/per_universe_stats.csv).

Usage:
    python scripts/make_companion_figure.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crlpa.evaluation import plots

# Canonical walk-forward mean CVaR-99 (constrained vs. unconstrained), per universe.
PER_UNIVERSE = pd.DataFrame(
    [
        {"universe": "macro (7 ETF)", "unconstrained": 0.0142, "constrained": 0.0114},
        {"universe": "sector (10 ETF)", "unconstrained": 0.0336, "constrained": 0.0251},
    ]
)


def main() -> None:
    for fig_dir in (Path("reports/figures"), Path("paper/figures")):
        out = plots.plot_dumbbell(
            PER_UNIVERSE, fig_dir / "multiuniverse_cvar99.png",
            annotations={"macro (7 ETF)": "", "sector (10 ETF)": "BH ✓"},
            title=("Constraint reduces out-of-sample tail risk across universes\n"
                   "(walk-forward CVaR$_{99}$; pooled 40-fold Wilcoxon $p<10^{-3}$)"),
        )
        print("wrote", out)


if __name__ == "__main__":
    main()
