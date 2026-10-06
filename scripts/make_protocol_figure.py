"""Paper figure: Sharpe on the 31-asset universe under two evaluation protocols.

A slope chart of the values in results/tables_protocol/protocol_sharpe.csv (paper
Section 7.8). It replaces the two-row protocol table: each line is one strategy, so the
crossing of minimum variance and the constrained learner shows the ranking inverting.

Usage:
    PYTHONPATH=src python scripts/make_protocol_figure.py
"""
from __future__ import annotations

import pandas as pd

import _paper_style as ps
from _paper_style import plt

SHARPE = pd.read_csv("results/tables_protocol/protocol_sharpe.csv").set_index("strategy")
DISPLAY = {
    "minimum variance": "minimum variance",
    "inverse vol": "inverse vol",
    "diff. constrained (ours)": "constrained (ours)",
    "diff. unconstrained": "unconstrained",
}
# Vertical label offsets (Sharpe units) where two endpoints sit close together.
NUDGE = {("walkforward", "inverse vol"): 0.035, ("walkforward", "diff. constrained (ours)"): -0.035}


def main() -> None:
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.75))
    for name, row in SHARPE.iterrows():
        ours = "ours" in name
        colour = ps.ACCENT if ours else (ps.DARK if name == "minimum variance" else ps.MID)
        lw = 1.6 if ours or name == "minimum variance" else 0.9
        ax.plot([0, 1], [row.stress, row.walkforward], color=colour, lw=lw, marker="o", ms=3.2,
                zorder=3 if ours else 2)
        for x, col, ha, dx in ((0, "stress", "right", -0.1), (1, "walkforward", "left", 0.1)):
            y = row[col] + NUDGE.get((col, name), 0.0)
            label = (f"{DISPLAY[name]}  {row[col]:.2f}" if x == 0 else f"{row[col]:.2f}  {DISPLAY[name]}")
            ax.text(x + dx, y, label, ha=ha, va="center", fontsize=6.5, color=colour,
                    bbox=ps.LABEL_BOX)
    ax.set_xlim(-1.25, 1.95)
    ax.set_ylim(0.1, 1.5)
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_xticks([0, 1], ["single stress split", "rolling walk-forward"])
    ax.set_ylabel("Sharpe ratio")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_protocol")


if __name__ == "__main__":
    main()
