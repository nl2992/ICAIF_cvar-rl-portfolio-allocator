"""Compliance frontier: breach rate vs Sharpe, one point per arm plus the
classical comparators, budget line drawn (PLAN_P5_make_the_constraint_bind.md
"new visual"). Sober house style (serif, muted navy/maroon/grey, no sentence
titles, no callout bubbles, thin lines, grid off) -- deliberately bypasses
crlpa.evaluation.plots.set_style(), which is bold/grid-on and used elsewhere
in this repo's early figures.

Data: results/tables_reanalysis/task4_filter_arms_summary.csv (A5-A8, this
task), results/tables_reanalysis/breach_vs_budget_table1.csv (classical
comparators + scaled-dual RL, task12_analysis.py), and the coupling-ablation
numbers already published in paper/main.tex Table~\\ref{tab:coupling}
(unconstrained, mis-scaled dual).

Usage: python scripts/make_filter_frontier_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_INK = "#2b2b2b"
_NAVY = "#1D4F91"
_MAROON = "#7A1F2B"
_GREY = "#8a8a8a"
_LIGHT_GREY = "#c9c9c9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": _INK,
    "axes.labelcolor": _INK,
    "axes.titlecolor": _INK,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "xtick.color": _INK,
    "ytick.color": _INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "font.family": "serif",
    "legend.fontsize": 7.6,
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.dpi": 240,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# (label, breach_rate, sharpe, colour, marker, group)
POINTS = [
    ("unconstrained", 1.000, 0.626, _GREY, "o", "penalty"),
    ("mis-scaled dual", 0.079, 0.987, _GREY, "o", "penalty"),
    ("scaled dual", 0.751, 0.911, _GREY, "o", "penalty"),
    ("scaled dual, loose budget", 0.780, 0.543, _GREY, "o", "penalty"),
    ("adaptive dual (A7)", 0.819, 0.826, _MAROON, "^", "adaptive"),
    ("equal weight", 0.628, 0.878, _LIGHT_GREY, "s", "classical"),
    ("risk parity", 0.753, 0.677, _LIGHT_GREY, "s", "classical"),
    ("inverse vol", 0.079, 0.780, _LIGHT_GREY, "s", "classical"),
    ("min-CVaR LP", 0.079, 0.817, _LIGHT_GREY, "s", "classical"),
    ("min variance", 0.079, 0.901, _LIGHT_GREY, "s", "classical"),
    ("filter only (A5)", 0.079, 0.987, _NAVY, "D", "filter"),
    ("filter + scaled dual (A6)", 0.074, 0.851, _NAVY, "D", "filter"),
    ("filter + adaptive dual (A8)", 0.146, 0.800, _NAVY, "D", "filter"),
]

BUDGET_BREACH_RATE = 0.05  # cfg model.cvar_budget: tolerated weekly breach rate


def main() -> None:
    df = pd.DataFrame(POINTS, columns=["label", "breach", "sharpe", "colour", "marker", "group"])

    fig, ax = plt.subplots(figsize=(9.6, 3.5))

    ax.axvline(BUDGET_BREACH_RATE, color=_INK, linewidth=0.8, linestyle=(0, (4, 3)), alpha=0.55)
    ax.text(BUDGET_BREACH_RATE + 0.012, 1.02, "5% target",
            transform=ax.get_xaxis_transform(), fontsize=8.2, color=_INK, alpha=0.8,
            ha="left", va="bottom")

    for group, marker in [("classical", "s"), ("penalty", "o"), ("adaptive", "^"), ("filter", "D")]:
        sub = df[df["group"] == group]
        ax.scatter(sub["breach"], sub["sharpe"], s=46 if group == "filter" else 34,
                   c=sub["colour"], marker=marker, edgecolors=_INK, linewidths=0.5,
                   zorder=4 if group == "filter" else 3)

    # Explicit label anchor points (data coords), laid out to avoid collisions
    # in the dense near-budget cluster on the left.
    label_pos = {
        "mis-scaled dual": (0.20, 1.048),
        "filter only (A5)": (0.20, 0.988),
        "min variance": (0.20, 0.928),
        "filter + scaled dual (A6)": (0.40, 0.866),
        "min-CVaR LP": (0.40, 0.810),
        "filter + adaptive dual (A8)": (0.40, 0.754),
        "inverse vol": (0.20, 0.698),
        "equal weight": (0.48, 0.948),
        "scaled dual": (0.70, 0.985),
        "adaptive dual (A7)": (0.86, 0.888),
        "risk parity": (0.81, 0.642),
        "unconstrained": (0.83, 0.590),
        "scaled dual, loose budget": (0.83, 0.508),
    }
    for _, row in df.iterrows():
        xt, yt = label_pos[row["label"]]
        ha = "left" if xt >= row["breach"] else "right"
        ax.annotate(row["label"], (row["breach"], row["sharpe"]), xytext=(xt, yt),
                    fontsize=8.0, color=_INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=_GREY, linewidth=0.5, shrinkA=2, shrinkB=4))

    ax.set_xlabel("breach rate (share of stress-window weeks over the CVaR budget)")
    ax.set_ylabel("Sharpe (stress window)")
    ax.set_xlim(-0.05, 1.16)
    ax.set_ylim(0.46, 1.09)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=_LIGHT_GREY,
                   markeredgecolor=_INK, markersize=6, label="classical optimiser"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=_GREY,
                   markeredgecolor=_INK, markersize=6, label="soft Lagrangian (no filter)"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=_MAROON,
                   markeredgecolor=_INK, markersize=6, label="adaptive dual, no filter (A7)"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=_NAVY,
                   markeredgecolor=_INK, markersize=6, label="decision-time filter (A5/A6/A8)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              ncol=2, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout()
    for fig_dir in (Path("reports/figures"), Path("paper/figures")):
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = fig_dir / "figure_filter_frontier.png"
        fig.savefig(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
