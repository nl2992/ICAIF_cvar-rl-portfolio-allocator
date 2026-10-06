"""Paper figure: drawdown paths of the unconstrained and constrained allocators, stress window.

Plots results/tables_camera_ready/stress_paths.csv (scripts/cr_stress_paths.py): the
drawdown of each seed (thin) and of the five-seed mean return path (thick). Test week 16
is the week of 16-20 March 2020 (SPY -14.6%, the worst week in the window), which fixes
the calendar: week k ends on 20 March 2020 + 7(k-16) days.

Usage:
    PYTHONPATH=src python scripts/make_stress_paths_figure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _paper_style as ps
from _paper_style import plt

PATHS = pd.read_csv("results/tables_camera_ready/stress_paths.csv")
COVID_WEEK, COVID_DATE = 16, pd.Timestamp("2020-03-20")
STYLE = {"unconstrained": (ps.MID, "unconstrained"), "constrained": (ps.ACCENT, "CVaR-constrained")}


def drawdown(ret: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1 + ret)
    return wealth / np.maximum.accumulate(wealth) - 1


def week_of(date: str) -> float:
    return COVID_WEEK + (pd.Timestamp(date) - COVID_DATE).days / 7


def main() -> None:
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.6))
    for arm, (colour, label) in STYLE.items():
        d = PATHS[PATHS.arm == arm]
        for _, g in d.groupby("seed"):
            ax.plot(np.arange(len(g)), 100 * drawdown(g.ret.to_numpy()), color=colour, lw=0.35, alpha=0.4)
        mean_ret = d.groupby(d.groupby("seed").cumcount()).ret.mean().to_numpy()
        ax.plot(np.arange(len(mean_ret)), 100 * drawdown(mean_ret), color=colour, lw=1.3, label=label)
    years = ["2020-01-03", "2021-01-01", "2022-01-07", "2023-01-06", "2024-01-05"]
    ax.set_xticks([week_of(y) for y in years], [y[:4] for y in years])
    ax.set_xlim(0, PATHS.groupby(["arm", "seed"]).size().max() - 1)
    ax.set_ylabel("drawdown (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.13, 0.0), handlelength=1.5)
    fig.tight_layout(pad=0.2)
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_stress_paths")


if __name__ == "__main__":
    main()
