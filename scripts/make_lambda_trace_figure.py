"""Paper figure: the Lagrange multiplier over training, mis-scaled vs scaled dual.

Plots results/tables_camera_ready/lambda_trace.csv (scripts/cr_lambda_trace.py): one thin
line per seed and the five-seed mean, for the two coupling arms of tab:coupling. The flat
trace is the symptom that Finding 1 says is the only visible sign of the coupling failure.

Usage:
    PYTHONPATH=src python scripts/make_lambda_trace_figure.py
"""
from __future__ import annotations

import pandas as pd

import _paper_style as ps
from _paper_style import plt

TRACE = pd.read_csv("results/tables_camera_ready/lambda_trace.csv")
STYLE = {"scaled dual": (ps.ACCENT, r"scaled dual ($\eta_\lambda=5$)"),
         "mis-scaled dual": (ps.DARK, r"mis-scaled dual ($\eta_\lambda=0.001$)")}


def main() -> None:
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.55))
    ends = {}
    for arm, (colour, label) in STYLE.items():
        d = TRACE[TRACE.arm == arm]
        for _, g in d.groupby("seed"):
            ax.plot(g["update"], g["lagrange"], color=colour, lw=0.4, alpha=0.35)
        mean = d.groupby("update").lagrange.mean()
        ax.plot(mean.index, mean.values, color=colour, lw=1.4)
        ends[arm] = (mean.index[-1], mean.values[-1], label, colour)
    top = TRACE.lagrange.max()
    ax.set_xlim(0, 1500)
    ax.set_ylim(-0.01, top * 1.05)
    x, y, label, colour = ends["scaled dual"]
    ax.text(30, top * 0.93, label, color=colour, fontsize=6.5, va="top", bbox=ps.LABEL_BOX)
    x, y, label, colour = ends["mis-scaled dual"]
    ax.text(15, top * 0.06, label.replace(" dual (", "\n("), color=colour, fontsize=6.5, ha="left",
            va="bottom", linespacing=1.0, bbox=ps.LABEL_BOX)
    ax.set_xlabel("training update")
    ax.set_ylabel(r"multiplier $\lambda$")
    fig.tight_layout(pad=0.2)
    issues = ps.check_layout(fig)
    print("layout issues:", issues or "none")
    ps.save(fig, "figure_lambda_trace")


if __name__ == "__main__":
    main()
