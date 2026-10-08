"""Poster-only figures, in the shared paper style (scripts/_paper_style.py).

figure_cvar_tail      Weekly return distribution of the constrained and unconstrained
                      allocators on the stress window, with -VaR_0.95 and -CVaR_0.95
                      marked in the left tail. Data: results/tables_camera_ready/stress_paths.csv.
figure_fold_scatter   Each of the 40 walk-forward folds as one point, CVaR_0.99
                      without against with the constraint; below the diagonal means
                      lower tail risk. Data: results/tables_stats/folds_{etf7,sector10}.csv.

Usage:
    PYTHONPATH=src python scripts/make_poster_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_style as ps  # noqa: E402
from _paper_style import plt  # noqa: E402

from crlpa.evaluation.metrics import cvar, value_at_risk  # noqa: E402

OUT = ("poster/figures",)
ALPHA = 0.95


def tail_stats(loss: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """VaR and CVaR of a loss series, with the evaluation code's estimators."""
    return value_at_risk(-loss, alpha), cvar(-loss, alpha)


def cvar_tail() -> None:
    """Pooled five-seed weekly-return histograms with VaR and CVaR of the plotted sample.

    Returns are on the x-axis, so the loss tail is the left tail. VaR and CVaR are
    losses (positive numbers); their lines sit at -VaR and -CVaR. The lines are
    computed on the pooled sample, so the shaded region holds the worst 5% of the
    plotted constrained weeks and each CVaR line is the mean of its own tail.
    Per-seed means (Table 1 of the poster) are printed alongside.
    """
    paths = pd.read_csv("results/tables_camera_ready/stress_paths.csv")
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.55))
    rets = {arm: 100 * paths[paths.arm == arm].ret.to_numpy() for arm in ("unconstrained", "constrained")}
    lo = np.floor(min(r.min() for r in rets.values()))
    hi = np.ceil(max(r.max() for r in rets.values()))
    bins = np.arange(lo, hi + 1e-9, 1 / 3)
    assert all(((r >= bins[0]) & (r <= bins[-1])).all() for r in rets.values()), "weeks outside bins"
    for arm, colour, face in (("unconstrained", ps.MID, ps.LIGHT), ("constrained", ps.ACCENT, "none")):
        ax.hist(rets[arm], bins=bins, density=True, histtype="stepfilled" if face != "none" else "step",
                facecolor=face, edgecolor=colour, linewidth=0.9, zorder=2 if face == "none" else 1)
    var_c, cvar_c = tail_stats(-rets["constrained"])
    var_u, cvar_u = tail_stats(-rets["unconstrained"])
    share = (rets["constrained"] <= -var_c).mean()
    per_seed = {arm: np.mean([tail_stats(-100 * g.ret.to_numpy()) for _, g in paths[paths.arm == arm].groupby("seed")],
                             axis=0) for arm in rets}
    top = ax.get_ylim()[1]
    ax.axvspan(bins[0], -var_c, color=ps.ACCENT, alpha=0.08, lw=0, zorder=0)
    ax.vlines(-var_c, 0, 1.17 * top, color=ps.ACCENT, lw=0.7, ls=(0, (3, 2)))
    ax.vlines(-cvar_c, 0, 1.17 * top, color=ps.ACCENT, lw=1.3)
    ax.vlines(-cvar_u, 0, 0.98 * top, color=ps.DARK, lw=1.3)
    ax.text(-var_c + 0.15, 1.2 * top, rf"$-$VaR$_{{0.95}}$", color=ps.ACCENT, fontsize=6.5, ha="left",
            va="bottom")
    ax.text(-cvar_c + 0.15, 1.2 * top, rf"$-$CVaR$_{{0.95}}=-{cvar_c:.2f}$%  ", color=ps.ACCENT, fontsize=6.5,
            ha="right", va="bottom")
    ax.text(-cvar_u - 0.2, 0.9 * top, f"$-$CVaR$_{{0.95}}=-{cvar_u:.2f}$%\n(unconstrained)", color=ps.DARK,
            fontsize=6.5, ha="right", va="top", linespacing=1.0)
    ax.text(bins[0] + 0.2, 0.42 * top, "shaded: worst 5% of\nconstrained weeks", color=ps.ACCENT,
            fontsize=6.5, ha="left", va="top", linespacing=1.0)
    ax.text(bins[-1] - 0.2, 0.9 * top, "constrained (navy)\nunconstrained (grey)", color=ps.DARK,
            fontsize=6.5, ha="right", va="top", linespacing=1.0)
    ax.set_ylim(0, 1.45 * top)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(np.arange(-8, 9, 2))
    ax.set_xlabel("weekly return (%), stress window, five seeds pooled")
    ax.set_ylabel("density")
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    issues = ps.check_layout(fig)
    print("cvar_tail layout issues:", issues or "none")
    print(f"cvar_tail pooled: constrained VaR {var_c:.3f} CVaR {cvar_c:.3f} (share in tail {share:.4f}); "
          f"unconstrained VaR {var_u:.3f} CVaR {cvar_u:.3f}")
    print("cvar_tail per-seed means (VaR, CVaR):", {k: np.round(v, 3).tolist() for k, v in per_seed.items()})
    ps.save(fig, "figure_cvar_tail", dirs=OUT)


TIE_TOL = 1e-12  # as in scripts/cr_fold_stats.py: a tie means the budget never bound
COVID_FOLD = 10   # fold k tests panel rows 364 + 26k .. 389 + 26k; k = 10 is the first half of 2020


def fold_scatter() -> None:
    """Each walk-forward fold as one point: unconstrained against constrained CVaR_0.99.

    Points below the diagonal are folds in which the constraint lowered tail risk;
    points on it are ties (the budget never bound and both learners coincide).
    Log axes, because fold CVaR ranges from 0.3% to 11% and the COVID fold dominates.
    """
    ps.apply()
    fig, ax = plt.subplots(figsize=(3.0, 2.92))
    lo, hi = 0.25, 14.0
    grid = np.geomspace(lo, hi, 50)
    ax.fill_between(grid, lo, grid, color=ps.ACCENT, alpha=0.07, lw=0, zorder=0)
    ax.plot(grid, grid, color=ps.DARK, lw=0.7, zorder=1)
    styles = {"etf7": ("o", ps.ACCENT, ps.INK, "macro ETF (7 assets)"),
              "sector10": ("s", "white", ps.ACCENT, "sector ETF (10 assets)")}
    for key, (marker, face, edge, name) in styles.items():
        d = pd.read_csv(f"results/tables_stats/folds_{key}.csv").sort_values("fold")
        unc, con = 100 * d.cvar99_unc.to_numpy(), 100 * d.cvar99_con.to_numpy()
        delta = con - unc
        lower, tied = delta < -100 * TIE_TOL, np.abs(delta) <= 100 * TIE_TOL
        ax.scatter(unc, con, s=17, marker=marker, facecolor=face, edgecolor=edge, linewidth=0.7, zorder=3,
                   label=f"{name}: lower in {lower.sum()} of {(~tied).sum()} untied folds")
        print(f"fold_scatter {key}: lower {lower.sum()}, tied {tied.sum()}, higher {(~lower & ~tied).sum()}, "
              f"mean CVaR99 {unc.mean():.2f}% -> {con.mean():.2f}%")
        k = int(np.where(d.fold.to_numpy() == COVID_FOLD)[0][0])
        offset = (0.5, 1.6) if key == "etf7" else (0.5, 1.05)
        ax.annotate("2020 H1", (unc[k], con[k]), xytext=(unc[k] * offset[0], con[k] * offset[1]), fontsize=6.5,
                    ha="center", va="bottom", color=ps.DARK, bbox=ps.LABEL_BOX,
                    arrowprops=dict(arrowstyle="-", color=ps.MID, lw=0.5, shrinkA=1, shrinkB=3))
    ax.text(5.5, 0.45, "below the diagonal:\nconstraint lowered\ntail risk", fontsize=6.5, color=ps.ACCENT,
            ha="center", va="bottom", linespacing=1.0)
    ax.text(0.32, 4.2, "above:\nconstraint\nraised it", fontsize=6.5, color=ps.DARK, ha="left",
            va="bottom", linespacing=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.5, 1, 2, 5, 10]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(plt.FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
        axis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(axis="both", color=ps.GRID, lw=0.5)
    ax.set_xlabel(r"CVaR$_{0.99}$ without the constraint")
    ax.set_ylabel(r"CVaR$_{0.99}$ with the constraint")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.2, rect=(0, 0.1, 1, 1))
    issues = ps.check_layout(fig)
    print("fold_scatter layout issues:", issues or "none")
    ps.save(fig, "figure_fold_scatter", dirs=OUT)


if __name__ == "__main__":
    cvar_tail()
    fold_scatter()
