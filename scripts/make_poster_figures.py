"""Poster-only figures, in the shared paper style (scripts/_paper_style.py).

figure_cvar_tail      Weekly loss distribution of the constrained and unconstrained
                      allocators on the stress window, with VaR_0.95 and CVaR_0.95
                      marked. Data: results/tables_camera_ready/stress_paths.csv.
figure_filter_shrink  The decision-time filter on one real decision (the week of
                      16 March 2020). Left, the Gaussian CVaR forecast along the
                      line w(s) = s w + (1 - s) w_MV and the chosen s*. Right, the
                      proposal, the filtered portfolio and the minimum-variance
                      anchor. Trailing statistics come from
                      crlpa.training.risk_filter.precompute_filter_state on
                      data/processed/aligned_portfolio_panel_etf.parquet.

figure_fold_strip     Fold-by-fold change in CVaR_0.99 (constrained minus
                      unconstrained) over the 20 walk-forward folds of each
                      universe. Data: results/tables_stats/folds_{etf7,sector10}.csv.

The proposal in figure_filter_shrink is an illustrative equity-heavy portfolio
at the weight cap, since per-week actor weights are not stored.

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
from crlpa.training.risk_filter import gaussian_cvar_multiplier, precompute_filter_state  # noqa: E402

OUT = ("poster/figures",)
ALPHA, BUDGET = 0.95, 0.012
COVID_ROW = 636          # panel row of the week ending 20 March 2020 (stress week 16)
PROPOSAL = {"SPY": 0.40, "HYG": 0.30, "DBC": 0.20, "GLD": 0.10}


def tail_stats(loss: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """VaR and CVaR of a loss series, with the evaluation code's estimators."""
    return value_at_risk(-loss, alpha), cvar(-loss, alpha)


def cvar_tail() -> None:
    """Pooled five-seed loss histograms with VaR and CVaR of exactly the plotted sample.

    The lines are computed on the pooled sample, so the shaded region holds the
    worst 5% of the plotted constrained weeks and each CVaR line is the mean of
    its own tail. Per-seed means (Table 1 of the poster) are printed alongside.
    """
    paths = pd.read_csv("results/tables_camera_ready/stress_paths.csv")
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.55))
    losses = {arm: -100 * paths[paths.arm == arm].ret.to_numpy() for arm in ("unconstrained", "constrained")}
    lo = np.floor(min(l.min() for l in losses.values()))
    hi = np.ceil(max(l.max() for l in losses.values()))
    bins = np.arange(lo, hi + 1e-9, 1 / 3)
    assert all(((l >= bins[0]) & (l <= bins[-1])).all() for l in losses.values()), "weeks outside bins"
    for arm, colour, face in (("unconstrained", ps.MID, ps.LIGHT), ("constrained", ps.ACCENT, "none")):
        ax.hist(losses[arm], bins=bins, density=True, histtype="stepfilled" if face != "none" else "step",
                facecolor=face, edgecolor=colour, linewidth=0.9, zorder=2 if face == "none" else 1)
    var_c, cvar_c = tail_stats(losses["constrained"])
    var_u, cvar_u = tail_stats(losses["unconstrained"])
    share = (losses["constrained"] >= var_c).mean()
    per_seed = {arm: np.mean([tail_stats(-100 * g.ret.to_numpy()) for _, g in paths[paths.arm == arm].groupby("seed")],
                             axis=0) for arm in losses}
    top = ax.get_ylim()[1]
    ax.axvspan(var_c, bins[-1], color=ps.ACCENT, alpha=0.08, lw=0, zorder=0)
    ax.vlines(var_c, 0, 1.17 * top, color=ps.ACCENT, lw=0.7, ls=(0, (3, 2)))
    ax.vlines(cvar_c, 0, 1.17 * top, color=ps.ACCENT, lw=1.3)
    ax.vlines(cvar_u, 0, 0.98 * top, color=ps.DARK, lw=1.3)
    ax.text(var_c - 0.15, 1.2 * top, r"VaR$_{0.95}$", color=ps.ACCENT, fontsize=6.5, ha="right", va="bottom")
    ax.text(cvar_c - 0.15, 1.2 * top, f"  CVaR$_{{0.95}}$ = {cvar_c:.2f}% (constrained)", color=ps.ACCENT,
            fontsize=6.5, ha="left", va="bottom")
    ax.text(cvar_u + 0.2, 0.9 * top, f"CVaR$_{{0.95}}$ = {cvar_u:.2f}%\n(unconstrained)", color=ps.DARK,
            fontsize=6.5, ha="left", va="top", linespacing=1.0)
    ax.text(bins[-1] - 0.2, 0.42 * top, "shaded: worst 5% of\nconstrained weeks", color=ps.ACCENT,
            fontsize=6.5, ha="right", va="top", linespacing=1.0)
    ax.set_ylim(0, 1.45 * top)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(np.arange(-8, 9, 2))
    ax.set_xlabel("weekly loss (%), stress window, five seeds pooled")
    ax.set_ylabel("density")
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    issues = ps.check_layout(fig)
    print("cvar_tail layout issues:", issues or "none")
    print(f"cvar_tail pooled: constrained VaR {var_c:.3f} CVaR {cvar_c:.3f} (share beyond VaR {share:.4f}); "
          f"unconstrained VaR {var_u:.3f} CVaR {cvar_u:.3f}")
    print("cvar_tail per-seed means (VaR, CVaR):", {k: np.round(v, 3).tolist() for k, v in per_seed.items()})
    ps.save(fig, "figure_cvar_tail", dirs=OUT)


def filter_shrink() -> None:
    panel = pd.read_parquet("data/processed/aligned_portfolio_panel_etf.parquet")
    state = precompute_filter_state(panel, lookback=104, max_weight=0.4)
    t = COVID_ROW
    sigma, mu, anchor = state.sigma[t], state.mu[t], state.anchor[t]
    w = np.array([PROPOSAL.get(c, 0.0) for c in panel.columns])
    c_alpha = gaussian_cvar_multiplier(ALPHA)

    def cvar(x: np.ndarray) -> float:
        return c_alpha * float(np.sqrt(x @ sigma @ x)) - float(mu @ x)

    s_grid = np.linspace(0, 1, 201)
    curve = np.array([cvar(s * w + (1 - s) * anchor) for s in s_grid])
    feasible = s_grid[curve <= BUDGET]
    s_star = float(feasible.max()) if len(feasible) else 0.0
    w_star = s_star * w + (1 - s_star) * anchor
    print(f"filter_shrink: CVaR(proposal)={100 * cvar(w):.2f}%  CVaR(anchor)={100 * cvar(anchor):.2f}%  "
          f"s*={s_star:.2f}  CVaR(filtered)={100 * cvar(w_star):.2f}%")

    ps.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 1.75), gridspec_kw={"width_ratios": [1.05, 1]})
    ax.plot(s_grid, 100 * curve, color=ps.DARK, lw=1.2)
    ax.axhline(100 * BUDGET, color=ps.ACCENT, lw=0.7, ls=(0, (3, 2)))
    ax.plot([s_star, s_star], [ax.get_ylim()[0], 100 * BUDGET], color=ps.ACCENT, lw=0.7)
    ax.scatter([0, 1], [100 * cvar(anchor), 100 * cvar(w)], s=16, marker="s", facecolor=ps.LIGHT,
               edgecolor=ps.INK, linewidth=0.5, zorder=4)
    ax.scatter([s_star], [100 * BUDGET], s=18, marker="D", facecolor=ps.ACCENT, edgecolor=ps.INK,
               linewidth=0.5, zorder=5)
    ax.text(0.05, 100 * cvar(anchor) - 0.05, r"$w^{\mathrm{MV}}$ ($s=0$)", fontsize=6.5, ha="left",
            va="top")
    ax.text(0.93, 100 * cvar(w) + 0.02, r"proposal $w$ ($s=1$)", fontsize=6.5, ha="right", va="center")
    ax.text(s_star + 0.03, 100 * BUDGET - 0.1, rf"$s^\ast={s_star:.2f}$", color=ps.ACCENT, fontsize=6.5,
            ha="left", va="top", bbox=ps.LABEL_BOX)
    ax.text(0.03, 100 * BUDGET + 0.1, r"budget $d=1.2\%$", color=ps.ACCENT, fontsize=6.5, ha="left",
            va="bottom", bbox=ps.LABEL_BOX)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(r"mixing weight $s$ in $s\,w+(1-s)\,w^{\mathrm{MV}}$")
    ax.set_ylabel(r"forecast CVaR$_{0.95}$ (%)")
    ax.set_title("(a) forecast along the shrinkage line", fontsize=7, loc="left")

    names = list(panel.columns)
    y = np.arange(len(names))[::-1]
    h = 0.26
    for k, (vec, face, label) in enumerate(((w, ps.LIGHT, "proposal"), (w_star, ps.ACCENT, "filtered"),
                                            (anchor, "white", "min variance"))):
        bx.barh(y + (1 - k) * h, 100 * vec, height=h, color=face, edgecolor=ps.INK, linewidth=0.4,
                label=label)
    bx.set_yticks(y, names)
    bx.set_xlabel("weight (%)")
    bx.grid(axis="x", color=ps.GRID, lw=0.5)
    bx.grid(axis="y", visible=False)
    bx.set_xlim(0, 60)
    bx.set_xticks([0, 10, 20, 30, 40])
    bx.legend(loc="center right", handlelength=1.0, handleheight=0.7, labelspacing=0.25)
    bx.set_title("(b) portfolio before and after the filter", fontsize=7, loc="left")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    issues = ps.check_layout(fig)
    print("filter_shrink layout issues:", issues or "none")
    ps.save(fig, "figure_filter_shrink", dirs=OUT)


TIE_TOL = 1e-12  # as in scripts/cr_fold_stats.py: a tie means the budget never bound
UNIVERSES = (("etf7", "macro ETF (7 assets)", "Wilcoxon $p=0.0077$"),
             ("sector10", "sector ETF (10 assets)", "Wilcoxon $p=0.0003$"))


def fold_strip() -> None:
    ps.apply()
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 2.1), sharex=True, sharey=True)
    lo = 0.0
    for ax, (key, name, test) in zip(axes, UNIVERSES):
        d = pd.read_csv(f"results/tables_stats/folds_{key}.csv").sort_values("fold")
        k = d.fold.to_numpy() + 1
        delta = 100 * (d.cvar99_con - d.cvar99_unc).to_numpy()
        lo = min(lo, delta.min())
        lower, tied = delta < -100 * TIE_TOL, np.abs(delta) <= 100 * TIE_TOL
        higher = ~lower & ~tied
        ax.axhline(0, color=ps.INK, lw=0.6, zorder=1)
        ax.vlines(k[~tied], 0, delta[~tied], color=ps.MID, lw=0.8, zorder=2)
        ax.scatter(k[lower], delta[lower], s=16, marker="o", facecolor=ps.ACCENT, edgecolor=ps.INK,
                   linewidth=0.4, zorder=3)
        ax.scatter(k[higher], delta[higher], s=16, marker="^", facecolor=ps.LIGHT, edgecolor=ps.INK,
                   linewidth=0.4, zorder=3)
        ax.scatter(k[tied], delta[tied], s=14, marker="o", facecolor="white", edgecolor=ps.DARK,
                   linewidth=0.6, zorder=3)
        summary = (f"{name}: {lower.sum()} lower, {tied.sum()} tied, {higher.sum()} higher, "
                   f"mean {delta.mean():+.2f} pp, {test}").replace("-0.", "−0.")
        ax.set_title(summary, fontsize=6.5, loc="left", pad=2)
        ax.grid(axis="y", color=ps.GRID, lw=0.5)
        print(key, summary.replace("−", "-"))  # cp1252 consoles cannot print U+2212
    axes[0].set_ylim(lo * 1.15, 0.6)
    axes[1].set_xticks(np.arange(1, 21), [str(i) if i % 2 else "" for i in range(1, 21)])
    axes[1].set_xlim(0.4, 20.6)
    axes[1].set_xlabel("walk-forward fold (chronological 26-week test blocks)")
    fig.supylabel(r"$\Delta\,\mathrm{CVaR}_{0.99}$ (pp)", fontsize=7.5, x=0.01)
    handles = [plt.Line2D([0], [0], marker=m, ls="none", markersize=3.6, markerfacecolor=f,
                          markeredgecolor=ps.INK, markeredgewidth=0.5) for m, f in
               (("o", ps.ACCENT), ("o", "white"), ("^", ps.LIGHT))]
    fig.legend(handles, ["constraint lowered the tail", "tie (budget never bound)", "constraint raised it"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.02), handletextpad=0.3,
               columnspacing=1.4)
    fig.tight_layout(pad=0.3, h_pad=0.6, rect=(0.02, 0.07, 1, 1))
    issues = ps.check_layout(fig)
    print("fold_strip layout issues:", issues or "none")
    ps.save(fig, "figure_fold_strip", dirs=OUT)


if __name__ == "__main__":
    cvar_tail()
    filter_shrink()
    fold_strip()
