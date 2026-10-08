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

from crlpa.training.risk_filter import gaussian_cvar_multiplier, precompute_filter_state  # noqa: E402

OUT = ("poster/figures",)
ALPHA, BUDGET = 0.95, 0.012
COVID_ROW = 636          # panel row of the week ending 20 March 2020 (stress week 16)
PROPOSAL = {"SPY": 0.40, "HYG": 0.30, "DBC": 0.20, "GLD": 0.10}


def tail_stats(loss: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    var = np.quantile(loss, alpha)
    k = int(np.ceil((1 - alpha) * len(loss)))
    cvar = np.sort(loss)[-k:].mean()
    return var, cvar


def cvar_tail() -> None:
    paths = pd.read_csv("results/tables_camera_ready/stress_paths.csv")
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH_IN, 1.55))
    bins = np.linspace(-6, 10, 49)
    stats = {}
    for arm, colour, face in (("unconstrained", ps.MID, ps.LIGHT), ("constrained", ps.ACCENT, "none")):
        loss = -100 * paths[paths.arm == arm].ret.to_numpy()
        per_seed = [tail_stats(-100 * g.ret.to_numpy()) for _, g in paths[paths.arm == arm].groupby("seed")]
        stats[arm] = np.mean(per_seed, axis=0)
        ax.hist(loss, bins=bins, density=True, histtype="stepfilled" if face != "none" else "step",
                facecolor=face, edgecolor=colour, linewidth=0.9, zorder=2 if face == "none" else 1)
    var_c, cvar_c = stats["constrained"]
    _, cvar_u = stats["unconstrained"]
    top = ax.get_ylim()[1]
    ax.axvspan(var_c, bins[-1], color=ps.ACCENT, alpha=0.08, lw=0, zorder=0)
    ax.vlines(var_c, 0, 1.17 * top, color=ps.ACCENT, lw=0.7, ls=(0, (3, 2)))
    ax.vlines(cvar_c, 0, 1.17 * top, color=ps.ACCENT, lw=1.3)
    ax.vlines(cvar_u, 0, 0.98 * top, color=ps.DARK, lw=1.3)
    ax.text(var_c - 0.12, 1.2 * top, r"VaR$_{0.95}$", color=ps.ACCENT, fontsize=6.5, ha="right", va="bottom")
    ax.text(cvar_c - 0.12, 1.2 * top, f"  CVaR$_{{0.95}}$ = {cvar_c:.1f}% (constrained)", color=ps.ACCENT,
            fontsize=6.5, ha="left", va="bottom")
    ax.text(cvar_u + 0.15, 0.9 * top, f"CVaR$_{{0.95}}$ = {cvar_u:.1f}%\n(unconstrained)", color=ps.DARK,
            fontsize=6.5, ha="left", va="top", linespacing=1.0)
    ax.text(bins[-1] - 0.2, 0.42 * top, "shaded: worst 5% of\nconstrained weeks", color=ps.ACCENT,
            fontsize=6.5, ha="right", va="top", linespacing=1.0)
    ax.set_ylim(0, 1.45 * top)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("weekly loss (%), stress window, five seeds pooled")
    ax.set_ylabel("density")
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    print("cvar_tail stats (per-seed mean VaR, CVaR):", {k: np.round(v, 3) for k, v in stats.items()})
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


if __name__ == "__main__":
    cvar_tail()
    filter_shrink()
