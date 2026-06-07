"""Matplotlib figure helpers for the report (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Curated, color-blind-friendly palette; protagonists get saturated hues,
# baselines stay muted so the eye lands on the learner vs. the optimiser.
_COLORS = {
    "rl_cvar_constrained": "#1b6ca8",   # ours (constrained)
    "rl_unconstrained": "#d1495b",      # ours (unconstrained)
    "min_variance": "#2e8b57",          # the optimiser that wins OOS
    "cvar_optimizer": "#3aa37a",
    "inverse_vol": "#9b8bd6",
    "equal_weight": "#9aa0a6",
    "ppo_best_unconstrained": "#e8853a",
    "ppo_best_cvar_constrained": "#3bb3c3",
    "sac_best_unconstrained": "#b3a829",
    "sac_best_cvar_constrained": "#cf6fae",
}
_GRID = "#d9dde2"
_INK = "#22262b"


def set_style() -> None:
    """Apply a consistent, publication-grade Matplotlib style for all figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": _INK,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.8,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "DejaVu Sans",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": _GRID,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


set_style()


def _despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _legend(ax, **kw):
    leg = ax.legend(**kw)
    if leg:
        leg.get_frame().set_linewidth(0.8)
    return leg


def _pretty(name: str) -> str:
    return name.replace("rl_", "").replace("_", " ")


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_wealth(returns: dict[str, pd.Series], path, title="Cumulative wealth (stress window)"):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, r in returns.items():
        lw = 2.6 if name.startswith("rl_") else 1.8
        ax.plot((1 + r.reset_index(drop=True)).cumprod(), label=_pretty(name),
                color=_COLORS.get(name), lw=lw, solid_capstyle="round")
    ax.axhline(1.0, color=_GRID, lw=1.0, zorder=0)
    ax.set_xlabel("week"); ax.set_ylabel("growth of \\$1"); ax.set_title(title)
    ax.margins(x=0.01); _despine(ax); _legend(ax, loc="best")
    return _save(fig, path)


def plot_drawdown(returns: dict[str, pd.Series], path, title="Drawdown (stress window)"):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, r in returns.items():
        w = (1 + r.reset_index(drop=True)).cumprod()
        dd = w / w.cummax() - 1
        c = _COLORS.get(name)
        lw = 2.4 if name.startswith("rl_") else 1.6
        ax.plot(dd, label=_pretty(name), color=c, lw=lw)
        if name.startswith("rl_"):
            ax.fill_between(range(len(dd)), dd, 0, color=c, alpha=0.10)
    ax.set_xlabel("week"); ax.set_ylabel("drawdown"); ax.set_title(title)
    ax.margins(x=0.01); _despine(ax); _legend(ax, loc="lower left")
    return _save(fig, path)


def plot_weights_area(weights: pd.DataFrame, path, title="Portfolio weights over time"):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = weights.reset_index(drop=True)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(w.shape[1])]
    ax.stackplot(range(len(w)), *[w[c] for c in w.columns], labels=list(w.columns),
                 colors=colors, alpha=0.9, edgecolor="white", linewidth=0.2)
    ax.set_xlabel("week"); ax.set_ylabel("weight"); ax.set_ylim(0, 1); ax.set_title(title)
    ax.margins(x=0); _despine(ax)
    ncol = min(len(w.columns), 7)
    _legend(ax, loc="upper center", ncol=ncol, fontsize=8, bbox_to_anchor=(0.5, -0.13))
    return _save(fig, path)


def plot_series(series: dict[str, pd.Series], path, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, s in series.items():
        ax.plot(s.reset_index(drop=True), label=_pretty(name),
                color=_COLORS.get(name), lw=2.0)
    ax.set_xlabel("week"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.margins(x=0.01); _despine(ax); _legend(ax, loc="best")
    return _save(fig, path)


def plot_training_path(history: pd.DataFrame, path, cols=("lagrange", "cvar"),
                       title="Training dynamics: Lagrange multiplier vs. CVaR"):
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    x = history["update"] if "update" in history else range(len(history))
    c0, c1 = "#1b6ca8", "#d1495b"
    ax1.plot(x, history[cols[0]], color=c0, lw=2.2, label=cols[0])
    ax1.set_xlabel("update"); ax1.set_ylabel(f"λ ({cols[0]})", color=c0)
    ax1.tick_params(axis="y", colors=c0)
    ax2 = ax1.twinx()
    ax2.plot(x, history[cols[1]], color=c1, lw=1.6, alpha=0.85, label=cols[1])
    ax2.set_ylabel(cols[1], color=c1); ax2.tick_params(axis="y", colors=c1)
    ax2.grid(False)
    for s in ("top",):
        ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    ax1.set_title(title)
    return _save(fig, path)


def plot_grouped_bars(df: pd.DataFrame, label_col, series, path, ylabel, title, colors=None,
                      annotate=True):
    """Grouped bar chart: one group per ``label_col`` row, one bar per ``series`` column."""
    fig, ax = plt.subplots(figsize=(max(7.2, 1.15 * len(df)), 4.4))
    x = np.arange(len(df))
    width = 0.8 / len(series)
    for i, (col, lbl) in enumerate(series.items()):
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x + i * width, vals, width, label=lbl, zorder=3,
                      color=(colors or {}).get(col), edgecolor="white", linewidth=0.6)
        if annotate:
            ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2, color=_INK)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(df[label_col].astype(str), rotation=25, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.margins(y=0.22)  # headroom so the legend clears the bars/value labels
    _despine(ax); ax.grid(alpha=1.0, axis="y"); ax.grid(False, axis="x")
    _legend(ax, loc="upper right", ncol=len(series))
    return _save(fig, path)


def plot_risk_return(df: pd.DataFrame, path, x="sharpe", y="cvar_99",
                     title="Risk vs. return (stress window)"):
    """Scatter of return (x, higher better) vs. tail risk (y, lower better) per strategy.

    Each strategy is one labelled point coloured by the shared palette; the y-axis is
    inverted so the desirable corner (high Sharpe, low tail) is top-right. The figure
    is meant to show the model-free arms chasing return into a high-tail region while
    the constrained allocator sits in the low-tail corner.
    """
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs, ys = df[x].to_numpy(float), df[y].to_numpy(float)
    # pad the data limits so points (and their labels) sit inside the border
    xpad = 0.08 * (xs.max() - xs.min() or 1.0)
    ypad = 0.08 * (ys.max() - ys.min() or 1.0)
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    xmid = 0.5 * (xs.min() + xs.max())
    for name, row in df.iterrows():
        ax.scatter(row[x], row[y], s=90, color=_COLORS.get(name, "#333333"),
                   edgecolor="black", linewidth=0.6, zorder=3)
        # flip the label to the inside for points on the right half, so the text
        # never runs off the right border
        right = row[x] > xmid
        ax.annotate(str(name), (row[x], row[y]), fontsize=7.5, zorder=4,
                    xytext=(-6 if right else 6, 4), textcoords="offset points",
                    ha="right" if right else "left", va="bottom")
    ax.invert_yaxis()  # lower tail risk = better = upward
    ax.set_xlabel(f"{x} (higher is better)")
    ax.set_ylabel(f"{y} (lower is better)")
    ax.set_title(title); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_val_vs_oos(groups: dict, path, reference: tuple | None = None,
                    ylabel="Sharpe", title="Validation selection vs. out-of-sample"):
    """Show how validation-selected model-free configs collapse out-of-sample.

    ``groups`` maps a label -> (array of per-config validation Sharpes, OOS Sharpe of
    the selected config). Each group is one x-slot: a jittered strip of the swept
    configs' validation Sharpes, the selected (max) config marked, and a bar at the
    realised out-of-sample Sharpe. ``reference`` is an optional (value, label) drawn
    as a horizontal line (e.g. the differentiable allocator's OOS Sharpe).
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    xs = np.arange(len(groups))
    vmax = max(np.asarray(v, dtype=float).max() for v, _ in groups.values())
    ax.set_ylim(0, vmax * 1.28)  # headroom so the legend clears the top points
    for i, (name, (val, oos)) in enumerate(groups.items()):
        val = np.asarray(val, dtype=float)
        ax.scatter(i + rng.uniform(-0.12, 0.12, val.size), val, s=24, alpha=0.55,
                   color="#9467bd", zorder=2,
                   label="swept configs (validation)" if i == 0 else None)
        ax.scatter([i], [val.max()], s=120, marker="*", color="#9467bd",
                   edgecolor="black", linewidth=0.6, zorder=4,
                   label="selected (best validation)" if i == 0 else None)
        ax.bar(i, oos, width=0.5, color="#d62728", alpha=0.55, zorder=1,
               label="selected, out-of-sample" if i == 0 else None)
    if reference is not None:
        ax.axhline(reference[0], ls="--", color="#1f77b4", lw=1.6, zorder=3,
                   label=reference[1])
    ax.set_xticks(xs); ax.set_xticklabels(list(groups), fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def plot_forest(df: pd.DataFrame, path, mean=None, ci=None,
                universe_colors=None, title="Per-fold CVaR-99 difference"):
    """Forest/lollipop of per-fold differences (treatment - control).

    ``df`` has columns ``universe``, ``fold``, ``diff``; one horizontal row per fold,
    grouped by universe. A vertical line at zero separates folds where the constraint
    helps (left, diff<0) from those where it hurts (right). ``mean`` and ``ci`` draw
    the pooled mean difference and its bootstrap CI as a line and shaded band.
    """
    palette = universe_colors or {}
    default = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    order = list(dict.fromkeys(df["universe"]))
    rows = []
    for u in order:
        sub = df[df["universe"] == u].sort_values("fold")
        for _, r in sub.iterrows():
            rows.append((u, r["diff"]))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.16 * n + 1.0)))
    if ci is not None:
        ax.axvspan(ci[0], ci[1], color="#1f77b4", alpha=0.12, zorder=0,
                   label="pooled 95% CI")
    if mean is not None:
        ax.axvline(mean, color="#1f77b4", ls="--", lw=1.5, zorder=2,
                   label=f"pooled mean {mean:+.4f}")
    ax.axvline(0, color="black", lw=1.0, zorder=2)
    seen = set()
    for y, (u, d) in enumerate(rows):
        col = palette.get(u, default[order.index(u) % len(default)])
        ax.plot([0, d], [y, y], color=col, lw=1.0, alpha=0.5, zorder=1)
        ax.scatter(d, y, s=26, color=col, edgecolor="black", linewidth=0.4, zorder=3,
                   label=u if u not in seen else None)
        seen.add(u)
    ax.set_yticks([]); ax.set_ylim(-1, n)
    ax.set_xlabel(r"CVaR$_{99}$ difference (constrained $-$ unconstrained); left = constraint reduces tail")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)


def plot_rank_reversal(
    df: pd.DataFrame,
    path,
    left_col: str = "stress",
    right_col: str = "walkforward",
    label_col: str = "strategy",
    left_title: str = "Single stress split",
    right_title: str = "Rolling walk-forward (OOS)",
    highlight: dict[str, str] | None = None,
    title: str = "Same data, opposite verdict: the strategy ranking inverts",
):
    """Slopegraph (bump chart) of a metric under two evaluation protocols.

    Each strategy is a line from its left-protocol value to its right-protocol
    value; crossing lines make a rank reversal unmistakable. ``highlight`` maps a
    strategy name to a colour (others are drawn muted/grey), used to foreground the
    protagonists — the learner that falls and the optimiser that rises.
    """
    highlight = highlight or {}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x0, x1 = 0.0, 1.0
    for _, r in df.iterrows():
        name = str(r[label_col])
        lv, rv = float(r[left_col]), float(r[right_col])
        color = highlight.get(name, "#b0b0b0")
        lw = 3.2 if name in highlight else 1.6
        z = 3 if name in highlight else 1
        ax.plot([x0, x1], [lv, rv], color=color, lw=lw, zorder=z,
                marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.0)
        ax.annotate(f"{name}  {lv:.2f}", (x0, lv), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=color, fontweight="bold" if name in highlight else "normal")
        ax.annotate(f"{rv:.2f}  {name}", (x1, rv), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=color, fontweight="bold" if name in highlight else "normal")

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([left_title, right_title], fontsize=10, fontweight="bold")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(title, fontsize=11)
    for x in (x0, x1):
        ax.axvline(x, color="#dddddd", lw=1.0, zorder=0)
    ax.grid(alpha=0.25, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _save(fig, path)


def plot_dumbbell(
    df: pd.DataFrame,
    path,
    label_col: str = "universe",
    unc_col: str = "unconstrained",
    con_col: str = "constrained",
    annotations: dict | None = None,
    title: str = "The CVaR constraint reduces tail risk across universes",
    xlabel: str = r"CVaR$_{99}$ (weekly; lower is better)",
):
    """Dumbbell/Cleveland chart: per group, a line from unconstrained to constrained.

    The red dot (unconstrained) and blue dot (constrained) with the connecting arrow
    make the size and direction of the tail-risk reduction obvious at a glance;
    ``annotations`` maps a group label to a string drawn at the right (e.g. a p-value).
    """
    annotations = annotations or {}
    c_unc, c_con = _COLORS["rl_unconstrained"], _COLORS["rl_cvar_constrained"]
    n = len(df)
    fig, ax = plt.subplots(figsize=(8.0, 1.05 * n + 1.8))
    ys = np.arange(n)[::-1]
    xmax = float(max(df[unc_col].max(), df[con_col].max()))
    for y, (_, r) in zip(ys, df.iterrows()):
        u, c = float(r[unc_col]), float(r[con_col])
        ax.plot([c, u], [y, y], color="#b8bdc4", lw=3.0, zorder=1, solid_capstyle="round")
        ax.annotate("", xy=(c, y), xytext=(u, y), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color="#7a8089", lw=0))
        ax.scatter([u], [y], s=150, color=c_unc, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([c], [y], s=150, color=c_con, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(f"{u:.4f}", (u, y), xytext=(6, 9), textcoords="offset points",
                    fontsize=8, color=c_unc, ha="center")
        ax.annotate(f"{c:.4f}", (c, y), xytext=(-6, 9), textcoords="offset points",
                    fontsize=8, color=c_con, ha="center")
        red = (u - c) / u * 100 if u else 0.0
        note = annotations.get(str(r[label_col]), "")
        ax.annotate(f"−{red:.0f}%" + (f"   {note}" if note else ""),
                    (xmax, y), xytext=(14, 0), textcoords="offset points",
                    fontsize=8.5, va="center", color=_INK, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(df[label_col].astype(str))
    ax.set_xlim(0, xmax * 1.18)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel); ax.set_title(title)
    _despine(ax); ax.grid(alpha=1.0, axis="x"); ax.grid(False, axis="y")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=c_unc, markersize=10,
               label="unconstrained learner"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=c_con, markersize=10,
               label="CVaR-constrained (ours)"),
    ]
    _legend(ax, handles=handles, loc="lower right")
    return _save(fig, path)
