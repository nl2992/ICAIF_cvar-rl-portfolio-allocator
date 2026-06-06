"""Matplotlib figure helpers for the report (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_COLORS = {
    "rl_cvar_constrained": "#1f77b4",
    "rl_unconstrained": "#d62728",
    "min_variance": "#2ca02c",
    "inverse_vol": "#9467bd",
    "equal_weight": "#7f7f7f",
    "ppo_best_unconstrained": "#ff7f0e",
    "ppo_best_cvar_constrained": "#17becf",
    "sac_best_unconstrained": "#bcbd22",
    "sac_best_cvar_constrained": "#e377c2",
}


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_wealth(returns: dict[str, pd.Series], path, title="Cumulative wealth (stress window)"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in returns.items():
        ax.plot((1 + r.reset_index(drop=True)).cumprod(), label=name,
                color=_COLORS.get(name), lw=1.8)
    ax.set_xlabel("week"); ax.set_ylabel("growth of $1"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_drawdown(returns: dict[str, pd.Series], path, title="Drawdown (stress window)"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in returns.items():
        w = (1 + r.reset_index(drop=True)).cumprod()
        dd = w / w.cummax() - 1
        ax.plot(dd, label=name, color=_COLORS.get(name), lw=1.5)
    ax.set_xlabel("week"); ax.set_ylabel("drawdown"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_weights_area(weights: pd.DataFrame, path, title="Portfolio weights over time"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = weights.reset_index(drop=True)
    ax.stackplot(range(len(w)), *[w[c] for c in w.columns], labels=list(w.columns), alpha=0.85)
    ax.set_xlabel("week"); ax.set_ylabel("weight"); ax.set_ylim(0, 1); ax.set_title(title)
    ax.legend(fontsize=7, ncol=4, loc="upper center"); ax.margins(x=0)
    return _save(fig, path)


def plot_series(series: dict[str, pd.Series], path, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 4.0))
    for name, s in series.items():
        ax.plot(s.reset_index(drop=True), label=name, color=_COLORS.get(name), lw=1.5)
    ax.set_xlabel("week"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_training_path(history: pd.DataFrame, path, cols=("lagrange", "cvar"),
                       title="Training: Lagrange multiplier & CVaR"):
    fig, ax1 = plt.subplots(figsize=(8, 4.0))
    x = history["update"] if "update" in history else range(len(history))
    ax1.plot(x, history[cols[0]], color="#1f77b4", lw=1.6, label=cols[0])
    ax1.set_xlabel("update"); ax1.set_ylabel(cols[0], color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, history[cols[1]], color="#d62728", lw=1.2, alpha=0.7, label=cols[1])
    ax2.set_ylabel(cols[1], color="#d62728")
    ax1.set_title(title); ax1.grid(alpha=0.3)
    return _save(fig, path)


def plot_grouped_bars(df: pd.DataFrame, label_col, series, path, ylabel, title, colors=None):
    """Grouped bar chart: one group per ``label_col`` row, one bar per ``series`` column."""
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(df)), 4.2))
    x = np.arange(len(df))
    width = 0.8 / len(series)
    for i, (col, lbl) in enumerate(series.items()):
        ax.bar(x + i * width, df[col].to_numpy(), width, label=lbl,
               color=(colors or {}).get(col))
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(df[label_col].astype(str), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
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
