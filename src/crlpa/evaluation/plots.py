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
    for name, row in df.iterrows():
        ax.scatter(row[x], row[y], s=90, color=_COLORS.get(name, "#333333"),
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(str(name), (row[x], row[y]), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.invert_yaxis()  # lower tail risk = better = upward
    ax.set_xlabel(f"{x} (higher is better)")
    ax.set_ylabel(f"{y} (lower is better)")
    ax.set_title(title); ax.grid(alpha=0.3)
    return _save(fig, path)
