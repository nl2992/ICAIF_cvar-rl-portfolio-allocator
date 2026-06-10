"""Reconstruct the CVaR hero figure (figure_cvar_hero) in the Columbia theme.

Two risk--return panels (x = CVaR99 in %, axis inverted so *right = safer*;
y = annualised Sharpe):
  (a) Stress window 2020--2022: the constrained allocator moves up-and-right of the
      unconstrained learner (higher Sharpe, lower tail), competitive with min-variance.
  (b) Ablation: tightening the tail budget improves BOTH CVaR and Sharpe monotonically.

All coordinates are the authoritative values committed in the paper tables
(Tables: stress window, tuned model-free, ablation, walk-forward), so the figure is
consistent with the text by construction. Colours come from the shared
``crlpa.evaluation.plots`` palette so the hero matches every other figure
(constrained = Columbia navy, unconstrained = red, min-variance = green).

    python scripts/make_hero_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crlpa.evaluation.plots import _COLORS, _INK, set_style  # noqa: E402

OUT = ROOT / "paper" / "figures" / "figure_cvar_hero.png"

C_CON = _COLORS["rl_cvar_constrained"]   # Columbia navy (ours)
C_UNC = _COLORS["rl_unconstrained"]      # red
C_MV = _COLORS["min_variance"]           # green
C_PPO = _COLORS["ppo_best_unconstrained"]  # orange
C_EW = _COLORS["equal_weight"]           # blue-grey

# --- Panel (a): stress-window (Sharpe, CVaR99 %) from the paper tables ---------
STRESS = {  # name: (CVaR99_pct, Sharpe, colour, marker, size)
    "Unconstrained RL": (6.47, 0.63, C_UNC, "o", 150),
    "PPO (best)": (7.00, 0.67, C_PPO, "s", 110),
    "Min-Variance": (4.40, 0.90, C_MV, "^", 150),
    "CVaR-Constrained RL": (3.27, 0.88, C_CON, "D", 200),
}

# --- Panel (b): ablation (tighten the budget) + min-variance reference ---------
ABLATION = [  # (label, CVaR99_pct, Sharpe)
    ("budget loose", 4.93, 0.515),
    ("base", 3.01, 0.865),
    ("budget tight", 1.35, 1.272),
]
MV_REF = (1.06, 1.45)  # walk-forward min-variance (Sharpe 1.45, CVaR99 1.06%)


def main() -> None:
    set_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # ---- Panel (a) -----------------------------------------------------------
    for name, (cvar, sharpe, color, marker, size) in STRESS.items():
        axA.scatter(cvar, sharpe, s=size, c=color, marker=marker,
                    edgecolor=_INK, linewidth=0.6, zorder=3, label=name)
    # arrow: unconstrained -> constrained
    axA.annotate("", xy=(3.27, 0.88), xytext=(6.47, 0.63),
                 arrowprops=dict(arrowstyle="->", color=C_CON, lw=1.8,
                                 connectionstyle="arc3,rad=0.18"), zorder=2)
    axA.text(3.05, 0.70,
             "CVaR$_{99}$  $-49$%\nSharpe  $+40$%\nMax-DD  $-43$%\nViolations  $7.2\\!\\to\\!0$",
             fontsize=8.5, color=_INK,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_CON, lw=1.2))
    axA.set_title("(a) Stress window (2020–2022 drawdowns)")
    axA.set_xlabel("CVaR$_{99}$ (%)  $\\leftarrow$ lower tail risk is safer")
    axA.set_ylabel("Annualised Sharpe ratio")
    axA.invert_xaxis()
    axA.legend(loc="best", fontsize=8, frameon=False)

    # ---- Panel (b) -----------------------------------------------------------
    xs = [c for _, c, _ in ABLATION]
    ys = [s for _, _, s in ABLATION]
    axB.plot(xs, ys, color=C_CON, lw=2.0, zorder=2,
             marker="D", markersize=9, markeredgecolor=_INK, markeredgewidth=0.6,
             label="constrained RL (tightening)")
    for label, cvar, sharpe in ABLATION:
        axB.annotate(label, (cvar, sharpe), fontsize=8.5, color=_INK,
                     xytext=(6, -12), textcoords="offset points")
    axB.scatter(MV_REF[0], MV_REF[1], s=150, c=C_MV, marker="^",
                edgecolor=_INK, linewidth=0.6, zorder=3, label="min-variance (classical)")
    axB.annotate("min-variance", MV_REF, fontsize=8.5, color=_INK,
                 xytext=(6, 6), textcoords="offset points")
    axB.text(4.4, 1.18,
             "Tightening the tail budget\nimproves BOTH CVaR$_{99}$\nand Sharpe (monotone)",
             fontsize=8.5, color=_INK,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_MV, lw=1.0))
    axB.set_title("(b) Ablation: tighter constraint $\\to$ better outcomes")
    axB.set_xlabel("CVaR$_{99}$ (%)  $\\leftarrow$ lower tail risk is safer")
    axB.set_ylabel("Annualised Sharpe ratio")
    axB.invert_xaxis()
    axB.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("CVaR constraint as a safety layer for learned allocators",
                 fontsize=13, fontweight="bold", color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
