"""Render the evaluation-protocol rank-reversal slopegraph (the paper's key visual).

Shows that on the 31-asset universe the *same* strategies rank oppositely under a
single stress split vs. a rolling walk-forward: the learned allocator looks best on
the stress split but minimum variance dominates out-of-sample (the ranking inverts).
Sharpe values are the recorded results in paper Table~\\ref{tab:protocol} /
reports/research_log.md (results/tables_wf_large/, regenerable from the pipeline).

Usage:
    python scripts/make_reversal_figure.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crlpa.evaluation import plots

# Authoritative Sharpe ratios (paper tab:protocol; 31-asset universe, 2008-2024).
PROTOCOL_SHARPE = pd.DataFrame(
    [
        {"strategy": "minimum variance", "stress": 0.23, "walkforward": 1.40},
        {"strategy": "inverse vol", "stress": 0.38, "walkforward": 0.78},
        {"strategy": "diff. constrained (ours)", "stress": 0.85, "walkforward": 0.74},
        {"strategy": "diff. unconstrained", "stress": 0.70, "walkforward": 0.56},
    ]
)

HIGHLIGHT = {
    "minimum variance": "#2ca02c",          # rises to #1 OOS
    "diff. constrained (ours)": "#1f77b4",   # falls from #1 on the stress split
}


def main() -> None:
    data_dir = Path("results/tables_protocol")
    data_dir.mkdir(parents=True, exist_ok=True)
    PROTOCOL_SHARPE.to_csv(data_dir / "protocol_sharpe.csv", index=False)

    for fig_dir in (Path("reports/figures"), Path("paper/figures")):
        out = plots.plot_rank_reversal(
            PROTOCOL_SHARPE, fig_dir / "eval_protocol_reversal.png", highlight=HIGHLIGHT,
            title=("Same data, opposite verdict (31 assets):\n"
                   "the learner ranks 1st on a single split but min-variance dominates OOS"),
        )
        print("wrote", out)


if __name__ == "__main__":
    main()
