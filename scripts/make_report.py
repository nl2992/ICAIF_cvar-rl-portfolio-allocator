"""Assemble the final markdown report from generated tables.

Usage:
    python scripts/make_report.py --experiment_id cvar_ac_v1
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


def _to_markdown(df: pd.DataFrame) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _table(path: Path, columns: list[str] | None = None) -> str:
    if not path.exists():
        return f"_missing: {path}_\n"
    df = pd.read_csv(path, index_col=0)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    return _to_markdown(df.round(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="cvar_ac_v1")
    parser.add_argument("--tables", default="results/tables")
    parser.add_argument("--out", default="reports/final_report.md")
    args = parser.parse_args()

    tables = Path(args.tables)
    perf_cols = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "cvar_95", "cvar_99",
                 "cvar_breach_rate", "avg_turnover", "final_wealth"]

    sections = [
        f"# Final Report — {args.experiment_id}",
        f"_Generated {date.today().isoformat()}._\n",
        "## Research question",
        "> Can a constrained actor-critic allocator reduce tail risk and constraint "
        "breaches versus unconstrained RL while remaining competitive with standard "
        "portfolio optimisers after costs?\n",
        "## Allocator vs. baselines (test split)",
        _table(tables / "allocator_metrics.csv", perf_cols),
        "## Statistical comparison (paired block bootstrap, Sharpe)",
        _table(tables / "statistical_tests.csv"),
        "## Deterministic baselines (full universe)",
        _table(tables / "baseline_metrics.csv", perf_cols),
        "## Notes",
        "- Metrics are averaged across training seeds for the RL variants.",
        "- CVaR breach rate is the fraction of weeks the rolling CVaR estimate "
        "exceeds the configured limit.",
        "- See `results/training_curves/` for per-episode Lagrange-multiplier and "
        "breach-rate paths.",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
