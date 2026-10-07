"""Materialise the modelling dataset to parquet.

For a paper configuration this fetches the configured ETF prices and writes a
frozen weekly-return panel. Training can then use ``data.source: parquet`` for
offline, reproducible runs.

Usage:
    python scripts/build_dataset.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crlpa.experiment import load_returns
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    returns, _ = load_returns(cfg)

    out = Path(args.out or cfg.get_path("data.parquet_path"))
    out.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(out)
    print(f"wrote {out} with shape {returns.shape}")


if __name__ == "__main__":
    main()
