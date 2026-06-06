"""Materialise the modelling dataset to parquet.

For the synthetic source this writes a regime-switching panel; the same entry
point will load and align real price/factor/macro data once those loaders are
wired in (see src/crlpa/data/). Writing parquet lets training switch to
``data.source: parquet`` for reproducible runs off a frozen dataset.

Usage:
    python scripts/build_dataset.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crlpa.experiment import load_returns
from crlpa.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out", default="data/processed/aligned_portfolio_panel.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    returns, regimes = load_returns(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(out)
    print(f"wrote {out} with shape {returns.shape}")
    if regimes is not None:
        regime_path = out.with_name("regime_labels.parquet")
        regimes.to_frame().to_parquet(regime_path)
        print(f"wrote {regime_path} with regime counts {regimes.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
