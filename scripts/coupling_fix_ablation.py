"""GATE 1 — Coupling Fix Ablation.

The Lagrangian scale-mismatch bug: when lagrange_lr is not calibrated to the ratio
of objective scale (Sharpe, ~O(1)) to constraint violation scale (CVaR in weekly return
units, ~O(0.01-0.03)), the multiplier never grows large enough to make the CVaR penalty
meaningful, effectively silencing the constraint.

Three variants, 5 seeds each, on the ETF universe:
  A) Buggy        : lagrange_lr=0.001  → lam stays near 0 → constraint silenced
  B) Fixed        : lagrange_lr=5.0    → lam grows to enforce constraint
  C) Unconstrained: constrained=False  → no penalty at all

Key finding: Buggy ≈ Unconstrained on breach_rate; Fixed << both.

Usage:
    cd /path/to/cvar-rl-portfolio-allocator
    python scripts/coupling_fix_ablation.py --config configs/experiment_etf.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crlpa.evaluation.backtest import run_policy
from crlpa.evaluation.metrics import cvar as compute_cvar
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns, make_env
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

SEEDS = [7, 13, 23, 42, 2025]
N_UPDATES = 1500

# The three variants: (constrained, lagrange_lr, label)
VARIANTS = [
    ("buggy",         True,  0.001),   # lam grows ~0→0.03 after 1500 steps — silenced
    ("fixed",         True,  5.0),     # lam can reach cap (100) — properly enforced
    ("unconstrained", False, 0.0),     # baseline: no constraint at all
]


def _rolling_breach_rate(returns: np.ndarray, limit: float, alpha: float, window: int) -> float:
    """Fraction of test weeks where rolling CVaR(alpha) > limit."""
    violations = 0
    total = 0
    for t in range(window, len(returns)):
        w_ret = returns[max(0, t - window): t]
        if len(w_ret) >= 5:
            cvar_val = compute_cvar(w_ret, alpha)
            violations += int(cvar_val > limit + 1e-6)
            total += 1
    return violations / max(1, total)


def main(config_path: str = "configs/experiment_etf.yaml",
         out_dir: str = "results/tables_ablations") -> None:
    cfg = load_config(config_path)
    returns, _ = load_returns(cfg)

    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    # Use the same tighter budget the existing ablations use
    limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))

    tr, va, te = stress_split(returns)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    breach_rows = []

    for variant_name, constrained, lagrange_lr in VARIANTS:
        print(f"\n--- Variant: {variant_name} (constrained={constrained}, lr={lagrange_lr}) ---")
        for seed in SEEDS:
            print(f"  seed={seed}…", flush=True)

            diff_cfg = DiffConfig(
                n_updates=N_UPDATES,
                horizon=104,
                objective="return",
                anchor="inverse_vol",
                constrained=constrained,
                lagrange_lr=lagrange_lr,
                seed=seed,
            )
            actor, history = train_differentiable(
                tr, cost_bps, alpha, limit, cvar_window, lookback,
                config=diff_cfg, val_returns=va,
            )

            env = make_env(cfg, te)
            env.cvar_alpha = alpha
            result = run_policy(env, diff_policy(actor, lookback, "inverse_vol"))
            m = result.metrics

            test_returns = result.returns.to_numpy()
            breach = _rolling_breach_rate(test_returns, limit, alpha, cvar_window)
            final_lam = float(history["lagrange"].iloc[-1]) if "lagrange" in history.columns else 0.0

            rows.append({
                "variant": variant_name,
                "constrained": constrained,
                "lagrange_lr": lagrange_lr,
                "seed": seed,
                "sharpe": m.get("sharpe", np.nan),
                "cvar_95": m.get("cvar_95", np.nan),
                "cvar_99": m.get("cvar_99", np.nan),
                "max_drawdown": m.get("max_drawdown", np.nan),
                "breach_rate": breach,
                "final_lam": final_lam,
                "avg_turnover": m.get("avg_turnover", np.nan),
            })

            # Weekly breach timeseries for plotting
            for t, r in enumerate(test_returns):
                if t >= cvar_window:
                    w_ret = test_returns[max(0, t - cvar_window): t]
                    cvar_val = compute_cvar(w_ret, alpha) if len(w_ret) >= 5 else 0.0
                    breach_rows.append({
                        "variant": variant_name, "seed": seed, "week": t,
                        "rolling_cvar": cvar_val,
                        "breach": int(cvar_val > limit + 1e-6),
                        "cvar_limit": limit,
                    })

            print(f"    sharpe={m.get('sharpe', 0):.3f}  "
                  f"cvar_99={m.get('cvar_99', 0):.4f}  "
                  f"breach_rate={breach:.3f}  final_lam={final_lam:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "coupling_fix_ablation.csv", index=False)
    pd.DataFrame(breach_rows).to_csv(out / "coupling_fix_breach_timeseries.csv", index=False)

    # Summary table
    summary = df.groupby("variant").agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        cvar_99_mean=("cvar_99", "mean"),
        breach_rate_mean=("breach_rate", "mean"),
        breach_rate_std=("breach_rate", "std"),
        final_lam_mean=("final_lam", "mean"),
    ).reindex(["buggy", "fixed", "unconstrained"])
    summary.to_csv(out / "coupling_fix_summary.csv")

    print("\n=== COUPLING FIX ABLATION — SUMMARY ===")
    print(f"CVaR limit used: {limit:.4f}  (alpha={alpha})")
    print()
    for vname, vrow in summary.iterrows():
        print(f"  {vname:15s}:  "
              f"breach={vrow['breach_rate_mean']:.3f}±{vrow['breach_rate_std']:.3f}  "
              f"sharpe={vrow['sharpe_mean']:.3f}  "
              f"cvar99={vrow['cvar_99_mean']:.4f}  "
              f"final_lam={vrow['final_lam_mean']:.1f}")

    # Key finding check
    buggy_br = summary.loc["buggy", "breach_rate_mean"]
    fixed_br = summary.loc["fixed", "breach_rate_mean"]
    unc_br = summary.loc["unconstrained", "breach_rate_mean"]
    delta_fix = buggy_br - fixed_br
    delta_unc = abs(buggy_br - unc_br)
    print(f"\nKey result: buggy vs fixed breach-rate delta = {delta_fix:.3f}")
    print(f"Key result: buggy vs unconstrained breach-rate delta = {delta_unc:.3f} (should be small)")
    if delta_fix > 0.05 and delta_unc < 0.05:
        print("✓ GATE 1 PASSES: buggy ≈ unconstrained << fixed")
    else:
        print("✗ Gate 1: unexpected result — inspect individual runs")

    print(f"\nWrote: {out}/coupling_fix_ablation.csv")
    print(f"       {out}/coupling_fix_breach_timeseries.csv")
    print(f"       {out}/coupling_fix_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--out", default="results/tables_ablations")
    args = parser.parse_args()
    main(args.config, args.out)
