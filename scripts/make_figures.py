"""Generate report figures from results tables and a fresh stress-window backtest.

CSV-based figures (robustness, walk-forward regimes, ablations) are drawn from
existing result tables. Time-series figures (wealth, drawdown, weights, turnover,
rolling CVaR, Lagrange path) train the constrained vs. unconstrained allocator on
the stress split and back-test on the held-out stress window.

Usage:
    python scripts/make_figures.py --config configs/experiment_etf.yaml          # all
    python scripts/make_figures.py --config configs/experiment_etf.yaml --csv-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crlpa.evaluation import plots
from crlpa.evaluation.backtest import rolling_weight_policy, run_policy
from crlpa.evaluation.stress import stress_split
from crlpa.experiment import load_returns, make_env
from crlpa.policies import baselines as b
from crlpa.training.differentiable import DiffConfig, diff_policy, train_differentiable
from crlpa.utils.config import load_config

FIG = Path("reports/figures")


def csv_figures() -> list[Path]:
    made = []
    rob = Path("results/tables_robustness/robustness_metrics.csv")
    if rob.exists():
        df = pd.read_csv(rob)
        made.append(plots.plot_grouped_bars(
            df, "scenario", {"unc_cvar99": "unconstrained", "con_cvar99": "constrained"},
            FIG / "robustness_cvar99.png", "CVaR-99",
            "Robustness: CVaR-99 by scenario",
            colors={"unc_cvar99": "#d62728", "con_cvar99": "#1f77b4"}))
    abl = Path("results/tables_ablations/ablation_metrics.csv")
    if abl.exists():
        df = pd.read_csv(abl)
        made.append(plots.plot_grouped_bars(
            df, "ablation", {"cvar_99": "CVaR-99", "sharpe": "Sharpe"},
            FIG / "ablation_cvar99.png", "value", "Ablations (stress window)"))
    # walk-forward per-regime CVaR-99: constrained vs unconstrained
    con = Path("results/tables_wf/walkforward_regime_rl_cvar_constrained.csv")
    unc = Path("results/tables_wf/walkforward_regime_rl_unconstrained.csv")
    if con.exists() and unc.exists():
        dc = pd.read_csv(con, index_col=0)["cvar_99"].rename("constrained")
        du = pd.read_csv(unc, index_col=0)["cvar_99"].rename("unconstrained")
        df = pd.concat([du, dc], axis=1).reset_index(names="regime")
        made.append(plots.plot_grouped_bars(
            df, "regime", {"unconstrained": "unconstrained", "constrained": "constrained"},
            FIG / "regime_cvar99.png", "CVaR-99", "Walk-forward CVaR-99 by regime",
            colors={"unconstrained": "#d62728", "constrained": "#1f77b4"}))
    return made


def timeseries_figures(cfg, seeds, updates) -> list[Path]:
    returns, _ = load_returns(cfg)
    tr, va, te = stress_split(returns)
    lookback = int(cfg.get_path("environment.lookback", 26))
    cost_bps = float(cfg.get_path("environment.transaction_cost_bps", 5.0))
    alpha = float(cfg.get_path("risk.cvar_alpha", 0.95))
    cvar_window = int(cfg.get_path("risk.cvar_window", 52))
    limit = float(cfg.get_path("risk.cvar_limit", 0.03)) * 0.4
    max_weight = float(cfg.get_path("constraints.max_weight", 0.4))

    def train(constrained, lim):
        actor, hist = train_differentiable(
            tr, cost_bps, alpha, lim, cvar_window, lookback,
            config=DiffConfig(n_updates=updates, horizon=104, objective="return",
                              constrained=constrained, lagrange_lr=5.0, seed=seeds[0]),
            val_returns=va)
        return actor, hist

    con_actor, con_hist = train(True, limit)
    unc_actor, _ = train(False, 0.03)

    res = {
        "rl_cvar_constrained": run_policy(make_env(cfg, te), diff_policy(con_actor, lookback)),
        "rl_unconstrained": run_policy(make_env(cfg, te), diff_policy(unc_actor, lookback)),
        "min_variance": run_policy(make_env(cfg, te),
                                   rolling_weight_policy(lambda h: b.min_variance(h, max_weight=max_weight))),
    }
    rets = {k: v.returns for k, v in res.items()}
    made = [
        plots.plot_wealth(rets, FIG / "wealth_curves.png"),
        plots.plot_drawdown(rets, FIG / "drawdown_curves.png"),
        plots.plot_weights_area(res["rl_cvar_constrained"].weights, FIG / "weights_constrained.png",
                                "Constrained allocator weights (stress window)"),
        plots.plot_weights_area(res["rl_unconstrained"].weights, FIG / "weights_unconstrained.png",
                                "Unconstrained allocator weights (stress window)"),
        plots.plot_series({k: res[k].info["turnover"] for k in ("rl_cvar_constrained", "rl_unconstrained")},
                          FIG / "turnover.png", "turnover", "Turnover over time"),
        plots.plot_series({k: res[k].info["cvar_estimate"] for k in ("rl_cvar_constrained", "rl_unconstrained")},
                          FIG / "rolling_cvar.png", "rolling CVaR-95", "Rolling CVaR estimate"),
        plots.plot_training_path(con_hist, FIG / "lagrange_path.png"),
    ]
    return made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_etf.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7])
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    made = csv_figures()
    if not args.csv_only:
        made += timeseries_figures(load_config(args.config), args.seeds, args.updates)
    for p in made:
        print("wrote", p)
    print(f"\n{len(made)} figures in {FIG}/")


if __name__ == "__main__":
    main()
