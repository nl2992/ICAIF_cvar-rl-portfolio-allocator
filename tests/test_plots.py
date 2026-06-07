from __future__ import annotations

import numpy as np
import pandas as pd

from crlpa.evaluation import plots


def test_plot_wealth_and_drawdown_write_files(tmp_path):
    rng = np.random.default_rng(0)
    rets = {
        "rl_cvar_constrained": pd.Series(rng.normal(0.001, 0.02, 60)),
        "rl_unconstrained": pd.Series(rng.normal(0.001, 0.03, 60)),
    }
    p1 = plots.plot_wealth(rets, tmp_path / "wealth.png")
    p2 = plots.plot_drawdown(rets, tmp_path / "dd.png")
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0


def test_plot_weights_area_writes_file(tmp_path):
    w = pd.DataFrame(np.full((40, 3), 1 / 3), columns=["a", "b", "c"])
    p = plots.plot_weights_area(w, tmp_path / "w.png")
    assert p.exists() and p.stat().st_size > 0


def test_plot_grouped_bars_writes_file(tmp_path):
    df = pd.DataFrame({"scenario": ["x", "y"], "unc": [0.06, 0.05], "con": [0.03, 0.02]})
    p = plots.plot_grouped_bars(df, "scenario", {"unc": "unconstrained", "con": "constrained"},
                                tmp_path / "bars.png", "CVaR-99", "test")
    assert p.exists() and p.stat().st_size > 0


def test_plot_rank_reversal_writes_file(tmp_path):
    import pandas as pd

    from crlpa.evaluation import plots
    df = pd.DataFrame([
        {"strategy": "min variance", "stress": 0.23, "walkforward": 1.40},
        {"strategy": "ours", "stress": 0.85, "walkforward": 0.74},
    ])
    p = plots.plot_rank_reversal(df, tmp_path / "rev.png", highlight={"min variance": "#2ca02c"})
    assert p.exists() and p.stat().st_size > 0


def test_plot_dumbbell_writes_file(tmp_path):
    df = pd.DataFrame([
        {"universe": "macro", "unconstrained": 0.0142, "constrained": 0.0114},
        {"universe": "sector", "unconstrained": 0.0336, "constrained": 0.0251},
    ])
    p = plots.plot_dumbbell(df, tmp_path / "db.png")
    assert p.exists() and p.stat().st_size > 0


def test_set_style_is_idempotent():
    plots.set_style(); plots.set_style()  # should not raise
