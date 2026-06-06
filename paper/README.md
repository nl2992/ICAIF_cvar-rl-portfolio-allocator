# Paper backbone — ICAIF-style write-up

`main.tex` is an [`acmart`](https://www.acm.org/publications/proceedings-template)
`sigconf` skeleton mirroring the structure of recent ICAIF submissions
(Abstract → Introduction with explicit contributions → Background/Related Work →
Method → Experimental Setup → Results → Limitations → Conclusion/Future Work →
References).

The paper is populated with the **real results** from the
`feat/cvar-allocator-pipeline` branch (seven-ETF macro study: differentiable
CVaR-constrained allocator, stress window, walk-forward, robustness, ablations).
Figures used by the paper are copied into `paper/figures/`; `\graphicspath` also
points at `../reports/figures`. A few `\TODO{...}` markers remain where results
are genuinely still incomplete (e.g. the model-free PPO/SAC sweep, and the extra
walk-forward folds / second universe needed to push the marginal significance
below 0.05).

## Build

Easiest on [Overleaf](https://overleaf.com): upload the `paper/` folder and
compile `main.tex`. Locally:

```bash
cd paper
latexmk -pdf main.tex      # requires a TeX distribution with the acmart class
```

Search for `\TODO` to find everything still to be filled in. Swap the placeholder
author block and drop the `nonacm` class option for the camera-ready.
