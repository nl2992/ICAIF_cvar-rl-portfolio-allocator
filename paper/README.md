# Paper backbone — ICAIF-style write-up

`main.tex` is an [`acmart`](https://www.acm.org/publications/proceedings-template)
`sigconf` skeleton mirroring the structure of recent ICAIF submissions
(Abstract → Introduction with explicit contributions → Background/Related Work →
Method → Experimental Setup → Results → Limitations → Conclusion/Future Work →
References).

**This is a backbone, pending results.** The method, environment and committed
configuration (drawn from `configs/experiment.yaml`) are written out in full; the
result tables are placeholders marked with the `\TODO{...}` macro, to be filled
from `results/` once the trained-agent runs complete.

## Build

Easiest on [Overleaf](https://overleaf.com): upload the `paper/` folder and
compile `main.tex`. Locally:

```bash
cd paper
latexmk -pdf main.tex      # requires a TeX distribution with the acmart class
```

Search for `\TODO` to find everything still to be filled in. Swap the placeholder
author block and drop the `nonacm` class option for the camera-ready.
