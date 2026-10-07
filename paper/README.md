# Paper

`main.tex` is the camera-ready ICAIF paper. The four figures it includes live in
`paper/figures/`, and all cited numeric results are kept in the committed
`data/processed/` panels and `results/` tables.

Build locally with:

```bash
cd paper
latexmk -pdf main.tex
```
