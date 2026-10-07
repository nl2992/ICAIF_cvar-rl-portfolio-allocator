"""Shared print style for the paper's figures, plus a layout check.

Monochrome with one muted accent for the proposed method; identity is carried
by marker shape and direct labels, never by colour alone. Serif type (STIX,
Times-compatible) sized for the ACM two-column layout. Figures are saved as
vector PDF for the paper and PNG for the repository README.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#222222"
ACCENT = "#2F4B7C"   # muted navy: the proposed method only
DARK = "#4D4D4D"
MID = "#8C8C8C"
LIGHT = "#CFCFCF"
GRID = "#EBEBEB"
# White backing for direct labels so faint gridlines never run through text.
LABEL_BOX = dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none")

COLUMN_WIDTH_IN = 3.33   # ACM sigconf single column
TEXT_WIDTH_IN = 7.0      # ACM sigconf full text width


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem: str, dirs=("paper/figures",)) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(Path(d) / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("wrote", Path(d) / f"{stem}.pdf", "and .png")


def _densify(xy: np.ndarray, step_px: float = 1.5) -> np.ndarray:
    """Points every ``step_px`` pixels along a polyline given in display coordinates."""
    out = []
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.hypot(*(b - a)) / step_px))
        out.append(np.linspace(a, b, n))
    return np.vstack(out) if out else xy


def check_layout(fig, marker_radius_pt: float = 3.5) -> list[str]:
    """Report every layout defect a reader would notice.

    Checks text-text overlaps, text over data markers, text crossed by a data
    line or by another label's leader line or arrow, and text outside the
    figure. Text extents exclude annotation leader lines; tick labels outside
    the axis limits (never drawn) are ignored.
    """
    from matplotlib.text import Annotation, Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bb = fig.bbox
    texts, markers, lines = [], [], []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        ticks = [t for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if x0 <= v <= x1]
        ticks += [t for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if y0 <= v <= y1]
        for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title] + ticks:
            if t.get_visible() and t.get_text().strip():
                texts.append((t, t.get_text().replace("\n", " "), Text.get_window_extent(t, renderer)))
        if ax.get_legend() is not None:
            texts.append((ax.get_legend(), "<legend>", ax.get_legend().get_window_extent(renderer)))
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs):
                markers.extend(ax.transData.transform(offs))
        for line in ax.lines:
            xy = ax.transData.transform(np.column_stack(line.get_data()))
            lines.append((None, _densify(xy)))
            if line.get_marker() not in (None, "None", "", " "):
                markers.extend(xy)
        for t in ax.texts:
            if isinstance(t, Annotation) and t.arrow_patch is not None:
                # Display-coordinate path (get_path() would return data coordinates).
                paths, _ = t.arrow_patch._get_path_in_displaycoord()
                for path in (paths if isinstance(paths, list) else [paths]):
                    for poly in path.to_polygons(closed_only=False):
                        lines.append((t, _densify(np.asarray(poly))))
    for leg in fig.legends:
        texts.append((leg, "<legend>", leg.get_window_extent(renderer)))
    r = marker_radius_pt * fig.dpi / 72.0
    issues = []
    for (_, na, a), (_, nb, b) in combinations(texts, 2):
        if a.overlaps(b):
            issues.append(f"text overlap: '{na}' x '{nb}'")
    for artist, name, bb in texts:
        inner = bb.padded(-0.5)
        for mx, my in markers:
            if bb.x0 - r < mx < bb.x1 + r and bb.y0 - r < my < bb.y1 + r:
                issues.append(f"text over marker: '{name}' at ({mx:.0f},{my:.0f})")
                break
        for owner, pts in lines:
            if owner is artist:
                continue
            if np.any(inner.contains_points(pts) if hasattr(inner, "contains_points") else
                      [(inner.x0 < px < inner.x1 and inner.y0 < py < inner.y1) for px, py in pts]):
                issues.append(f"line or arrow through text: '{name}'")
                break
        if bb.x0 < fig_bb.x0 - 1 or bb.x1 > fig_bb.x1 + 1 or bb.y0 < fig_bb.y0 - 1 or bb.y1 > fig_bb.y1 + 1:
            issues.append(f"text outside figure: '{name}'")
    return issues
