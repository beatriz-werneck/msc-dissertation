# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_24_mode_table — "Contribution of transport modes" table (PNG)
# -------------------------------------------------------------------------
"""
Renders the mode-contribution table (walk / walk+bike / walk+bike+public
transport, by mean/median/25th/75th, at 15/20/30 min) as a styled PNG matching
the black-header format.

Model is selected via env:
  CARBON_WEIGHTED=1 (default) -> carbon-weighted index
  CARBON_WEIGHTED=0           -> mode-agnostic (all weights 1.0); output _noweight
"""
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import bcw_dis_00_config as cfg

CARBON_WEIGHTED = os.environ.get("CARBON_WEIGHTED", "1") != "0"
SUFFIX = "" if CARBON_WEIGHTED else "_noweight"
CK = cfg.OUTPUT_DIR / "tt_checkpoints"
MODES = ["walk", "bike", "bus", "metro", "train"]
W = cfg.CARBON_WEIGHTS if CARBON_WEIGHTED else {m: 1.0 for m in MODES}

rich = {m: pd.read_parquet(CK / f"rich_full29552_{m}.parquet").set_index("orig_id") for m in MODES}
keys = [c[:-5] for c in rich["walk"].columns if c.endswith("__r15")]
cats = {}
[cats.setdefault(k.split("::")[0], []).append(k) for k in keys]


def coverage_series(subset, T):
    catcov = []
    for cat, cklist in cats.items():
        subs = []
        for ck in cklist:
            score = None
            for m in subset:
                s = (rich[m][f"{ck}__r{T}"] > 0).astype(float) * W[m]
                score = s if score is None else np.maximum(score, s)
            subs.append(score)
        catcov.append(pd.concat(subs, axis=1).mean(axis=1))
    return pd.concat(catcov, axis=1).mean(axis=1)


MODE_LABELS = ["Walk", "Walk + Bike", "Walk + Bike + Public Transport"]
SUBSETS = [["walk"], ["walk", "bike"], MODES]
rows = []   # (threshold or None, mode label, mean, median, q25, q75)
for T in [15, 20, 30]:
    for mi, (mlbl, sub) in enumerate(zip(MODE_LABELS, SUBSETS)):
        s = coverage_series(sub, T)
        rows.append((T if mi == 1 else None, mlbl,
                     s.mean(), s.median(), s.quantile(.25), s.quantile(.75)))

# -------------------------------------------------------------------------
# Draw the table
# -------------------------------------------------------------------------
cols = ["Threshold", "Transport mode", "Mean", "Median", "25th\nQuartile", "75th\nQuartile"]
widths = [0.16, 0.31, 0.1325, 0.1325, 0.1325, 0.1325]
xe = np.cumsum([0] + widths)
BLACK, G1, G2 = "black", "#e8e8e8", "#c9c9c9"

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(-0.07, 1); ax.axis("off")
title_h, head_h = 0.11, 0.11
top = 1 - title_h - head_h
row_h = top / len(rows)


def cell(x, w, y, h, color, text="", tc="black", fs=9, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=1.4))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold" if bold else "normal")


# header
cell(xe[2], 1 - xe[2], 1 - title_h, title_h, BLACK,
     "X-Minute City Index (coverage)", "white", 12, True)
cell(xe[0], widths[0], 1 - title_h - head_h, title_h + head_h, BLACK, cols[0], "white", 10, True)
cell(xe[1], widths[1], 1 - title_h - head_h, title_h + head_h, BLACK, cols[1], "white", 10, True)
for j in [2, 3, 4, 5]:
    cell(xe[j], widths[j], 1 - title_h - head_h, head_h, BLACK, cols[j], "white", 8.5, True)

# data
for i, (tl, ml, me, md, q1, q3) in enumerate(rows):
    y = top - (i + 1) * row_h
    bg = G1 if (i // 3) % 2 == 0 else G2
    cell(xe[0], widths[0], y, row_h, bg, f"{tl}-Minutes" if tl else "", "black", 10, True)
    cell(xe[1], widths[1], y, row_h, bg, ml, "black", 9)
    for k, v in enumerate([me, md, q1, q3]):
        cell(xe[2 + k], widths[2 + k], y, row_h, bg, f"{v:.3f}", "black", 9)

caption = "Mode-agnostic index — no carbon weighting" if not CARBON_WEIGHTED else "Carbon-weighted index"
ax.text(0.5, -0.035, caption, ha="center", va="center", fontsize=9, style="italic", color="0.25")

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / f"24_mode_contribution_table{SUFFIX}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {out}")
