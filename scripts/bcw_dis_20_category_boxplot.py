# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_20_category_boxplot — distribution of coverage per category
# -------------------------------------------------------------------------
"""
Horizontal boxplot of the per-origin coverage index (15 min) for each of the 10
amenity categories, ordered by median, each box filled with a distinct viridis
colour. Transparent-background PNG.

Unit = unique origin locations (deduplicated), so frequent-origin locations do
not dominate the distribution.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bcw_dis_00_config as cfg

LABELS = {"civic_religion": "Civic & religion", "culture": "Culture", "dining": "Dining",
          "education": "Education", "fitness": "Fitness", "groceries": "Groceries",
          "healthcare": "Healthcare", "transport": "Transport", "retail": "Retail",
          "services": "Services"}

import os
SUFFIX = os.environ.get("MODEL_SUFFIX", "")   # "" weighted; "_noweight" for mode-agnostic
df = pd.read_parquet(cfg.OUTPUT_DIR / f"pmc_index_rich{SUFFIX}.parquet")
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"])

data = {c: locs[f"cov_{c}"].dropna().values for c in cfg.CATEGORIES}
order = sorted(cfg.CATEGORIES, key=lambda c: np.median(data[c]), reverse=True)  # highest median first
vals = [data[c] for c in order]
labels = [LABELS[c] for c in order]
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(order)))

fig, ax = plt.subplots(figsize=(8.27, 3.7))     # A5 width, compact (short) height
ax.patch.set_alpha(0)
bp = ax.boxplot(vals, vert=False, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="black", linewidth=1.1),
                whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"),
                boxprops=dict(edgecolor="0.3"))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.92)

ax.set_yticklabels(labels, fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.invert_yaxis()                       # highest-median category at the top
ax.set_xlim(0, 1)
ax.set_xlabel("Coverage index (15 min)", fontsize=9)
ax.grid(axis="x", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / f"20_category_coverage_boxplot{SUFFIX}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {out}")
