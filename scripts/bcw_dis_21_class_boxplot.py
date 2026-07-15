# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_21_class_boxplot — coverage index distribution by economic class
# -------------------------------------------------------------------------
"""
Horizontal boxplot of the per-origin coverage index (15 min) for each Critério
Brasil economic class (A ... D-E), styled like the per-category boxplot: viridis
box fills, transparent PNG, compact A5-width strip.

Unit = each class's unique origin locations (deduplicated within class).
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bcw_dis_00_config as cfg

INDEX_COL = "index_coverage_15"
CLASS_ORDER = ["A", "B1", "B2", "C1", "C2", "D-E"]   # A at top -> D-E at bottom

import os
SUFFIX = os.environ.get("MODEL_SUFFIX", "")   # "" weighted; "_noweight" for mode-agnostic
df = pd.read_parquet(cfg.OUTPUT_DIR / f"pmc_index_rich{SUFFIX}.parquet")

vals = []
for cls in CLASS_ORDER:
    sub = (df[df["classe_economica"] == cls]
           .dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]))
    vals.append(sub[INDEX_COL].dropna().values)
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(CLASS_ORDER)))

fig, ax = plt.subplots(figsize=(8.27, 3.0))     # A5 width, compact height
ax.patch.set_alpha(0)
bp = ax.boxplot(vals, vert=False, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="black", linewidth=1.1),
                whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"),
                boxprops=dict(edgecolor="0.3"))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.92)

ax.set_yticklabels(CLASS_ORDER, fontsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.invert_yaxis()                       # A at the top
ax.set_xlim(0, 1)
ax.set_xlabel("Coverage index (15 min)", fontsize=9)
ax.set_ylabel("Economic class", fontsize=9)
ax.grid(axis="x", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / f"21_class_coverage_boxplot{SUFFIX}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {out}")
