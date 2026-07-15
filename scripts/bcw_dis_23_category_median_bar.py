# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_23_category_median_bar — median coverage per category (15 min)
# -------------------------------------------------------------------------
"""
Compact horizontal bar chart of the MEDIAN coverage index (15 min) for each of
the 10 amenity categories, ranked, viridis-coloured, transparent PNG.
Unit = unique origin locations.
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

df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"])

meds = {c: locs[f"cov_{c}"].median() for c in cfg.CATEGORIES}
order = sorted(cfg.CATEGORIES, key=lambda c: meds[c], reverse=True)   # highest first
vals = [meds[c] for c in order]
labels = [LABELS[c] for c in order]
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(order)))

fig, ax = plt.subplots(figsize=(8.27, 3.4))
ax.patch.set_alpha(0)
y = np.arange(len(order))
ax.barh(y, vals, color=colors, edgecolor="0.3", height=0.72)
for yi, v in zip(y, vals):
    ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()                       # highest-median category at the top
ax.set_xlim(0, 1)
ax.tick_params(axis="x", labelsize=8)
ax.set_xlabel("Median coverage index (15 min)", fontsize=9)
ax.set_title("Median coverage index by amenity category — São Paulo (15 min)", fontsize=10)
ax.grid(axis="x", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "23_category_median_bar.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {out}")
