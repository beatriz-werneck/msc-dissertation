# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_37_global_moran — Global Moran's I per amenity category
# -------------------------------------------------------------------------
"""
Global Moran's I of the per-category coverage index, to measure how spatially
clustered (vs evenly distributed) each amenity category is across São Paulo.
One value per category: higher I = stronger clustering (similar accessibility
values grouped together); I near 0 = more evenly / randomly distributed.

Same H3 res-9 hexagons and queen-contiguity weights as the Gi* analysis;
significance from 999 random permutations. Computed for 15 / 20 / 30 min.
"""
# %%
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from libpysal.weights import Queen
from esda.moran import Moran

import bcw_dis_00_config as cfg

H3_RES, PERM, SEED = 9, 999, 42
THRESHOLDS = [15, 20, 30]
LABELS = {"civic_religion": "Civic & religion", "culture": "Culture", "dining": "Dining",
          "education": "Education", "fitness": "Fitness", "groceries": "Groceries",
          "healthcare": "Healthcare", "transport": "Transport", "retail": "Retail",
          "services": "Services"}


def cov_col(cat, T):
    return f"cov_{cat}" if T == 15 else f"cov_{cat}_{T}"


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


# 1. Aggregate per-category coverage to H3 res-9
df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
val_cols = [cov_col(c, T) for T in THRESHOLDS for c in cfg.CATEGORIES]
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3")[val_cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)

w = Queen.from_dataframe(gdf, silence_warnings=True)
if w.islands:
    gdf = gdf.drop(index=w.islands).reset_index(drop=True)
    w = Queen.from_dataframe(gdf, silence_warnings=True)
w.transform = "r"                                   # row-standardised, standard for Moran's I
print(f"hexagons: {len(gdf):,}")

# 2. Global Moran's I per category and threshold
np.random.seed(SEED)
rows = []
for c in cfg.CATEGORIES:
    r = {"Category": LABELS[c]}
    for T in THRESHOLDS:
        m = Moran(gdf[cov_col(c, T)].to_numpy(float), w, permutations=PERM)
        r[f"I_{T}"] = round(m.I, 3)
        if T == 15:
            r["z_15"] = round(m.z_sim, 1)
            r["p_15"] = f"{m.p_sim:.3f}"
    rows.append(r)
tab = pd.DataFrame(rows).sort_values("I_15", ascending=False).reset_index(drop=True)
print("\n=== Global Moran's I by category (sorted, most clustered first) ===")
print(tab.to_string(index=False))
tab.to_csv(cfg.OUTPUT_DIR / "global_moran_categories.csv", index=False)

# 3. Bar chart (15 min, sorted)
order = tab["Category"].tolist()
vals = tab["I_15"].tolist()
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(order)))
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.patch.set_alpha(0)
ax.barh(range(len(order)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
ax.invert_yaxis()                                   # most clustered on top
ax.set_xlabel("Global Moran's I (15 min) — higher = more spatially clustered", fontsize=9)
ax.tick_params(axis="x", labelsize=8)
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
ax.grid(axis="x", alpha=0.3)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "37_global_moran_categories.png"
fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
print(f"\nSaved figure: {out}")
