# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_11_category_maps — per-amenity-category accessibility maps (15 min)
# -------------------------------------------------------------------------
"""
Maps the carbon-weighted accessibility SCORE of each of the 10 amenity
categories separately, at the 15-minute threshold, on the H3 grid.

For each origin and category, the score is the highest carbon weight among the
modes (walk/bike/bus/metro/train) that reach an amenity of that category within
15 min (0 if none) — i.e. the per-category term that the index averages. These
scores are recomputed instantly from the cached per-mode travel-time matrices
written by bcw_dis_09_index.py (no re-routing).

Dependencies: h3 (v4), geopandas, shapely, pandas, numpy, matplotlib.
"""
# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

THRESHOLD = 15
H3_RES = 9
MODES = ["walk", "bike", "bus", "metro", "train"]

# -------------------------------------------------------------------------
# 1. Rebuild the unique-origin set EXACTLY as the index run did (deterministic),
#    so orig_id aligns with the travel-time checkpoints.
# -------------------------------------------------------------------------
# %%
trips = pd.read_parquet(cfg.OD_ENRICHED_PARQUET)
origins = (trips[["lat_o", "lon_o"]].dropna().drop_duplicates().reset_index(drop=True))
origins["orig_id"] = origins.index.astype(str)
SCOPE = f"full{len(origins)}"
CKPT_DIR = cfg.OUTPUT_DIR / "tt_checkpoints"
print(f"Origins: {len(origins):,}  | scope: {SCOPE}")

# -------------------------------------------------------------------------
# 2. Load per-mode travel-time checkpoints and score each category.
# -------------------------------------------------------------------------
# %%
def load_mode_tt(mode):
    p = CKPT_DIR / f"{SCOPE}_{mode}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing travel-time checkpoint: {p}\n"
                                "Run bcw_dis_09_index.py (full) first.")
    d = pd.read_parquet(p).set_index("orig_id")
    return d.reindex(index=origins["orig_id"].values, columns=cfg.CATEGORIES)


tt = {m: load_mode_tt(m) for m in MODES}
print("Loaded travel-time checkpoints for:", ", ".join(MODES))

# Per-category score = max carbon weight among modes reaching it within THRESHOLD.
cat_score = pd.DataFrame(index=origins["orig_id"].values)
for cat in cfg.CATEGORIES:
    per_mode = [(tt[m][cat] <= THRESHOLD).astype(float) * cfg.CARBON_WEIGHTS[m]
                for m in MODES]
    cat_score[cat] = pd.concat(per_mode, axis=1).max(axis=1).to_numpy()

# Attach origin coordinates for the spatial join.
cat_score = cat_score.reset_index(names="orig_id").merge(
    origins, on="orig_id", how="left")
print("Mean per-category score (15 min):")
print(cat_score[cfg.CATEGORIES].mean().round(3).to_string())

# -------------------------------------------------------------------------
# 3. Aggregate each category onto H3 (mean score over unique origin locations).
# -------------------------------------------------------------------------
# %%
def hexagon_polygon(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


locs = cat_score.drop_duplicates(subset=["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(lat, lon, H3_RES)
              for lat, lon in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3").agg({c: "mean" for c in cfg.CATEGORIES}).reset_index()
agg["geometry"] = agg["h3"].map(hexagon_polygon)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)
print(f"H3 res {H3_RES}: {len(gdf):,} cells")

# -------------------------------------------------------------------------
# 4. Plot one panel per category (2 x 5), shared 0-1 colour scale.
# -------------------------------------------------------------------------
# %%
sp_boundary = cfg.load_sp_boundary()
fig, axes = plt.subplots(2, 5, figsize=(26, 11))
for ax, cat in zip(axes.flat, cfg.CATEGORIES):
    sp_boundary.boundary.plot(ax=ax, color="black", linewidth=0.4)
    gdf.plot(ax=ax, column=cat, cmap="viridis", vmin=0, vmax=1, legend=False)
    ax.set_title(f"{cat}  (mean {gdf[cat].mean():.2f})", fontsize=11)
    ax.set_axis_off()

# one shared colorbar
sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
fig.colorbar(sm, ax=axes, shrink=0.5, label="category score (carbon-weighted)")
fig.suptitle(f"Per-category 15-min accessibility score — H3 res {H3_RES}",
             y=1.0, fontsize=15)
cfg.save_fig(f"11_category_maps_h3_res{H3_RES}_{THRESHOLD}min", fig)

# Export the per-category hex layer for QGIS.
out = cfg.OUTPUT_DIR / f"category_scores_h3_res{H3_RES}_{THRESHOLD}min.gpkg"
gdf.drop(columns="h3").to_file(out, driver="GPKG")
print(f"saved hex layer: {out}")

# %% [markdown]
# Notes:
# - Score scale: 1.0 = reachable by walk/bike (carbon weight 1.0); ~0.13/0.12 =
#   reachable only by metro/train; ~0.007 = only by bus; 0 = not reachable in 15
#   min by any mode. So darker (low) cells = a category only reachable by higher-
#   carbon modes (or not at all).
# - culture is the sparsest category (lowest mean); transport the densest — useful
#   context for the saturation discussion.
