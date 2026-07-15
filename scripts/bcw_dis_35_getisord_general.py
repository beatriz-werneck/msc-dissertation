# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_35_getisord_general — Getis-Ord Gi* on the OVERALL index (15/20/30)
# -------------------------------------------------------------------------
"""
Getis-Ord Gi* hot-spot / cold-spot analysis of the general (all-category)
X-Minute City Accessibility Index, at the three thresholds. Same method as
bcw_dis_34: overall coverage index aggregated to H3 res-9, Gi* with binary
queen-contiguity weights (star form), classified at 95% confidence
(z > 1.96 hotspot, z < -1.96 coldspot). No FDR correction.
"""
# %%
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon
from libpysal.weights import Queen
from esda.getisord import G_Local

import bcw_dis_00_config as cfg

H3_RES, Z95 = 9, 1.96
THRESHOLDS = [15, 20, 30]
URBAN_GREY = "0.80"
COLOR = {"Hotspot": plt.cm.viridis(1.0), "Coldspot": plt.cm.viridis(0.10)}


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


# 1. Aggregate the overall index to H3 res-9
print("Aggregating overall index to H3 res-9 ...", flush=True)
df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
val_cols = [f"index_coverage_{T}" for T in THRESHOLDS]
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3")[val_cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)

w = Queen.from_dataframe(gdf, silence_warnings=True)
if w.islands:
    gdf = gdf.drop(index=w.islands).reset_index(drop=True)
    w = Queen.from_dataframe(gdf, silence_warnings=True)
w.transform = "B"
print(f"  hexagons: {len(gdf):,}")

# 2. Gi* per threshold
counts = {}
for T in THRESHOLDS:
    gi = G_Local(gdf[f"index_coverage_{T}"].to_numpy(float), w, transform="B", star=True, permutations=0)
    z = gi.Zs
    gdf[f"gi_{T}"] = np.where(z > Z95, "Hotspot", np.where(z < -Z95, "Coldspot", "ns")).astype(object)
    counts[T] = (int((gdf[f"gi_{T}"] == "Hotspot").sum()), int((gdf[f"gi_{T}"] == "Coldspot").sum()))
    print(f"  {T} min: hotspots={counts[T][0]:,}  coldspots={counts[T][1]:,}")

# 3. Three-panel map (15 / 20 / 30) — same layout/size as figure 16b
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()
fig, axes = plt.subplots(1, 3, figsize=(8.27, 5.83), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.0,
                            rect=(0, 0.10, 1, 1.0))
for ax, T in zip(axes, THRESHOLDS):
    ax.patch.set_alpha(0)
    sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.4, zorder=0)
    urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
    sub = gdf[gdf[f"gi_{T}"].isin(["Hotspot", "Coldspot"])]
    if len(sub):
        sub.plot(ax=ax, color=sub[f"gi_{T}"].map(COLOR), edgecolor="none", zorder=2)
    ax.set_title(f"{T} minutes", fontsize=11)
    ax.set_axis_off(); ax.margins(0)
handles = [Patch(facecolor=COLOR["Hotspot"], edgecolor="0.4", label="Hotspot (high accessibility)"),
           Patch(facecolor=COLOR["Coldspot"], edgecolor="0.4", label="Coldspot (deficiency)"),
           Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.01),
           frameon=False, fontsize=9)
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "35_getisord_general_res9_A5.png"
fig.savefig(out, dpi=300, transparent=True)
print(f"Saved figure: {out}")
