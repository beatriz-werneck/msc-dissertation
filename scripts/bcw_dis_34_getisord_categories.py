# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_34_getisord_categories — Getis-Ord Gi* hotspots/coldspots per category
# -------------------------------------------------------------------------
"""
Hot-spot / cold-spot analysis of the per-category X-Minute accessibility index,
using the Getis-Ord Gi* statistic (Getis & Ord, 1992).

For each category and threshold (15/20/30 min), the coverage index is aggregated
to H3 res-9 hexagons and Gi* is computed with queen-contiguity weights, including
the focal hexagon itself (the "star" form). Each hexagon is classified by its Gi*
z-score at the 95% confidence level:
    HOTSPOT   z >  1.96  (high accessibility clustering)
    COLDSPOT  z < -1.96  (low accessibility clustering, deficiency)
No multiple-testing (FDR) correction is applied; significance is the standard
95% z-score threshold.

Outputs: cluster maps for 15/20/30 min + a hotspot/coldspot count table.
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
LABELS = {"civic_religion": "Civic & religion", "culture": "Culture", "dining": "Dining",
          "education": "Education", "fitness": "Fitness", "groceries": "Groceries",
          "healthcare": "Healthcare", "transport": "Transport", "retail": "Retail",
          "services": "Services"}


def cov_col(cat, T):
    return f"cov_{cat}" if T == 15 else f"cov_{cat}_{T}"


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


# -------------------------------------------------------------------------
# 1. Aggregate per-category coverage (all thresholds) to H3 res-9
# -------------------------------------------------------------------------
print("Aggregating per-category coverage to H3 res-9 ...", flush=True)
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
w.transform = "B"          # binary weights: the classic Getis-Ord Gi* form
print(f"  hexagons: {len(gdf):,}")

# -------------------------------------------------------------------------
# 2. Getis-Ord Gi* per (threshold, category); classify by 95% z-score
# -------------------------------------------------------------------------
counts = {T: {} for T in THRESHOLDS}
for T in THRESHOLDS:
    for c in cfg.CATEGORIES:
        gi = G_Local(gdf[cov_col(c, T)].to_numpy(float), w, transform="B", star=True, permutations=0)
        z = gi.Zs
        cls = np.where(z > Z95, "Hotspot", np.where(z < -Z95, "Coldspot", "ns")).astype(object)
        gdf[f"gi_{c}_{T}"] = cls
        counts[T][c] = ((cls == "Hotspot").sum(), (cls == "Coldspot").sum())

# -------------------------------------------------------------------------
# 3. Maps for each threshold (same style as the LISA figures)
# -------------------------------------------------------------------------
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
for T in THRESHOLDS:
    fig, axes = plt.subplots(2, 5, figsize=(8.27, 5.83), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.02)
    for ax, c in zip(axes.flat, cfg.CATEGORIES):
        ax.patch.set_alpha(0)
        sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.3, zorder=0)
        urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
        sub = gdf[gdf[f"gi_{c}_{T}"].isin(["Hotspot", "Coldspot"])]
        if len(sub):
            sub.plot(ax=ax, color=sub[f"gi_{c}_{T}"].map(COLOR), edgecolor="none", zorder=2)
        ax.set_title(LABELS[c], fontsize=9); ax.set_axis_off(); ax.margins(0)
    handles = [Patch(facecolor=COLOR["Hotspot"], edgecolor="0.4", label="Hotspot (high accessibility)"),
               Patch(facecolor=COLOR["Coldspot"], edgecolor="0.4", label="Coldspot (deficiency)"),
               Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    out = cfg.FIG_DIR / f"34_getisord_categories_res9_{T}min_A5.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved figure: {out}")

# -------------------------------------------------------------------------
# 4. Hotspot / coldspot count table (CSV + PNG)
# -------------------------------------------------------------------------
rows = [[LABELS[c]] + [counts[T][c][i] for T in THRESHOLDS for i in (0, 1)] for c in cfg.CATEGORIES]
tbl = pd.DataFrame(rows, columns=["Category"] +
                   [f"{T}min {k}" for T in THRESHOLDS for k in ("Hot", "Cold")])
tbl.to_csv(cfg.OUTPUT_DIR / "getisord_counts_thresholds.csv", index=False)
print("\n" + tbl.to_string(index=False))

fig, ax = plt.subplots(figsize=(9.5, 3.8)); ax.axis("off")
collabels = ["Category"] + [f"{T} min\n{k}" for T in THRESHOLDS for k in ("Hotspot", "Coldspot")]
t = ax.table(cellText=tbl.values, colLabels=collabels, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
for (r, cc), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("black"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f0f0")
    if cc == 0 and r > 0:
        cell.set_text_props(ha="left")
ax.set_title("Getis-Ord Gi* hotspot / coldspot counts by category and threshold (95% confidence)",
             fontsize=11, pad=12)
out = cfg.FIG_DIR / "34_getisord_counts_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
print(f"Saved table: {out}")
