# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_15_rich_maps — maps of the sub-type-resolved (coverage) index
# -------------------------------------------------------------------------
"""
H3 maps for the richer index (bcw_dis_14_index_rich -> pmc_index_rich.parquet):

  A. Overall: coverage / binary / count side by side, at 15/20/30 min (res 9).
  B. Per-category COVERAGE at 15 min (10 panels).
  C. COVERAGE by economic class at 15 min (6 panels) — all anchor trips of each
     class (not residence-only).

Aggregated over UNIQUE origin locations (no trip-frequency bias). Coverage and
count are on a 0-0.85 scale (their real range); the overall panel uses 0-1 so
the binary baseline's saturation is visible next to the de-masked coverage.

Dependencies: h3 (v4), geopandas, shapely, pandas, matplotlib.
"""
# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

H3_RES = 9
COV_VMAX = 0.85          # colour ceiling for coverage-only maps (data max ~0.84)
CLASS_ORDER = ["A", "B1", "B2", "C1", "C2", "D-E"]

df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()       # grey base: SP urban zone (Plano Diretor 2014)
print(f"Loaded {len(df):,} rows from pmc_index_rich.parquet")


def basemap(ax, lw=0.5):
    """Grey urban-zone fill + municipality outline, under the hex layer.
    Gaps that stay WHITE inside the boundary are non-urban (parks, reservoirs,
    rural fringe) -> they explain where there are no trip anchors."""
    urban.plot(ax=ax, color="0.85", edgecolor="none", zorder=0)
    sp.boundary.plot(ax=ax, color="black", linewidth=lw, zorder=5)


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def aggregate(frame, cols):
    """Mean of `cols` per H3 cell over unique origin locations -> GeoDataFrame."""
    locs = frame.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
    locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES)
                  for la, lo in zip(locs["lat_o"], locs["lon_o"])]
    agg = locs.groupby("h3")[cols].mean().reset_index()
    agg["geometry"] = agg["h3"].map(hexpoly)
    return gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)


def shared_colorbar(fig, axes, vmax, label):
    sm = ScalarMappable(norm=Normalize(0, vmax), cmap="viridis")
    fig.colorbar(sm, ax=axes, shrink=0.5, label=label)


# -------------------------------------------------------------------------
# A. Overall: coverage / binary / count at each threshold
# -------------------------------------------------------------------------
# %%
variants = {"Coverage (carbon-weighted diversity)": "index_coverage_{t}",
            "Binary (baseline)": "index_binary_{t}",
            "Count (intensity)": "index_count_{t}"}
gdf_all = aggregate(df, [c.format(t=t) for c in variants.values() for t in cfg.THRESHOLDS])

for t in cfg.THRESHOLDS:
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, (label, col) in zip(axes, variants.items()):
        basemap(ax)
        gdf_all.plot(ax=ax, column=col.format(t=t), cmap="viridis", vmin=0, vmax=1,
                     legend=True, legend_kwds={"shrink": 0.5, "label": "index"})
        ax.set_title(f"{label} ({t} min)\nmean {gdf_all[col.format(t=t)].mean():.2f}")
        ax.set_axis_off()
    fig.suptitle(f"Rich 15-Minute City index — H3 res {H3_RES} — {t} min", y=1.02, fontsize=14)
    cfg.save_fig(f"15_rich_overall_res{H3_RES}_{t}min", fig)

gdf_all.drop(columns="h3").to_file(
    cfg.OUTPUT_DIR / f"rich_index_h3_res{H3_RES}.gpkg", driver="GPKG")


# -------------------------------------------------------------------------
# B. Per-category coverage (15 min)
# -------------------------------------------------------------------------
# %%
cov_cols = [f"cov_{c}" for c in cfg.CATEGORIES]
gdf_cat = aggregate(df, cov_cols)
fig, axes = plt.subplots(2, 5, figsize=(26, 11))
for ax, cat in zip(axes.flat, cfg.CATEGORIES):
    basemap(ax, lw=0.4)
    gdf_cat.plot(ax=ax, column=f"cov_{cat}", cmap="viridis", vmin=0, vmax=COV_VMAX, legend=False)
    ax.set_title(f"{cat}  (mean {gdf_cat[f'cov_{cat}'].mean():.2f})", fontsize=11)
    ax.set_axis_off()
shared_colorbar(fig, axes, COV_VMAX, "category coverage (carbon-weighted)")
fig.suptitle(f"Per-category coverage, 15 min — H3 res {H3_RES}", y=1.0, fontsize=15)
cfg.save_fig(f"15_rich_category_coverage_res{H3_RES}", fig)
gdf_cat.drop(columns="h3").to_file(
    cfg.OUTPUT_DIR / f"rich_category_coverage_h3_res{H3_RES}.gpkg", driver="GPKG")


# -------------------------------------------------------------------------
# C. Coverage by economic class (15 min) — all anchor trips of each class
# -------------------------------------------------------------------------
# %%
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
eq_parts = []
for ax, cls in zip(axes.flat, CLASS_ORDER):
    sub = df[df["classe_economica"] == cls]
    g = aggregate(sub, ["index_coverage_15"])
    basemap(ax, lw=0.4)
    if len(g):
        g.plot(ax=ax, column="index_coverage_15", cmap="viridis", vmin=0, vmax=COV_VMAX, legend=False)
        g["classe"] = cls; eq_parts.append(g)
    wm = (sub["index_coverage_15"] * sub["fe_via"]).sum() / sub["fe_via"].sum()
    ax.set_title(f"Class {cls}  (mean {wm:.2f}, n={len(sub):,})", fontsize=12)
    ax.set_axis_off()
shared_colorbar(fig, axes, COV_VMAX, "coverage index (15 min)")
fig.suptitle(f"Coverage index by economic class, 15 min — H3 res {H3_RES} "
             "(all anchor trips of each class)", y=1.0, fontsize=15)
cfg.save_fig(f"15_rich_equity_coverage_res{H3_RES}", fig)
pd.concat(eq_parts, ignore_index=True).drop(columns="h3").to_file(
    cfg.OUTPUT_DIR / f"rich_equity_coverage_h3_res{H3_RES}.gpkg", driver="GPKG")

print("Done — saved overall (15/20/30), per-category, and per-class coverage maps.")
