# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_12_equity_maps — index mapped by socio-economic class
# -------------------------------------------------------------------------
"""
Maps the 15-Minute City index separately for each Critério Brasil economic
class (A, B1, B2, C1, C2, D-E) on the H3 grid.

Each panel is built from ALL the anchor-origin trips made by that class — i.e.
every trip origin (residence, work, study and chained stops; all anchor
purposes) belonging to people of that class — NOT just their home location. The
index at each of those origin locations is aggregated (mean) onto the hex grid,
so a panel shows where that class starts trips and how accessible those places
are. Comparing panels reveals the spatial equity gradient.

Set EXCLUDE_RESIDENCE = True to drop residence origins (motivo_o == 8) and map
only non-home anchors (work/study/other).

Reads the per-trip output of bcw_dis_09_index.py (pmc_index.parquet).
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

# --- Settings ---
H3_RES = 9
INDEX_COL = "index_15"            # multimodal carbon-weighted index at 15 min
CLASS_ORDER = ["A", "B1", "B2", "C1", "C2", "D-E"]
MIN_ORIGINS_PER_HEX = 1
EXCLUDE_RESIDENCE = False         # True -> drop residence anchors (motivo_o == 8)

df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index.parquet")
if EXCLUDE_RESIDENCE and "motivo_o" in df.columns:
    df = df[df["motivo_o"] != 8].copy()
    print("Excluding residence anchors (motivo_o == 8).")
print(f"Loaded {len(df):,} anchor-trip rows; classes present: "
      f"{sorted(df['classe_economica'].dropna().unique())}")


def hexagon_polygon(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def aggregate_class(sub):
    """Mean index per H3 cell over a class's UNIQUE origin locations."""
    locs = sub.drop_duplicates(subset=["lat_o", "lon_o"]).copy()
    locs["h3"] = [h3.latlng_to_cell(lat, lon, H3_RES)
                  for lat, lon in zip(locs["lat_o"], locs["lon_o"])]
    agg = (locs.groupby("h3")
                .agg(n=("h3", "size"), idx=(INDEX_COL, "mean"))
                .reset_index())
    agg = agg[agg["n"] >= MIN_ORIGINS_PER_HEX].copy()
    agg["geometry"] = agg["h3"].map(hexagon_polygon)
    return gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)


# -------------------------------------------------------------------------
# Maps: one panel per class (2 x 3), shared 0-1 colour scale.
# -------------------------------------------------------------------------
# %%
sp_boundary = cfg.load_sp_boundary()
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
gpkg_parts = []
for ax, cls in zip(axes.flat, CLASS_ORDER):
    sub = df[df["classe_economica"] == cls]
    gdf = aggregate_class(sub)
    sp_boundary.boundary.plot(ax=ax, color="black", linewidth=0.4)
    if len(gdf):
        gdf.plot(ax=ax, column="idx", cmap="viridis", vmin=0, vmax=1, legend=False)
        gdf["classe"] = cls
        gpkg_parts.append(gdf)
    # population-representative class mean (expansion-weighted) for the title
    wm = (sub[INDEX_COL] * sub["fe_via"]).sum() / sub["fe_via"].sum()
    ax.set_title(f"Class {cls}  (mean {wm:.2f}, n={len(sub):,})", fontsize=12)
    ax.set_axis_off()

sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
fig.colorbar(sm, ax=axes, shrink=0.5, label=f"{INDEX_COL} (multimodal, carbon-weighted)")
_scope = "non-residence anchors" if EXCLUDE_RESIDENCE else "all anchor trips"
fig.suptitle(f"15-Minute City index by economic class — H3 res {H3_RES} "
             f"({_scope} made by each class)", y=1.0, fontsize=15)
cfg.save_fig(f"12_equity_maps_h3_res{H3_RES}_{INDEX_COL}", fig)

# Export a single long-format hex layer (hex x class) for QGIS.
out = cfg.OUTPUT_DIR / f"equity_{INDEX_COL}_h3_res{H3_RES}.gpkg"
pd.concat(gpkg_parts, ignore_index=True).drop(columns="h3").to_file(out, driver="GPKG")
print(f"saved hex layer: {out}")

# %% [markdown]
# Notes:
# - Each panel is restricted to hexes where that class has origins, so coverage
#   differs by class (A is concentrated centrally; D-E is sparser and more
#   peripheral). The shared 0-1 scale makes the panels directly comparable.
# - Title mean is the expansion-weighted (fe_via) class mean, matching the equity
#   table in bcw_dis_09_index.py.
# - Swap INDEX_COL to "index_walk_15" to map the pedestrian-only index, which
#   shows the steepest class gradient.
