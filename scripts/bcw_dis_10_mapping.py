# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_10_mapping — H3 spatial aggregation + choropleth maps
# -------------------------------------------------------------------------
"""
A-posteriori spatial aggregation of the 15-Minute City index (Saraiva & Barros,
2022) onto an Uber H3 hexagonal grid, by trip-origin location.

Two resolutions are produced for comparison:
  - res 10  (~66 m edge, ~0.015 km^2 per cell)  -- fine detail
  - res 9   (~174 m edge, ~0.10 km^2 per cell)  -- less sparse

Each index value is a property of the origin LOCATION, so we aggregate over
UNIQUE origin points per hex (each location once) -> the map shows the spatial
distribution of accessibility without trip-frequency bias. Cell value = mean
index of the origins falling in it; `n` = number of distinct origins (use it to
hide thin cells).

Reads the index output written by bcw_dis_09_index.py (full run preferred;
falls back to the sample). Maps the walk / active / all variants at 15 min.

Dependencies: h3 (v4), geopandas, shapely, pandas, matplotlib.
"""
# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

# --- Settings ---
H3_RESOLUTIONS = [9, 10]
THRESHOLDS_TO_MAP = [15, 20, 30]
MIN_ORIGINS_PER_HEX = 1     # raise (e.g. 3) to suppress thinly-sampled cells

# Per threshold, the three nested-index columns (walk / active / all).
def variant_cols(t):
    return {"Walk only": f"index_walk_{t}",
            "Walk + bike": f"index_active_{t}",
            "Multimodal, carbon-weighted": f"index_{t}"}

# Every index column we need to aggregate (3 variants x 3 thresholds).
MAP_INDICES = [c for t in THRESHOLDS_TO_MAP for c in variant_cols(t).values()]

# Prefer the full-run output; fall back to the sample.
FULL = cfg.OUTPUT_DIR / "pmc_index.parquet"
SAMPLE = cfg.OUTPUT_DIR / "pmc_index_sample.parquet"
src = FULL if FULL.exists() else SAMPLE
print(f"Reading index results: {src.name}")
df = pd.read_parquet(src)


# -------------------------------------------------------------------------
# 1. Aggregate index onto the H3 grid (per unique origin location)
# -------------------------------------------------------------------------
def hexagon_polygon(cell):
    """shapely Polygon (lon, lat) for an H3 cell (v4 returns lat, lng pairs)."""
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in boundary])


def aggregate_to_h3(df, res):
    """Mean index per H3 cell over UNIQUE origin locations. Returns a GeoDataFrame."""
    # one row per distinct origin location (avoids trip-frequency double counting)
    locs = df.drop_duplicates(subset=["lat_o", "lon_o"]).copy()
    locs["h3"] = [h3.latlng_to_cell(lat, lon, res)
                  for lat, lon in zip(locs["lat_o"], locs["lon_o"])]
    agg = (locs.groupby("h3")
                .agg(n=("h3", "size"),
                     **{c: (c, "mean") for c in MAP_INDICES})
                .reset_index())
    agg = agg[agg["n"] >= MIN_ORIGINS_PER_HEX].copy()
    agg["geometry"] = agg["h3"].map(hexagon_polygon)
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)
    return gdf


# -------------------------------------------------------------------------
# 2. Maps
# -------------------------------------------------------------------------
# %%
sp_boundary = cfg.load_sp_boundary()

for res in H3_RESOLUTIONS:
    edge_m = h3.average_hexagon_edge_length(res, unit="m")
    gdf = aggregate_to_h3(df, res)
    print(f"\nH3 res {res} (~{edge_m:.0f} m edge): {len(gdf):,} cells "
          f"(>= {MIN_ORIGINS_PER_HEX} origins); "
          f"median origins/cell = {gdf['n'].median():.0f}")

    # One figure per threshold (3 panels: walk / active / all).
    for t in THRESHOLDS_TO_MAP:
        cols = variant_cols(t)
        fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 7))
        for ax, (label, col) in zip(axes, cols.items()):
            sp_boundary.boundary.plot(ax=ax, color="black", linewidth=0.6)
            gdf.plot(ax=ax, column=col, cmap="viridis", vmin=0, vmax=1,
                     legend=True, legend_kwds={"shrink": 0.5, "label": "index"})
            ax.set_title(f"{label} ({t} min)")
            ax.set_axis_off()
        fig.suptitle(f"15-Minute City index — H3 resolution {res} "
                     f"(~{edge_m:.0f} m edge) — {t} min", y=1.02, fontsize=14)
        cfg.save_fig(f"10_index_map_h3_res{res}_{t}min", fig)

    # Save the aggregated hex layer (all thresholds/variants) for GIS.
    out = cfg.OUTPUT_DIR / f"index_h3_res{res}.gpkg"
    gdf.drop(columns="h3").to_file(out, driver="GPKG")
    print(f"  saved hex layer: {out}")

# %% [markdown]
# Notes / suggestions:
# - Run bcw_dis_09_index.py with SAMPLE_N = None first; 500 sampled origins are
#   far too sparse to fill the grid (especially res 10). The map is only
#   meaningful on the full ~29.5k-origin output.
# - res 10 gives fine detail but many 1-origin cells (noisy); res 9 is smoother.
#   Use the printed median origins/cell and MIN_ORIGINS_PER_HEX to judge/clean.
# - vmin/vmax fixed to [0,1] so the walk/active/all panels are directly
#   comparable — the walk panel should show the real spatial gradient.
# - The exported .gpkg layers open directly in QGIS for cartography.
