# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_22_resolution_compare — H3 res 9 vs res 10, coverage 15 min
# -------------------------------------------------------------------------
"""
Same 15-min coverage index aggregated at two H3 resolutions (9 and 10), side by
side, to show why res 9 reads better than res 10 (more origins per cell -> less
salt-and-pepper noise). Figure-16 styling: viridis, single shared scale, grey
urban base, transparent PNG with the rural interior kept white.
"""
# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

INDEX_COL = "index_coverage_15"
URBAN_GREY = "0.80"
RES = [9, 10]

df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def aggregate(res):
    locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
    locs["h3"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
    a = locs.groupby("h3").agg(n=("h3", "size"), idx=(INDEX_COL, "mean")).reset_index()
    a["geometry"] = a["h3"].map(hexpoly)
    return gpd.GeoDataFrame(a, geometry="geometry", crs=cfg.CRS_WGS84)


fig, axes = plt.subplots(1, 2, figsize=(8.27, 6.0), layout="constrained")
for ax, res in zip(axes, RES):
    gdf = aggregate(res)
    edge = h3.average_hexagon_edge_length(res, unit="m")
    ax.patch.set_alpha(0)
    sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.4, zorder=0)
    urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
    gdf.plot(ax=ax, column="idx", cmap="viridis", vmin=0, vmax=1, legend=False, zorder=2)
    ax.set_title(f"H3 resolution {res}  (~{edge:.0f} m)\n"
                 f"{len(gdf):,} cells · median {gdf['n'].median():.0f} origin(s)/cell",
                 fontsize=11)
    ax.set_axis_off()
    ax.margins(0)

sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
cb = fig.colorbar(sm, ax=list(axes), location="bottom", shrink=0.55, aspect=45, pad=0.01)
cb.set_label("Accessibility index (coverage, 15 min)", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.legend(handles=[Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")],
           loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=9)

fig.suptitle("X-Minute City Index (15 min) — H3 resolution comparison", fontsize=12)

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "22_h3_resolution_compare_15min.png"
fig.savefig(out, dpi=300, transparent=True)
print(f"Saved figure: {out}")
