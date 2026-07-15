# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_16c_general_index_map_metro — figure 16 + metro-only overlay
# -------------------------------------------------------------------------
"""
Variant of figure 16 (general X-Minute City Accessibility Index at 15/20/30 min)
with only the metro lines drawn on top (red, thin, white casing). No title, so it
can sit under a caption in the document. Transparent background.
"""
# %%
import os
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

H3_RES = 9
INDEX_COLS = {15: "index_coverage_15", 20: "index_coverage_20", 30: "index_coverage_30"}
URBAN_GREY = "0.80"
METRO_C = "#e41a1c"                              # metro = red
SUFFIX = os.environ.get("MODEL_SUFFIX", "")

df = pd.read_parquet(cfg.OUTPUT_DIR / f"pmc_index_rich{SUFFIX}.parquet")
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()

metro = pd.read_pickle(cfg.CACHE_DIR / "osm_metro" / "metro_lines.pkl").to_crs(sp.crs)
metro = gpd.clip(metro, sp)


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


cols = list(INDEX_COLS.values())
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3")[cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)


fig, axes = plt.subplots(1, 3, figsize=(8.27, 5.83), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.0,
                            rect=(0, 0.10, 1, 1.0))
for ax, (t, col) in zip(axes, INDEX_COLS.items()):
    ax.patch.set_alpha(0)
    sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.4, zorder=0)
    urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
    gdf.plot(ax=ax, column=col, cmap="viridis", vmin=0, vmax=1, legend=False, zorder=2)
    if len(metro):
        metro.plot(ax=ax, color="white", linewidth=1.3, zorder=3)
        metro.plot(ax=ax, color=METRO_C, linewidth=0.6, zorder=4)
    ax.set_title(f"{t} minutes", fontsize=11)
    ax.set_axis_off()
    ax.margins(0)

sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
cb = fig.colorbar(sm, ax=list(axes), location="bottom", shrink=0.6, aspect=45, pad=0.01)
cb.set_label("X-Minute City Accessibility Index (0 = low → 1 = high)", fontsize=9)
cb.ax.tick_params(labelsize=8)

legend_items = [Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area"),
                Line2D([0], [0], color=METRO_C, lw=1.6, label="Metro")]
fig.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.02, 0.01),
           frameon=False, fontsize=9)

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / f"16c_general_index_metro_res{H3_RES}_A5{SUFFIX}.png"
fig.savefig(out, dpi=300, transparent=True)
print(f"Saved figure: {out}  (metro segs {len(metro)})")
