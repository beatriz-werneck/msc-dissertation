# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_17_overall_variants_A5 — binary / count / coverage at 15 min (A5)
# -------------------------------------------------------------------------
"""
One A5 figure with the three index variants at 15 min, in order
Binary -> Count -> Coverage (left to right). Same styling as figure 16:
viridis, single shared scale, grey 'Urban area' legend, transparent PNG with
the rural interior of São Paulo kept white.
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

H3_RES = 9
URBAN_GREY = "0.80"
# Panels left -> right: binary, count, coverage (all at 15 min).
PANELS = [("Binary (baseline)", "index_binary_15"),
          ("Count (intensity)", "index_count_15"),
          ("Coverage (index)", "index_coverage_15")]

import os
SUFFIX = os.environ.get("MODEL_SUFFIX", "")   # "" weighted; "_noweight" for mode-agnostic
df = pd.read_parquet(cfg.OUTPUT_DIR / f"pmc_index_rich{SUFFIX}.parquet")
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


cols = [c for _, c in PANELS]
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3")[cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)

# A5 landscape canvas (210 x 148 mm), maps packed to fill it.
fig, axes = plt.subplots(1, 3, figsize=(8.27, 5.83), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.0)
for ax, (label, col) in zip(axes, PANELS):
    ax.patch.set_alpha(0)
    sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.4, zorder=0)
    urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
    gdf.plot(ax=ax, column=col, cmap="viridis", vmin=0, vmax=1, legend=False, zorder=2)
    ax.set_title(label, fontsize=11)
    ax.set_axis_off()
    ax.margins(0)

sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
cb = fig.colorbar(sm, ax=list(axes), location="bottom", shrink=0.6, aspect=45, pad=0.01)
cb.set_label("Accessibility score (0 = low → 1 = high)", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.legend(handles=[Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")],
           loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=9)


cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / f"17_rich_overall_variants_15min_res{H3_RES}_A5{SUFFIX}.png"
fig.savefig(out, dpi=300, transparent=True)
print(f"Saved figure: {out}  ({fig.get_size_inches()[0]*25.4:.0f} x {fig.get_size_inches()[1]*25.4:.0f} mm)")
