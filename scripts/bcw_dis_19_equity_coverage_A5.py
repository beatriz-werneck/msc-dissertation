# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_19_equity_coverage_A5 — coverage index by economic class (A5)
# -------------------------------------------------------------------------
"""
One A5 figure with the coverage index at 15 min mapped separately for each
Critério Brasil economic class (A, B1, B2, C1, C2, D-E; 2 x 3 grid), using all
anchor trips of each class. Same styling as figure 16: viridis, single shared
scale, grey 'Urban area' legend, transparent PNG with the rural interior kept
white.
"""
# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

H3_RES = 9
URBAN_GREY = "0.80"
INDEX_COL = "index_coverage_15"
CLASS_ORDER = ["A", "B1", "B2", "C1", "C2", "D-E"]

import os
SUFFIX = os.environ.get("MODEL_SUFFIX", "")   # "" weighted; "_noweight" for mode-agnostic
df = pd.read_parquet(cfg.OUTPUT_DIR / f"pmc_index_rich{SUFFIX}.parquet")
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def aggregate(sub):
    locs = sub.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
    locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
    a = locs.groupby("h3")[INDEX_COL].mean().reset_index()
    a["geometry"] = a["h3"].map(hexpoly)
    return gpd.GeoDataFrame(a, geometry="geometry", crs=cfg.CRS_WGS84)


# A5 landscape canvas (210 x 148 mm), 2 x 3 grid packed to fill it.
fig, axes = plt.subplots(2, 3, figsize=(8.27, 5.83), layout="constrained")
fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.02)
for ax, cls in zip(axes.flat, CLASS_ORDER):
    sub = df[df["classe_economica"] == cls]
    ax.patch.set_alpha(0)
    sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.3, zorder=0)
    urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
    g = aggregate(sub)
    if len(g):
        g.plot(ax=ax, column=INDEX_COL, cmap="viridis", vmin=0, vmax=1, legend=False, zorder=2)
    wm = (sub[INDEX_COL] * sub["fe_via"]).sum() / sub["fe_via"].sum()   # expansion-weighted class mean
    ax.set_title(f"Class {cls}  (mean {wm:.2f})", fontsize=10)
    ax.set_axis_off()
    ax.margins(0)

sm = ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), location="bottom", shrink=0.5, aspect=45, pad=0.01)
cb.set_label("Coverage index, 15 min (0 = low → 1 = high)", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.legend(handles=[Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")],
           loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=9)


cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / f"19_equity_coverage_res{H3_RES}_A5{SUFFIX}.png"
fig.savefig(out, dpi=300, transparent=True)
print(f"Saved figure: {out}  ({fig.get_size_inches()[0]*25.4:.0f} x {fig.get_size_inches()[1]*25.4:.0f} mm)")
