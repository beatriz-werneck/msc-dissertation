# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_15b_overall — overall rich-index maps (dark mode, GnBu, green=high)
# -------------------------------------------------------------------------
"""
Overall rich-index maps (H3 res 9) for 15/20/30 min, panels Binary -> Count ->
Diversity. Dark-mode styling: black background (non-urban São Paulo reads as
black), grey urban-zone base, GnBu palette with GREEN = high / dark blue = low.
A single shared colour scale and one 'Urban area' legend per figure.
"""
# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from shapely.geometry import Polygon

import bcw_dis_00_config as cfg

H3_RES = 9
BG = "black"
URBAN_GREY = "#4a4a4a"
TEXT = "white"

# Rich green -> teal -> dark blue from ColorBrewer YlGnBu (drop pale low end),
# then REVERSED so GREEN = high index and dark blue = low index.
_base = plt.get_cmap("YlGnBu")
_gnbu = LinearSegmentedColormap.from_list("gnbu_rich", _base(np.linspace(0.25, 1.0, 256)))
cmap = _gnbu.reversed()

PANELS = [("Binary (baseline)", "index_binary_{t}"),
          ("Count (intensity)", "index_count_{t}"),
          ("Diversity (coverage)", "index_coverage_{t}")]

df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


cols = [c.format(t=t) for _, c in PANELS for t in cfg.THRESHOLDS]
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3")[cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)


def save_dark(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved figure: {path}")


for t in cfg.THRESHOLDS:
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor=BG)
    for ax, (label, col) in zip(axes, PANELS):
        ax.set_facecolor(BG)
        sp.plot(ax=ax, color=BG, edgecolor=TEXT, linewidth=0.6, zorder=0)   # municipality silhouette
        urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)     # urban zone (grey)
        gdf.plot(ax=ax, column=col.format(t=t), cmap=cmap, vmin=0, vmax=1,
                 legend=False, zorder=2)
        ax.set_title(f"{label} ({t} min)", color=TEXT, fontsize=13)
        ax.set_axis_off()

    # One shared colour scale for the whole figure.
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cb = fig.colorbar(sm, ax=list(axes), location="bottom", shrink=0.4, aspect=40, pad=0.05)
    cb.set_label("Accessibility index  (0 = low → 1 = high)", color=TEXT)
    cb.ax.xaxis.set_tick_params(color=TEXT)
    plt.setp(plt.getp(cb.ax, "xticklabels"), color=TEXT)
    cb.outline.set_edgecolor(TEXT)

    # One urban-area legend, next to the scale.
    fig.legend(handles=[Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")],
               loc="lower left", bbox_to_anchor=(0.13, 0.05), frameon=False,
               labelcolor=TEXT, fontsize=11)

    fig.suptitle(f"Rich X-Minute City index — H3 res {H3_RES} — {t} min",
                 y=0.98, fontsize=15, color=TEXT)
    save_dark(fig, f"15e_rich_overall_dark_res{H3_RES}_{t}min")

print("Done — dark-mode GnBu (green=high) overall maps saved for 15/20/30 min.")
