# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_27_health_density_maps — CNEFE vs OSM healthcare density (3 maps)
# -------------------------------------------------------------------------
"""
Three district-level maps for São Paulo, healthcare category:
  1) CNEFE healthcare density (establishments per km2)
  2) OSM healthcare density (amenities per km2)
  3) Completeness ratio OSM / CNEFE per district
District scale is used because OSM healthcare is present in only ~5% of census
sectors, so a sector-level ratio map would be almost entirely empty.
"""
# %%
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import bcw_dis_00_config as cfg

BIAS_GPKG = cfg.OUTPUT_DIR / "cnefe_osm_bias_district.gpkg"
CRS_M = "EPSG:31983"                       # SIRGAS 2000 / UTM 23S (metres)
SE_LONLAT = (-46.6333, -23.5505)           # Praça da Sé (centre)

d = gpd.read_file(BIAS_GPKG)
d["area_km2"] = d.to_crs(CRS_M).geometry.area / 1e6
d["cnefe_dens"] = d["cnefe_healthcare"] / d["area_km2"]
d["osm_dens"] = d["osm_healthcare"] / d["area_km2"]
d["ratio"] = d["ratio_healthcare"]

se = gpd.GeoSeries(gpd.points_from_xy([SE_LONLAT[0]], [SE_LONLAT[1]]),
                   crs=cfg.CRS_WGS84).to_crs(d.crs)

panels = [
    ("cnefe_dens", "CNEFE density\n(establishments / km²)", "viridis"),
    ("osm_dens",   "OSM density\n(amenities / km²)",        "viridis"),
    ("ratio",      "Completeness ratio\n(OSM / CNEFE)",     "plasma"),
]

fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2))
for ax, (col, title, cmap) in zip(axes, panels):
    vmax = float(np.nanpercentile(d[col], 95))
    d.plot(ax=ax, column=col, cmap=cmap, vmin=0, vmax=vmax,
           edgecolor="0.75", linewidth=0.15)
    se.plot(ax=ax, color="red", marker="*", markersize=70, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.margins(0)
    sm = ScalarMappable(norm=Normalize(0, vmax), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.8, pad=0.02, aspect=25)
    cb.ax.tick_params(labelsize=7)

fig.suptitle("Healthcare provision — CNEFE vs OSM, by district (São Paulo, 2022)  •  red star = centre (Sé)",
             fontsize=12)
fig.tight_layout()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "27_health_cnefe_osm_density_maps.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved figure: {out}")
print(d[["NM_DIST", "cnefe_dens", "osm_dens", "ratio"]].describe().round(3).to_string())
