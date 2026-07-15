# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_28_health_points_maps — CNEFE vs OSM healthcare POINTS (2 maps)
# -------------------------------------------------------------------------
"""
Side-by-side maps of the raw healthcare POINTS (no density), to compare the
spatial distribution of the two sources directly:
  1) CNEFE healthcare establishments (COD_ESPECIE = 5), São Paulo municipality
  2) OSM healthcare amenities
Point over-plotting itself conveys where each source concentrates.
"""
# %%
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import bcw_dis_00_config as cfg

SP_MUNI = "3550308"
CNEFE_CSV = cfg.RAW_DIR / "cnefe" / "35_SP.csv"
CHUNK = 2_000_000
# São Paulo municipality bounding box (drop obviously bad coordinates)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -24.10, -23.30, -46.90, -46.30

# -------------------------------------------------------------------------
# 1. CNEFE healthcare points (chunked read; keep coordinates this time)
# -------------------------------------------------------------------------
print("Reading CNEFE healthcare points ...", flush=True)
parts = []
for i, ch in enumerate(pd.read_csv(
        CNEFE_CSV, sep=";",
        usecols=["COD_MUNICIPIO", "COD_ESPECIE", "LATITUDE", "LONGITUDE"],
        dtype=str, encoding="latin-1", chunksize=CHUNK)):
    ch = ch[(ch["COD_MUNICIPIO"] == SP_MUNI) & (ch["COD_ESPECIE"] == "5")]
    if not ch.empty:
        parts.append(ch[["LATITUDE", "LONGITUDE"]])
    print(f"    chunk {i} done", flush=True)

cn = pd.concat(parts, ignore_index=True)
cn["lat"] = pd.to_numeric(cn["LATITUDE"], errors="coerce")
cn["lon"] = pd.to_numeric(cn["LONGITUDE"], errors="coerce")
cn = cn.dropna(subset=["lat", "lon"])
cn = cn[(cn["lat"].between(LAT_MIN, LAT_MAX)) & (cn["lon"].between(LON_MIN, LON_MAX))]
cnefe = gpd.GeoDataFrame(cn, geometry=gpd.points_from_xy(cn["lon"], cn["lat"]),
                         crs="EPSG:4674").to_crs(cfg.CRS_WGS84)
print(f"  CNEFE healthcare points: {len(cnefe):,}")

# -------------------------------------------------------------------------
# 2. OSM healthcare points
# -------------------------------------------------------------------------
amen = pd.read_pickle(cfg.AMENITIES_PKL)
amen = gpd.GeoDataFrame(amen[amen.geometry.notna()].copy(), geometry="geometry")
if amen.crs is None:
    amen.set_crs(cfg.CRS_WGS84, inplace=True)
osm = amen[amen["category"] == "healthcare"].to_crs(cfg.CRS_WGS84)
print(f"  OSM healthcare points: {len(osm):,}")

# -------------------------------------------------------------------------
# 3. Plot the two point maps side by side (same extent)
# -------------------------------------------------------------------------
sp = cfg.load_sp_boundary()
xmin, ymin, xmax, ymax = sp.total_bounds

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, (gdf, title, color) in zip(axes, [
        (cnefe, f"CNEFE healthcare establishments (n = {len(cnefe):,})", "#08519c"),
        (osm,   f"OSM healthcare amenities (n = {len(osm):,})",          "#08519c")]):
    sp.plot(ax=ax, color="0.96", edgecolor="0.6", linewidth=0.5, zorder=0)
    gdf.plot(ax=ax, markersize=1.2, color=color, alpha=0.35, zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_axis_off()

fig.suptitle("Healthcare points — CNEFE vs OSM, São Paulo municipality (2022)", fontsize=12)
fig.tight_layout()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "28_health_cnefe_osm_points.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved figure: {out}")
