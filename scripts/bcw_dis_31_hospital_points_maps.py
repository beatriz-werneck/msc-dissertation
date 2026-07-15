# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_31_hospital_points_maps — CNEFE vs OSM HOSPITAL points (2 maps)
# -------------------------------------------------------------------------
"""
Side-by-side maps of hospital POINTS from each source (the like-for-like facility
type). CNEFE: species 5 whose name contains HOSPITAL / PRONTO SOCORRO /
MATERNIDADE / SANTA CASA (excluding veterinary). OSM: amenity/healthcare=hospital.
"""
# %%
import unicodedata
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import bcw_dis_00_config as cfg

SP_MUNI = "3550308"
CNEFE_CSV = cfg.RAW_DIR / "cnefe" / "35_SP.csv"
CHUNK = 2_000_000
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -24.10, -23.30, -46.90, -46.30
HOSP_KW = ["HOSPITAL", "PRONTO SOCORRO", "PRONTO-SOCORRO", "MATERNIDADE", "SANTA CASA"]


def norm(s):
    s = "" if s is None else str(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()


def is_hospital(name):
    return any(k in name for k in HOSP_KW) and "VETERIN" not in name


# 1. CNEFE hospital points
print("Reading CNEFE hospital points ...", flush=True)
parts = []
for i, ch in enumerate(pd.read_csv(
        CNEFE_CSV, sep=";",
        usecols=["COD_MUNICIPIO", "COD_ESPECIE", "DSC_ESTABELECIMENTO", "LATITUDE", "LONGITUDE"],
        dtype=str, encoding="latin-1", chunksize=CHUNK)):
    ch = ch[(ch["COD_MUNICIPIO"] == SP_MUNI) & (ch["COD_ESPECIE"] == "5")]
    if not ch.empty:
        ch = ch[ch["DSC_ESTABELECIMENTO"].map(norm).map(is_hospital)]
        if not ch.empty:
            parts.append(ch[["LATITUDE", "LONGITUDE"]])
    print(f"    chunk {i} done", flush=True)

cn = pd.concat(parts, ignore_index=True)
cn["lat"] = pd.to_numeric(cn["LATITUDE"], errors="coerce")
cn["lon"] = pd.to_numeric(cn["LONGITUDE"], errors="coerce")
cn = cn.dropna(subset=["lat", "lon"])
cn = cn[cn["lat"].between(LAT_MIN, LAT_MAX) & cn["lon"].between(LON_MIN, LON_MAX)]
cnefe = gpd.GeoDataFrame(cn, geometry=gpd.points_from_xy(cn["lon"], cn["lat"]),
                         crs="EPSG:4674").to_crs(cfg.CRS_WGS84)
print(f"  CNEFE hospital points: {len(cnefe):,}")

# 2. OSM hospital points
amen = pd.read_pickle(cfg.AMENITIES_PKL)
amen = gpd.GeoDataFrame(amen[amen.geometry.notna()].copy(), geometry="geometry")
if amen.crs is None:
    amen.set_crs(cfg.CRS_WGS84, inplace=True)
osm = amen[(amen.get("amenity") == "hospital") | (amen.get("healthcare") == "hospital")].to_crs(cfg.CRS_WGS84)
print(f"  OSM hospital points: {len(osm):,}")

# 3. Two point maps side by side (same extent)
sp = cfg.load_sp_boundary()
xmin, ymin, xmax, ymax = sp.total_bounds
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, (gdf, title) in zip(axes, [
        (cnefe, f"CNEFE hospitals (n = {len(cnefe):,})"),
        (osm,   f"OSM hospitals (n = {len(osm):,})")]):
    sp.plot(ax=ax, color="0.96", edgecolor="0.6", linewidth=0.5, zorder=0)
    gdf.plot(ax=ax, markersize=9, color="#c0392b", alpha=0.6, edgecolor="none", zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
fig.suptitle("Hospital points — CNEFE vs OSM, São Paulo municipality (2022)", fontsize=12)
fig.tight_layout()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "31_hospital_cnefe_osm_points.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"Saved figure: {out}")
