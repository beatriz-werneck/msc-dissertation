# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_30_hospital_bias — OSM completeness bias, HOSPITAL level (apples-to-apples)
# -------------------------------------------------------------------------
"""
The all-healthcare comparison mixed non-equivalent sets (CNEFE saúde is dominated
by small dental/medical consultórios OSM never maps; OSM healthcare is inflated
by pharmacies, which CNEFE files under commerce). Hospitals are large, prominent
facilities that BOTH sources map reasonably, so they give a like-for-like test.

CNEFE hospitals: species 5 whose name contains HOSPITAL / PRONTO SOCORRO /
MATERNIDADE / SANTA CASA (excluding veterinary), assigned to district by sector
code. OSM hospitals: amenity=hospital OR healthcare=hospital, assigned by spatial
join. Test: Spearman(district completeness ratio OSM/CNEFE, distance to centre).
"""
# %%
import re
import unicodedata
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import bcw_dis_00_config as cfg

SP_MUNI = "3550308"
CNEFE_CSV = cfg.RAW_DIR / "cnefe" / "35_SP.csv"
BIAS_GPKG = cfg.OUTPUT_DIR / "cnefe_osm_bias_district.gpkg"   # district geom + dist_centre_km
CHUNK = 2_000_000
HOSP_KW = ["HOSPITAL", "PRONTO SOCORRO", "PRONTO-SOCORRO", "MATERNIDADE", "SANTA CASA"]


def norm(s):
    s = "" if s is None else str(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()


def is_hospital(name):
    return any(k in name for k in HOSP_KW) and "VETERIN" not in name


# -------------------------------------------------------------------------
# 1. CNEFE hospitals per district (species 5 + name filter, sector-code join)
# -------------------------------------------------------------------------
print("Reading CNEFE hospitals ...", flush=True)
parts = []
for i, ch in enumerate(pd.read_csv(
        CNEFE_CSV, sep=";", usecols=["COD_MUNICIPIO", "COD_ESPECIE", "COD_SETOR", "DSC_ESTABELECIMENTO"],
        dtype=str, encoding="latin-1", chunksize=CHUNK)):
    ch = ch[(ch["COD_MUNICIPIO"] == SP_MUNI) & (ch["COD_ESPECIE"] == "5")]
    if not ch.empty:
        ch = ch[ch["DSC_ESTABELECIMENTO"].map(norm).map(is_hospital)]
        if not ch.empty:
            parts.append(ch[["COD_SETOR"]])
    print(f"    chunk {i} done", flush=True)

cn = pd.concat(parts, ignore_index=True)
cn["CD_DIST"] = cn["COD_SETOR"].str.replace(r"\D", "", regex=True).str[:9]
cnefe_h = cn.groupby("CD_DIST").size().rename("cnefe_hospital").reset_index()
print(f"  CNEFE hospitals (SP municipality): {len(cn):,}")

# -------------------------------------------------------------------------
# 2. OSM hospitals per district (spatial join)
# -------------------------------------------------------------------------
d = gpd.read_file(BIAS_GPKG)[["CD_DIST", "NM_DIST", "dist_centre_km", "geometry"]]
amen = pd.read_pickle(cfg.AMENITIES_PKL)
amen = gpd.GeoDataFrame(amen[amen.geometry.notna()].copy(), geometry="geometry")
if amen.crs is None:
    amen.set_crs(cfg.CRS_WGS84, inplace=True)
hosp = amen[(amen.get("amenity") == "hospital") | (amen.get("healthcare") == "hospital")].to_crs(d.crs)
print(f"  OSM hospitals: {len(hosp):,}")
j = gpd.sjoin(hosp[["geometry"]], d[["CD_DIST", "geometry"]], how="inner", predicate="within")
osm_h = j.groupby("CD_DIST").size().rename("osm_hospital").reset_index()

# -------------------------------------------------------------------------
# 3. Merge, ratio, test
# -------------------------------------------------------------------------
g = d.merge(cnefe_h, on="CD_DIST", how="left").merge(osm_h, on="CD_DIST", how="left")
g[["cnefe_hospital", "osm_hospital"]] = g[["cnefe_hospital", "osm_hospital"]].fillna(0)
g["ratio"] = g["osm_hospital"] / g["cnefe_hospital"].replace(0, np.nan)

sub = g.dropna(subset=["ratio"])
rho, p = spearmanr(sub["dist_centre_km"], sub["ratio"])
print("\n=== HOSPITAL-level completeness ratio (OSM/CNEFE) vs distance to centre ===")
print(f"  districts with CNEFE hospitals (usable): {len(sub)} / {len(g)}")
print(f"  totals   -> OSM {int(g['osm_hospital'].sum())}  |  CNEFE {int(g['cnefe_hospital'].sum())}")
print(f"  mean completeness ratio: {sub['ratio'].mean():.3f}  (median {sub['ratio'].median():.3f})")
print(f"  Spearman(distance, ratio): rho = {rho:+.3f}   p = {p:.1e}")
print("  (rho ~ 0 => no spatial bias at hospital level; rho < 0 => OSM better in centre)")

g[["CD_DIST", "NM_DIST", "dist_centre_km", "cnefe_hospital", "osm_hospital", "ratio"]] \
    .to_csv(cfg.OUTPUT_DIR / "hospital_bias_district.csv", index=False)

# -------------------------------------------------------------------------
# 4. Scatter: hospital completeness ratio vs distance
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.2))
x, y = sub["dist_centre_km"].to_numpy(), sub["ratio"].to_numpy()
ax.scatter(x, y, s=22, alpha=0.65, color="#2c7fb8", edgecolor="none")
b, a = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, a + b * xs, color="#d95f0e", lw=1.4)
ax.set_title(f"Hospital completeness (OSM / CNEFE) vs distance to centre\n"
             f"São Paulo districts — Spearman ρ = {rho:+.2f} (p = {p:.1e})", fontsize=10)
ax.set_xlabel("Distance to city centre (km)", fontsize=9)
ax.set_ylabel("Completeness ratio  OSM / CNEFE", fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "30_hospital_ratio_vs_distance.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved figure: {out}")
