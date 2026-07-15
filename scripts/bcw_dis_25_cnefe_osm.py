# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_25_cnefe_osm — OSM vs CNEFE amenity-density comparison (validation)
# -------------------------------------------------------------------------
"""
Robustness check for the OSM amenity layer. OSM is thought to be better mapped
in central / wealthier areas; if that bias is strong, part of the "centre is
highly accessible" and "richer classes have more access" patterns could be a
data artefact. To test this, the spatial distribution of OSM amenities is
compared against CNEFE (IBGE's official address register, more complete and
uniform), using the 2022 census SECTORS as the spatial unit (dissolvable to
DISTRICT for a more robust comparison).

Method:
  - CNEFE establishments are counted per census sector using the sector code
    the register already carries (COD_SETOR) -> direct code join, no spatial
    join and no dependence on CNEFE point coordinates.
  - OSM amenities are counted per sector by spatial join into the sector
    polygons.
  - Densities (count / km2) and Spearman rank correlations between the two
    sources are computed per comparable category, at sector and district scale.

Comparable categories (CNEFE COD_ESPECIE -> OSM category):
  education (4 -> education), healthcare (5 -> healthcare),
  religion (8 -> civic_religion), commerce (6 -> dining/retail/services/
  groceries/culture/fitness), and total (all of the above).
CNEFE residences (1,2), agriculture (3) and construction (7) are excluded.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

import bcw_dis_00_config as cfg

SP_MUNI = "3550308"                              # São Paulo capital
CNEFE_CSV = cfg.RAW_DIR / "cnefe" / "35_SP.csv"
SETORES_SHP = cfg.RAW_DIR / "SP_setores_CD2022" / "SP_setores_CD2022.shp"
OUT_DIR = cfg.OUTPUT_DIR
CHUNK = 2_000_000

# CNEFE establishment species -> comparison category
ESP_MAP = {"4": "education", "5": "healthcare", "6": "commerce", "8": "religion"}
# OSM 10-category scheme -> comparison category
OSM_MAP = {"education": "education", "healthcare": "healthcare",
           "civic_religion": "religion",
           "dining": "commerce", "retail": "commerce", "services": "commerce",
           "groceries": "commerce", "culture": "commerce", "fitness": "commerce"}
CMP_CATS = ["education", "healthcare", "religion", "commerce"]   # + "total"


# -------------------------------------------------------------------------
# 1. Census sectors of São Paulo municipality
# -------------------------------------------------------------------------
print("Loading census sectors ...", flush=True)
sect = gpd.read_file(SETORES_SHP)
sect = sect[sect["CD_MUN"] == SP_MUNI].copy()
sect["CD_SETOR"] = sect["CD_SETOR"].astype(str)
print(f"  sectors in São Paulo: {len(sect):,}  | districts: {sect['CD_DIST'].nunique()}")


# -------------------------------------------------------------------------
# 2. CNEFE establishment counts per sector (direct code join, chunked)
# -------------------------------------------------------------------------
print("Counting CNEFE establishments per sector (chunked read) ...", flush=True)
parts = []
n_rows = 0
for i, ch in enumerate(pd.read_csv(
        CNEFE_CSV, sep=";", usecols=["COD_MUNICIPIO", "COD_SETOR", "COD_ESPECIE"],
        dtype=str, encoding="latin-1", chunksize=CHUNK)):
    n_rows += len(ch)
    ch = ch[ch["COD_MUNICIPIO"] == SP_MUNI]
    ch = ch[ch["COD_ESPECIE"].isin(ESP_MAP)]
    if ch.empty:
        continue
    ch["cmp"] = ch["COD_ESPECIE"].map(ESP_MAP)
    ch["CD_SETOR"] = ch["COD_SETOR"].str.replace(r"\D", "", regex=True)   # strip trailing letter
    parts.append(ch.groupby(["CD_SETOR", "cmp"]).size())
    print(f"    chunk {i}: scanned {n_rows:,} rows", flush=True)

cnefe = (pd.concat(parts).groupby(level=[0, 1]).sum()
         .unstack("cmp", fill_value=0).reset_index())
cnefe.columns.name = None
for c in CMP_CATS:
    if c not in cnefe.columns:
        cnefe[c] = 0
cnefe = cnefe.rename(columns={c: f"cnefe_{c}" for c in CMP_CATS})
match = cnefe["CD_SETOR"].isin(set(sect["CD_SETOR"])).mean()
print(f"  CNEFE sector codes matching the shapefile: {match:.1%}")


# -------------------------------------------------------------------------
# 3. OSM amenity counts per sector (spatial join)
# -------------------------------------------------------------------------
print("Counting OSM amenities per sector (spatial join) ...", flush=True)
amen = pd.read_pickle(cfg.AMENITIES_PKL)
amen = gpd.GeoDataFrame(amen[amen.geometry.notna()].copy(), geometry="geometry")
if amen.crs is None:
    amen.set_crs(cfg.CRS_WGS84, inplace=True)
amen = amen[amen["category"].isin(OSM_MAP)].copy()
amen["cmp"] = amen["category"].map(OSM_MAP)
amen = amen.to_crs(sect.crs)

j = gpd.sjoin(amen[["cmp", "geometry"]], sect[["CD_SETOR", "geometry"]],
              how="inner", predicate="within")
osm = (j.groupby(["CD_SETOR", "cmp"]).size()
       .unstack("cmp", fill_value=0).reset_index())
osm.columns.name = None
for c in CMP_CATS:
    if c not in osm.columns:
        osm[c] = 0
osm = osm.rename(columns={c: f"osm_{c}" for c in CMP_CATS})


# -------------------------------------------------------------------------
# 4. Merge onto sectors, densities, totals
# -------------------------------------------------------------------------
g = sect[["CD_SETOR", "CD_DIST", "NM_DIST", "AREA_KM2", "geometry"]].merge(
    cnefe, on="CD_SETOR", how="left").merge(osm, on="CD_SETOR", how="left")
cnt_cols = [f"{s}_{c}" for s in ("cnefe", "osm") for c in CMP_CATS]
g[cnt_cols] = g[cnt_cols].fillna(0)
for s in ("cnefe", "osm"):
    g[f"{s}_total"] = g[[f"{s}_{c}" for c in CMP_CATS]].sum(axis=1)
# densities (per km2)
for col in [f"{s}_{c}" for s in ("cnefe", "osm") for c in CMP_CATS + ["total"]]:
    g[f"{col}_dens"] = g[col] / g["AREA_KM2"].replace(0, np.nan)


# -------------------------------------------------------------------------
# 5. Correlations (sector scale and district scale)
# -------------------------------------------------------------------------
def corr_table(df, level):
    rows = []
    for c in CMP_CATS + ["total"]:
        x, y = df[f"cnefe_{c}"].to_numpy(float), df[f"osm_{c}"].to_numpy(float)
        rho, p = spearmanr(x, y)
        rows.append((level, c, len(df), int((x > 0).sum()), int((y > 0).sum()),
                     round(rho, 3), f"{p:.1e}"))
    return pd.DataFrame(rows, columns=["scale", "category", "n_units",
                                       "cnefe>0", "osm>0", "spearman_rho", "p"])

sec_tbl = corr_table(g, "sector")
dist = g.groupby(["CD_DIST", "NM_DIST"], as_index=False)[cnt_cols].sum()
for s in ("cnefe", "osm"):
    dist[f"{s}_total"] = dist[[f"{s}_{c}" for c in CMP_CATS]].sum(axis=1)
dist_tbl = corr_table(dist, "district")
summary = pd.concat([sec_tbl, dist_tbl], ignore_index=True)
print("\n=== OSM vs CNEFE — Spearman correlation ===")
print(summary.to_string(index=False))


# -------------------------------------------------------------------------
# 6. Save outputs
# -------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
g.drop(columns="geometry").to_parquet(OUT_DIR / "cnefe_osm_sector.parquet", index=False)
g.to_file(OUT_DIR / "cnefe_osm_sector.gpkg", driver="GPKG")
dist.to_parquet(OUT_DIR / "cnefe_osm_district.parquet", index=False)
summary.to_csv(OUT_DIR / "cnefe_osm_correlation.csv", index=False)
print(f"\nSaved: cnefe_osm_sector.parquet/.gpkg, cnefe_osm_district.parquet, "
      f"cnefe_osm_correlation.csv  -> {OUT_DIR}")


# -------------------------------------------------------------------------
# 7. Scatter figure (district scale, log-log) — English labels
# -------------------------------------------------------------------------
LABELS = {"education": "Education", "healthcare": "Healthcare", "religion": "Religion",
          "commerce": "Commerce & services", "total": "All establishments"}
cats = CMP_CATS + ["total"]
fig, axes = plt.subplots(1, len(cats), figsize=(3.0 * len(cats), 3.1))
for ax, c in zip(axes, cats):
    x, y = dist[f"cnefe_{c}"] + 1, dist[f"osm_{c}"] + 1
    ax.scatter(x, y, s=10, alpha=0.5, color="#2c7fb8", edgecolor="none")
    lim = [1, max(x.max(), y.max())]
    ax.plot(lim, lim, color="0.6", lw=0.8, ls="--")     # 1:1 reference
    rho = spearmanr(dist[f"cnefe_{c}"], dist[f"osm_{c}"]).statistic
    ax.set(xscale="log", yscale="log", title=f"{LABELS[c]}\nρ = {rho:.2f}")
    ax.set_xlabel("CNEFE count (+1)", fontsize=8)
    if c == cats[0]:
        ax.set_ylabel("OSM count (+1)", fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle("OSM vs CNEFE establishment counts by district — São Paulo (2022)", fontsize=11)
fig.tight_layout()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "25_cnefe_osm_scatter_district.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved figure: {out}")
