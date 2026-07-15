# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_26_cnefe_osm_bias — is OSM completeness spatially biased?
# -------------------------------------------------------------------------
"""
Direct test of the hypothesis that OSM is better mapped toward the centre /
wealthier areas, which could make the accessibility gradient a data artefact.

Instead of correlating raw counts (confounded by district size and by CNEFE's
broad "other establishments" bucket), this compares OSM COMPLETENESS relative to
the official register: the ratio OSM / CNEFE per district. Because both numerator
and denominator scale with the true number of establishments, the ratio is a
clean completeness measure that cancels area and population.

Restricted to the two categories that map cleanly between the sources:
EDUCATION (CNEFE species 4) and HEALTHCARE (CNEFE species 5). Commerce/total are
dropped from the bias claim because CNEFE species 6 (offices, warehouses,
industry, workshops) is not comparable to OSM's consumer-facing amenities.

Test: Spearman correlation between the completeness ratio and each district's
distance to the historic centre (Praça da Sé, km zero).
  rho ~ 0            -> OSM uniformly (in)complete: NO spatial bias -> gradient real
  rho clearly < 0    -> ratio falls with distance: OSM better in the centre -> bias
"""
# %%
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

import bcw_dis_00_config as cfg

SECT_GPKG = cfg.OUTPUT_DIR / "cnefe_osm_sector.gpkg"
SE_LONLAT = (-46.6333, -23.5505)          # Praça da Sé (city centre / km zero)
CRS_M = "EPSG:31983"                       # SIRGAS 2000 / UTM 23S (metres), São Paulo
CATS = ["education", "healthcare"]
CNT = [f"{s}_{c}" for s in ("cnefe", "osm") for c in CATS]

# -------------------------------------------------------------------------
# 1. Dissolve sectors -> districts (sum counts, union geometry)
# -------------------------------------------------------------------------
g = gpd.read_file(SECT_GPKG)
dsum = g.groupby(["CD_DIST", "NM_DIST"])[CNT].sum().reset_index()
dgeom = g.dissolve(by="CD_DIST")[["geometry"]].reset_index()
dist = gpd.GeoDataFrame(dgeom.merge(dsum, on="CD_DIST"), geometry="geometry", crs=g.crs)
print(f"districts: {len(dist)}")

# -------------------------------------------------------------------------
# 2. Distance of each district centroid to the centre (Sé)
# -------------------------------------------------------------------------
dm = dist.to_crs(CRS_M)
se = gpd.GeoSeries(gpd.points_from_xy([SE_LONLAT[0]], [SE_LONLAT[1]]),
                   crs=cfg.CRS_WGS84).to_crs(CRS_M).iloc[0]
dist["dist_centre_km"] = dm.geometry.centroid.distance(se) / 1000.0

# -------------------------------------------------------------------------
# 3. Completeness ratios (OSM / CNEFE)
# -------------------------------------------------------------------------
for c in CATS:
    dist[f"ratio_{c}"] = dist[f"osm_{c}"] / dist[f"cnefe_{c}"].replace(0, np.nan)
dist["osm_eduhealth"] = dist[[f"osm_{c}" for c in CATS]].sum(axis=1)
dist["cnefe_eduhealth"] = dist[[f"cnefe_{c}" for c in CATS]].sum(axis=1)
dist["ratio_eduhealth"] = dist["osm_eduhealth"] / dist["cnefe_eduhealth"].replace(0, np.nan)

# -------------------------------------------------------------------------
# 4. Bias test: ratio vs distance to centre
# -------------------------------------------------------------------------
print("\n=== Completeness ratio (OSM/CNEFE) vs distance to centre (Sé) ===")
print(f"{'category':12} {'n':>3}  {'mean_ratio':>10}  {'rho(dist,ratio)':>16}  {'p':>9}")
res = {}
for c in CATS + ["eduhealth"]:
    sub = dist.dropna(subset=[f"ratio_{c}"])
    rho, p = spearmanr(sub["dist_centre_km"], sub[f"ratio_{c}"])
    res[c] = (rho, p)
    print(f"{c:12} {len(sub):3d}  {sub[f'ratio_{c}'].mean():10.3f}  {rho:16.3f}  {p:9.1e}")
print("\n(interpretation: rho ~ 0 => no spatial bias; rho clearly negative => OSM "
      "relatively better in the centre)")

# -------------------------------------------------------------------------
# 5. Save district table
# -------------------------------------------------------------------------
keep = (["CD_DIST", "NM_DIST", "dist_centre_km"] + CNT +
        ["osm_eduhealth", "cnefe_eduhealth", "ratio_education", "ratio_healthcare",
         "ratio_eduhealth"])
dist[keep].to_csv(cfg.OUTPUT_DIR / "cnefe_osm_bias_district.csv", index=False)
dist.to_file(cfg.OUTPUT_DIR / "cnefe_osm_bias_district.gpkg", driver="GPKG")
print(f"\nSaved: cnefe_osm_bias_district.csv/.gpkg -> {cfg.OUTPUT_DIR}")

# -------------------------------------------------------------------------
# 6a. Scatter: completeness ratio vs distance to centre
# -------------------------------------------------------------------------
LAB = {"education": "Education", "healthcare": "Healthcare",
       "eduhealth": "Education + Healthcare"}
cats = CATS + ["eduhealth"]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, c in zip(axes, cats):
    sub = dist.dropna(subset=[f"ratio_{c}"])
    x, y = sub["dist_centre_km"].to_numpy(), sub[f"ratio_{c}"].to_numpy()
    ax.scatter(x, y, s=14, alpha=0.6, color="#2c7fb8", edgecolor="none")
    b, a = np.polyfit(x, y, 1)                       # linear trend
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color="#d95f0e", lw=1.3)
    rho, p = res[c]
    ax.set_title(f"{LAB[c]}\nSpearman ρ = {rho:+.2f} (p = {p:.1e})", fontsize=9)
    ax.set_xlabel("Distance to city centre (km)", fontsize=8)
    if c == cats[0]:
        ax.set_ylabel("Completeness ratio  OSM / CNEFE", fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle("OSM completeness relative to CNEFE, by distance to centre — São Paulo districts",
             fontsize=11)
fig.tight_layout()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out1 = cfg.FIG_DIR / "26_cnefe_osm_ratio_vs_distance.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
print(f"Saved figure: {out1}")

# -------------------------------------------------------------------------
# 6b. Map: completeness ratio (education + healthcare) by district
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 6.4))
vmax = float(np.nanpercentile(dist["ratio_eduhealth"], 95))
dist.plot(ax=ax, column="ratio_eduhealth", cmap="viridis", vmin=0, vmax=vmax,
          edgecolor="0.7", linewidth=0.2)
se_wgs = gpd.GeoSeries(gpd.points_from_xy([SE_LONLAT[0]], [SE_LONLAT[1]]), crs=cfg.CRS_WGS84).to_crs(dist.crs)
se_wgs.plot(ax=ax, color="red", marker="*", markersize=90, zorder=5)
ax.annotate("Centre (Sé)", xy=(se_wgs.iloc[0].x, se_wgs.iloc[0].y),
            xytext=(6, 6), textcoords="offset points", fontsize=8, color="red")
ax.set_axis_off()
ax.set_title("OSM completeness (OSM / CNEFE) — Education + Healthcare\nby district, São Paulo",
             fontsize=11)
sm = ScalarMappable(norm=Normalize(0, vmax), cmap="viridis")
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
cb.set_label("Completeness ratio  OSM / CNEFE", fontsize=9)
fig.tight_layout()
out2 = cfg.FIG_DIR / "26_cnefe_osm_ratio_map.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved figure: {out2}")
