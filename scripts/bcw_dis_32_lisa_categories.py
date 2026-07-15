# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_32_lisa_categories — LISA (Local Moran's I) per amenity category
# -------------------------------------------------------------------------
"""
Local spatial autocorrelation of the per-category X-Minute accessibility index,
to locate accessibility HOTSPOTS and DEFICIENCY clusters for each of the 10
categories.

Method: the per-origin coverage index (cov_<cat>, 15 min) is aggregated to H3
res-9 hexagons (mean over unique origins). For each category, Local Moran's I is
computed with QUEEN contiguity weights (999 permutations); pseudo p-values are
FDR-adjusted (Benjamini-Hochberg, alpha 0.05). Each hexagon is labelled:
  High-High  = accessibility hotspot (high, surrounded by high)
  Low-Low    = deficiency cluster    (low, surrounded by low)
  High-Low / Low-High = spatial outliers
  (blank)    = not significant
"""
# %%
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon
from libpysal.weights import Queen
from esda.moran import Moran_Local

import bcw_dis_00_config as cfg

H3_RES = 9
ALPHA = 0.05
PERM = 999
SEED = 42
LABELS = {"civic_religion": "Civic & religion", "culture": "Culture", "dining": "Dining",
          "education": "Education", "fitness": "Fitness", "groceries": "Groceries",
          "healthcare": "Healthcare", "transport": "Transport", "retail": "Retail",
          "services": "Services"}
# LISA classes -> colours (viridis ends: high accessibility = yellow, low = purple,
# matching the coverage maps; only HH and LL are drawn)
URBAN_GREY = "0.80"
CLASS_COLOR = {"High-High": plt.cm.viridis(1.0), "Low-Low": plt.cm.viridis(0.10)}
Q_TO_CLASS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def fdr_bh(p, alpha):
    """Benjamini-Hochberg: return boolean mask of significant p-values."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, bool)
    cutoff = p[order][np.max(np.where(below))]
    return p <= cutoff


# -------------------------------------------------------------------------
# 1. Aggregate the per-category index to H3 res-9 hexagons
# -------------------------------------------------------------------------
print("Aggregating index to H3 res-9 ...", flush=True)
df = pd.read_parquet(cfg.OUTPUT_DIR / "pmc_index_rich.parquet")
cov_cols = [f"cov_{c}" for c in cfg.CATEGORIES]
locs = df.dropna(subset=["lat_o", "lon_o"]).drop_duplicates(["lat_o", "lon_o"]).copy()
locs["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(locs["lat_o"], locs["lon_o"])]
agg = locs.groupby("h3")[cov_cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)
print(f"  hexagons: {len(gdf):,}")

# -------------------------------------------------------------------------
# 2. Queen contiguity weights (drop isolated hexagons)
# -------------------------------------------------------------------------
w = Queen.from_dataframe(gdf, silence_warnings=True)
if w.islands:
    print(f"  dropping {len(w.islands)} isolated hexagon(s)")
    gdf = gdf.drop(index=w.islands).reset_index(drop=True)
    w = Queen.from_dataframe(gdf, silence_warnings=True)
w.transform = "r"

# -------------------------------------------------------------------------
# 3. Local Moran's I per category + FDR
# -------------------------------------------------------------------------
print("\n=== LISA cluster counts (FDR-significant) ===")
print(f"{'category':16} {'HH':>5} {'LL':>5} {'HL':>5} {'LH':>5} {'sig%':>6}")
for c in cfg.CATEGORIES:
    y = gdf[f"cov_{c}"].to_numpy(float)
    lm = Moran_Local(y, w, permutations=PERM, seed=SEED)
    sig = fdr_bh(lm.p_sim, ALPHA)
    cls = np.array([Q_TO_CLASS[q] for q in lm.q], dtype=object)
    cls[~sig] = "ns"
    gdf[f"lisa_{c}"] = cls
    n = len(gdf)
    print(f"{c:16} {(cls=='High-High').sum():5d} {(cls=='Low-Low').sum():5d} "
          f"{(cls=='High-Low').sum():5d} {(cls=='Low-High').sum():5d} {sig.mean():6.1%}")

# -------------------------------------------------------------------------
# 4. Cluster maps (2 x 5) + shared legend
# -------------------------------------------------------------------------
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()
fig, axes = plt.subplots(2, 5, figsize=(8.27, 5.83), layout="constrained")
fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.02)
for ax, c in zip(axes.flat, cfg.CATEGORIES):
    ax.patch.set_alpha(0)
    sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.3, zorder=0)
    urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
    sub = gdf[gdf[f"lisa_{c}"].isin(["High-High", "Low-Low"])]      # drop ns + outliers
    if len(sub):
        sub.plot(ax=ax, color=sub[f"lisa_{c}"].map(CLASS_COLOR), edgecolor="none", zorder=2)
    ax.set_title(LABELS[c], fontsize=9)
    ax.set_axis_off()
    ax.margins(0)

handles = [Patch(facecolor=CLASS_COLOR["High-High"], edgecolor="0.4", label="High-High (hotspot)"),
           Patch(facecolor=CLASS_COLOR["Low-Low"], edgecolor="0.4", label="Low-Low (deficiency)"),
           Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Local Moran's I (LISA) by category (15 min)", fontsize=12)

cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
out = cfg.FIG_DIR / "32_lisa_categories_res9_A5.png"
fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
print(f"\nSaved figure: {out}")

gdf.to_file(cfg.OUTPUT_DIR / "lisa_categories_res9.gpkg", driver="GPKG")
print(f"Saved clusters: {cfg.OUTPUT_DIR / 'lisa_categories_res9.gpkg'}")
