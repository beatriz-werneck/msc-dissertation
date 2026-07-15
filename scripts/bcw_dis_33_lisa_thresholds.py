# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_33_lisa_thresholds — LISA per category at 15 / 20 / 30 min
# -------------------------------------------------------------------------
"""
Extends the per-category LISA (bcw_dis_32) to the 20- and 30-minute thresholds.
Per-category coverage is only stored for 15 min in pmc_index_rich, so coverage at
each threshold T is re-derived from the mode checkpoints exactly as the index
does: per sub-type, score = max over modes of weight * (count within T > 0);
per category = mean over its sub-types. Hexagon set and weights are identical
across thresholds (they depend only on origin locations), so LISA is computed on
the same Queen graph for each T.

Outputs: cluster maps for 20 and 30 min (same style as fig 32), plus a combined
HH/LL cluster-count table (CSV + PNG) for 15 / 20 / 30 min.
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

H3_RES, ALPHA, PERM, SEED = 9, 0.05, 999, 42
THRESHOLDS = [15, 20, 30]
MODES = ["walk", "bike", "bus", "metro", "train"]
W = cfg.CARBON_WEIGHTS
CK = cfg.OUTPUT_DIR / "tt_checkpoints"
SCOPE = "rich_full29552"
URBAN_GREY = "0.80"
CLASS_COLOR = {"High-High": plt.cm.viridis(1.0), "Low-Low": plt.cm.viridis(0.10)}
Q_TO_CLASS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
LABELS = {"civic_religion": "Civic & religion", "culture": "Culture", "dining": "Dining",
          "education": "Education", "fitness": "Fitness", "groceries": "Groceries",
          "healthcare": "Healthcare", "transport": "Transport", "retail": "Retail",
          "services": "Services"}


def hexpoly(cell):
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def fdr_bh(p, alpha):
    p = np.asarray(p); m = len(p); order = np.argsort(p)
    below = p[order] <= alpha * (np.arange(1, m + 1) / m)
    if not below.any():
        return np.zeros(m, bool)
    return p <= p[order][np.max(np.where(below))]


# -------------------------------------------------------------------------
# 1. Origins + mode checkpoints; derive cov_<cat>_<T> per origin
# -------------------------------------------------------------------------
print("Loading origins + checkpoints ...", flush=True)
trips = pd.read_parquet(cfg.OD_ENRICHED_PARQUET)
origins = trips[["lat_o", "lon_o"]].dropna().drop_duplicates().reset_index(drop=True)
origins["orig_id"] = origins.index.astype(str)
rich = {m: pd.read_parquet(CK / f"{SCOPE}_{m}.parquet").set_index("orig_id")
        .reindex(origins["orig_id"].values).fillna(0) for m in MODES}

cks = sorted({c.rsplit("__r", 1)[0] for c in rich["walk"].columns if "__r" in c})
cats = {}
for ck in cks:
    cats.setdefault(ck.split("::")[0], []).append(ck)
CATEGORIES = [c for c in cfg.CATEGORIES if c in cats]

print("Deriving per-category coverage for each threshold ...", flush=True)
per = origins[["lat_o", "lon_o"]].copy()
for T in THRESHOLDS:
    for cat, cklist in cats.items():
        subs = [np.maximum.reduce([(rich[m][f"{ck}__r{T}"].to_numpy() > 0) * W[m] for m in MODES])
                for ck in cklist]
        per[f"cov_{cat}_{T}"] = np.mean(subs, axis=0)

# -------------------------------------------------------------------------
# 2. Aggregate to H3 res-9 (one hex set) + Queen weights
# -------------------------------------------------------------------------
per["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(per["lat_o"], per["lon_o"])]
val_cols = [f"cov_{c}_{T}" for T in THRESHOLDS for c in CATEGORIES]
agg = per.groupby("h3")[val_cols].mean().reset_index()
agg["geometry"] = agg["h3"].map(hexpoly)
gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=cfg.CRS_WGS84)

w = Queen.from_dataframe(gdf, silence_warnings=True)
if w.islands:
    gdf = gdf.drop(index=w.islands).reset_index(drop=True)
    w = Queen.from_dataframe(gdf, silence_warnings=True)
w.transform = "r"
print(f"  hexagons: {len(gdf):,}")

# -------------------------------------------------------------------------
# 3. LISA per (threshold, category); collect labels + counts
# -------------------------------------------------------------------------
counts = {T: {} for T in THRESHOLDS}
for T in THRESHOLDS:
    for c in CATEGORIES:
        lm = Moran_Local(gdf[f"cov_{c}_{T}"].to_numpy(float), w, permutations=PERM, seed=SEED)
        cls = np.array([Q_TO_CLASS[q] for q in lm.q], dtype=object)
        cls[~fdr_bh(lm.p_sim, ALPHA)] = "ns"
        gdf[f"lisa_{c}_{T}"] = cls
        counts[T][c] = ((cls == "High-High").sum(), (cls == "Low-Low").sum())

# -------------------------------------------------------------------------
# 4. Cluster maps for 20 and 30 min (same style as fig 32)
# -------------------------------------------------------------------------
sp = cfg.load_sp_boundary()
urban = cfg.load_urban_area()
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
for T in [20, 30]:
    fig, axes = plt.subplots(2, 5, figsize=(8.27, 5.83), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.005, hspace=0.02)
    for ax, c in zip(axes.flat, CATEGORIES):
        ax.patch.set_alpha(0)
        sp.plot(ax=ax, color="white", edgecolor="0.5", linewidth=0.3, zorder=0)
        urban.plot(ax=ax, color=URBAN_GREY, edgecolor="none", zorder=1)
        sub = gdf[gdf[f"lisa_{c}_{T}"].isin(["High-High", "Low-Low"])]
        if len(sub):
            sub.plot(ax=ax, color=sub[f"lisa_{c}_{T}"].map(CLASS_COLOR), edgecolor="none", zorder=2)
        ax.set_title(LABELS[c], fontsize=9); ax.set_axis_off(); ax.margins(0)
    handles = [Patch(facecolor=CLASS_COLOR["High-High"], edgecolor="0.4", label="High-High (hotspot)"),
               Patch(facecolor=CLASS_COLOR["Low-Low"], edgecolor="0.4", label="Low-Low (deficiency)"),
               Patch(facecolor=URBAN_GREY, edgecolor="none", label="Urban area")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Local Moran's I (LISA) by category ({T} min)", fontsize=12)
    out = cfg.FIG_DIR / f"33_lisa_categories_res9_{T}min_A5.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved figure: {out}")

# -------------------------------------------------------------------------
# 5. Cluster-count table (CSV + PNG), 15 / 20 / 30 min
# -------------------------------------------------------------------------
rows = []
for c in CATEGORIES:
    rows.append([LABELS[c]] + [counts[T][c][i] for T in THRESHOLDS for i in (0, 1)])
tbl = pd.DataFrame(rows, columns=["Category"] +
                   [f"{T}min {k}" for T in THRESHOLDS for k in ("HH", "LL")])
tbl.to_csv(cfg.OUTPUT_DIR / "lisa_cluster_counts_thresholds.csv", index=False)
print("\n" + tbl.to_string(index=False))

fig, ax = plt.subplots(figsize=(9.5, 3.8)); ax.axis("off")
collabels = ["Category"] + [f"{T} min\n{k}" for T in THRESHOLDS for k in ("HH", "LL")]
t = ax.table(cellText=tbl.values, colLabels=collabels, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
for (r, cc), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("black"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f0f0")
    if cc == 0 and r > 0:
        cell.set_text_props(ha="left")
ax.set_title("LISA cluster counts by category and threshold — High-High (hotspot) / Low-Low (deficiency)",
             fontsize=11, pad=12)
out = cfg.FIG_DIR / "33_lisa_cluster_counts_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
print(f"Saved table: {out}")
