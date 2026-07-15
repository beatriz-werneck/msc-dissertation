# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_13_intensity_prototype — WALK-only test of richer index metrics
# -------------------------------------------------------------------------
"""
Prototype (walk only, fast, no Java) testing three ways to score each amenity
category, to see how much they de-mask the binary "nearest" index:

  1. BINARY   (baseline)  : 1 if >=1 POI of the category is reachable in X min.
  2. COVERAGE (diversity) : fraction of the category's SUB-TYPES reachable in
                            X min (e.g. healthcare = hospital/clinic/doctors/
                            dentist/pharmacy/centre). Directly fixes "only a
                            pharmacy nearby still counts the whole category".
  3. COUNT    (intensity) : number of POIs reachable in X min, normalised per
                            category (count / 95th pct, clipped to 1).

Counting uses pandana set()/aggregate() (POIs within X network-minutes) on the
SRTM slope-adjusted walk network. Per-origin x sub-type counts are saved as the
first (walk) slice of the future per-mode rich artifact.

Index variant = mean over the 10 categories (walk only -> carbon weight 1.0).
"""
# %%
import warnings

import numpy as np
import osmnx as ox
import pandas as pd
import pandana

import bcw_dis_00_config as cfg

THRESHOLDS = [15, 20, 30]
GRADE_FACTOR_WALK = 3.5
CKPT_DIR = cfg.OUTPUT_DIR / "tt_checkpoints"


# -------------------------------------------------------------------------
# 1. Origins (same deterministic unique set as the index run) + amenities w/ sub-type
# -------------------------------------------------------------------------
# %%
trips = pd.read_parquet(cfg.OD_ENRICHED_PARQUET)
trips["classe_economica"] = trips["criteriobr"].astype(float).map(cfg.ECONOMIC_CLASS)
origins = trips[["lat_o", "lon_o"]].dropna().drop_duplicates().reset_index(drop=True)
origins["orig_id"] = origins.index.astype(str)
print(f"Origins: {len(origins):,}")

amen = pd.read_pickle(cfg.AMENITIES_PKL)
amen = amen[amen.geometry.notna()].copy()
amen["lon"], amen["lat"] = amen.geometry.x, amen.geometry.y

# Canonical sub-type = value of the first matching (key,value) pair in the
# POI's category (merges amenity=hospital & healthcare=hospital -> "hospital").
amen["sub_type"] = pd.NA
for cat, pairs in cfg.AMENITY_TAGS.items():
    cmask = amen["category"] == cat
    for key, value in pairs:
        if key not in amen.columns:
            continue
        m = cmask & amen["sub_type"].isna() & (amen[key] == value)
        amen.loc[m, "sub_type"] = value
amen = amen[amen["sub_type"].notna()].copy()

SUBTYPES = {cat: sorted(amen.loc[amen["category"] == cat, "sub_type"].unique())
            for cat in cfg.CATEGORIES}
print("Sub-types per category:",
      {c: len(s) for c, s in SUBTYPES.items()})


# -------------------------------------------------------------------------
# 2. Build slope-adjusted walk pandana network
# -------------------------------------------------------------------------
# %%
def build_walk_net():
    G = ox.load_graphml(cfg.WALK_GRAPHML)
    try:
        G = ox.elevation.add_node_elevations_raster(G, str(cfg.SRTM_TIF), cpus=1)
        G = ox.elevation.add_edge_grades(G)
    except Exception as e:
        warnings.warn(f"Elevation failed ({e}); flat terrain.")
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()
    grade_abs = edges["grade_abs"].fillna(0) if "grade_abs" in edges.columns else 0.0
    spd = cfg.WALK_SPEED_KMH * 1000 / 60.0
    edges["minutes"] = edges["length"] * (1 + GRADE_FACTOR_WALK * grade_abs) / spd
    edges = edges[edges["minutes"] > 0].copy()
    nodes.index = nodes.index.astype("int64")
    edges["u"] = edges["u"].astype("int64"); edges["v"] = edges["v"].astype("int64")
    net = pandana.Network(nodes["x"], nodes["y"], edges["u"], edges["v"], edges[["minutes"]])
    net.precompute(max(THRESHOLDS))
    return net


print("Building walk network...")
net = build_walk_net()
o_nodes = np.asarray(net.get_node_ids(origins["lon_o"].values, origins["lat_o"].values))


# -------------------------------------------------------------------------
# 3. Count reachable POIs per SUB-TYPE within each threshold (pandana aggregate)
# -------------------------------------------------------------------------
# %%
# counts[thr] -> DataFrame: index=orig_id, columns="cat::subtype" (POI count <= thr min)
counts = {t: pd.DataFrame(index=origins["orig_id"].values) for t in THRESHOLDS}
for cat in cfg.CATEGORIES:
    for st in SUBTYPES[cat]:
        sub = amen[(amen["category"] == cat) & (amen["sub_type"] == st)]
        poi_nodes = net.get_node_ids(sub["lon"].values, sub["lat"].values)
        node_var = pd.Series(1.0, index=np.asarray(poi_nodes)).groupby(level=0).sum()
        net.set(node_var.index.values, variable=node_var.values, name="poi")
        for t in THRESHOLDS:
            agg = net.aggregate(t, type="sum", decay="flat", imp_name="minutes", name="poi")
            counts[t][f"{cat}::{st}"] = agg.reindex(o_nodes).to_numpy()
print("Counted POIs for", sum(len(s) for s in SUBTYPES.values()), "sub-types.")

# Save the walk slice of the rich artifact (per-origin x sub-type counts).
rich = counts[15].copy()
rich.index.name = "orig_id"
rich.reset_index().to_parquet(CKPT_DIR / "rich_walk_subtype_counts_15.parquet", index=False)


# -------------------------------------------------------------------------
# 4. Derive the three category metrics + index variants (15 min)
# -------------------------------------------------------------------------
# %%
def category_metrics(count_df):
    """Return per-origin DataFrames of binary, coverage, count_norm (10 cats)."""
    binary = pd.DataFrame(index=count_df.index)
    coverage = pd.DataFrame(index=count_df.index)
    countn = pd.DataFrame(index=count_df.index)
    for cat in cfg.CATEGORIES:
        cols = [f"{cat}::{st}" for st in SUBTYPES[cat]]
        sub = count_df[cols]
        total = sub.sum(axis=1)
        binary[cat] = (total > 0).astype(float)
        coverage[cat] = (sub > 0).sum(axis=1) / len(cols)
        p95 = total[total > 0].quantile(0.95) if (total > 0).any() else 1.0
        countn[cat] = (total / p95).clip(upper=1.0) if p95 > 0 else 0.0
    return binary, coverage, countn


binary, coverage, countn = category_metrics(counts[15])
idx = pd.DataFrame({
    "index_binary": binary.mean(axis=1),
    "index_coverage": coverage.mean(axis=1),
    "index_count": countn.mean(axis=1),
}, index=binary.index)

print("\n=== Index variant distributions (walk, 15 min) ===")
print(idx.describe().round(3))

# Per-category mean coverage (shows which categories were most over-credited).
print("\n=== Mean per-category: binary vs coverage (walk, 15 min) ===")
cmp = pd.DataFrame({"binary": binary.mean(), "coverage": coverage.mean(),
                    "n_subtypes": [len(SUBTYPES[c]) for c in cfg.CATEGORIES]})
print(cmp.round(3).to_string())


# -------------------------------------------------------------------------
# 5. Equity by economic class (expansion-weighted)
# -------------------------------------------------------------------------
# %%
res = idx.reset_index(names="orig_id").merge(origins, on="orig_id").merge(
    trips[["lat_o", "lon_o", "classe_economica", "fe_via"]].drop_duplicates(["lat_o", "lon_o"]),
    on=["lat_o", "lon_o"], how="left")


def wmean(s, w):
    m = s.notna() & w.notna()
    return np.average(s[m], weights=w[m]) if m.any() else np.nan


print("\n=== Mean index by class (expansion-weighted, walk 15 min) ===")
print(f"{'class':<6}{'binary':>9}{'coverage':>10}{'count':>9}")
for cls in ["A", "B1", "B2", "C1", "C2", "D-E"]:
    s = res[res["classe_economica"] == cls]
    if len(s):
        print(f"{cls:<6}{wmean(s['index_binary'], s['fe_via']):>9.3f}"
              f"{wmean(s['index_coverage'], s['fe_via']):>10.3f}"
              f"{wmean(s['index_count'], s['fe_via']):>9.3f}")

# %% [markdown]
# Read the output as: does coverage/count spread the distribution (higher std,
# lower 25th pct) vs the saturated binary, and does the class gradient widen?
