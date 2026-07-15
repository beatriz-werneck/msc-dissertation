# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_14_index_rich — coverage/count/binary index (sub-type resolved)
# -------------------------------------------------------------------------
"""
Richer 15-Minute City index that resolves each category into its amenity
SUB-TYPES, so a category is no longer "accessible" just because one nearby POI
(e.g. a pharmacy) is reachable.

For every origin it counts the POIs of each sub-type reachable within time rings
[5,10,15,20,25,30] by each mode (walk/bike via pandana; bus/metro/train via
r5py). From this single primitive it derives, per threshold X in {15,20,30}:

  - COVERAGE (primary): per sub-type, score = highest carbon weight among modes
    reaching it within X; per category = mean of its sub-type scores; index =
    mean over the 10 categories. (Carbon-weighted diversity.)
  - BINARY (baseline): category score = best (max) sub-type score = "reachable
    by any mode" weighted by lowest-carbon mode (matches bcw_dis_09).
  - COUNT (intensity): normalised number of reachable POIs (best mode per
    sub-type), per category / its 95th pct.

Per-origin x mode x (sub-type x ring) counts are checkpointed (walk/bike once;
transit per origin-batch) so an overnight run is fully resumable. Finer rings
are stored so a distance-decay (gravity) variant can be derived later with no
re-routing.

Run controls: SAMPLE_N (None = full ~29.5k). Launch under `caffeinate -i`.
"""
# %%
import os
os.environ.setdefault("JAVA_HOME", "/opt/anaconda3/lib/jvm")

import subprocess
import time
import warnings
from datetime import timedelta

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pandana

import bcw_dis_00_config as cfg

# --- Run controls ---
SAMPLE_N = None            # set to None for the full run
RUN_TRANSIT = True
CARBON_WEIGHTED = os.environ.get("CARBON_WEIGHTED", "1") != "0"   # 0 -> all modes weight 1.0
RINGS = [5, 10, 15, 20, 25, 30]      # minutes; finer rings enable gravity later
ORIGIN_BATCH = 2000
PANDANA_MODES = ["walk", "bike"]
TRANSIT_MODES = ["bus", "metro", "train"]

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = cfg.OUTPUT_DIR / "tt_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# 0. Verify caffeinate is holding the system awake (overnight safety)
# -------------------------------------------------------------------------
def check_caffeinate():
    try:
        out = subprocess.run(["pmset", "-g", "assertions"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        print(f"[caffeinate check] could not run pmset ({e}) — cannot verify.")
        return
    lines = [l for l in out.splitlines() if "PreventUserIdleSystemSleep" in l
             and "caffeinate" in l.lower()]
    if lines:
        print("OK  caffeinate is holding PreventUserIdleSystemSleep — safe for an "
              "overnight run (keep the lid OPEN + on power).")
    else:
        print("=" * 72)
        print("WARNING: no caffeinate system-sleep assertion detected!")
        print("  The Mac may idle/clamshell-sleep and PAUSE this run overnight.")
        print("  Relaunch as:  caffeinate -i python bcw_dis_14_index_rich.py")
        print("  (lid OPEN, on power). Continuing in 15 s — Ctrl-C to abort.")
        print("=" * 72)
        time.sleep(15)


check_caffeinate()


# -------------------------------------------------------------------------
# 1. Inputs: origins, amenities + sub-type, sub-type keys
# -------------------------------------------------------------------------
# %%
print("\nLoading inputs...")
trips = pd.read_parquet(cfg.OD_ENRICHED_PARQUET)
trips["classe_economica"] = trips["criteriobr"].astype(float).map(cfg.ECONOMIC_CLASS)
origins = trips[["lat_o", "lon_o"]].dropna().drop_duplicates().reset_index(drop=True)
origins["orig_id"] = origins.index.astype(str)
if SAMPLE_N is not None:
    origins = origins.sample(n=min(SAMPLE_N, len(origins)), random_state=42).reset_index(drop=True)
    origins["orig_id"] = origins.index.astype(str)
SCOPE = f"rich_sample{SAMPLE_N}" if SAMPLE_N else f"rich_full{len(origins)}"
print(f"  origins: {len(origins):,}  | scope: {SCOPE}")

amen = pd.read_pickle(cfg.AMENITIES_PKL)
amen = amen[amen.geometry.notna()].copy()
amen["lon"], amen["lat"] = amen.geometry.x, amen.geometry.y
amen["sub_type"] = pd.NA
for cat, pairs in cfg.AMENITY_TAGS.items():
    cmask = amen["category"] == cat
    for key, value in pairs:
        if key not in amen.columns:
            continue
        m = cmask & amen["sub_type"].isna() & (amen[key] == value)
        amen.loc[m, "sub_type"] = value
amen = amen[amen["sub_type"].notna()].copy()
amen["ck"] = amen["category"] + "::" + amen["sub_type"]      # category::subtype key
amen["amen_id"] = np.arange(len(amen)).astype(str)

KEYS_BY_CAT = {cat: sorted(amen.loc[amen["category"] == cat, "ck"].unique())
               for cat in cfg.CATEGORIES}
ALL_KEYS = [k for cat in cfg.CATEGORIES for k in KEYS_BY_CAT[cat]]
RING_COLS = [f"{k}__r{r}" for k in ALL_KEYS for r in RINGS]
print(f"  sub-type keys: {len(ALL_KEYS)} | ring-columns/mode: {len(RING_COLS)}")


def _save(df, name):
    d = df.copy(); d.index.name = "orig_id"
    d.reset_index().to_parquet(CKPT_DIR / f"{SCOPE}_{name}.parquet", index=False)


def _load(name):
    d = pd.read_parquet(CKPT_DIR / f"{SCOPE}_{name}.parquet").set_index("orig_id")
    return d.reindex(index=origins["orig_id"].values, columns=RING_COLS).fillna(0)


def _exists(name):
    return (CKPT_DIR / f"{SCOPE}_{name}.parquet").exists()


# -------------------------------------------------------------------------
# 2. WALK / BIKE rich counts (pandana set/aggregate per sub-type x ring)
# -------------------------------------------------------------------------
# %%
# Slope-adjusted edge speed from the SIGNED edge grade (+ uphill, - downhill).
#   Walk : Tobler's (1993) hiking function, rescaled so the zero-grade speed
#          equals the adopted flat design speed (WALK_SPEED_KMH).
#   Bike : Parkin & Rotheram (2010) speed regression (Table 2), taken as a
#          RELATIVE effect and applied to the flat design speed (BIKE_SPEED_KMH).
#          Per unit grade: uphill 40.02/6.01 = 6.66 ; downhill 23.79/6.01 = 3.96.
_TOBLER_FLAT = np.exp(-3.5 * 0.05)          # Tobler value at zero grade (normaliser)


def slope_speed_kmh(mode, grade, flat_kmh):
    # Clip gradient to a plausible urban range first: SRTM nodata/outlier pixels
    # can yield extreme grades that (esp. under Tobler) drive speed toward zero,
    # producing near-infinite edge times that make CH preprocessing explode.
    g = np.clip(np.asarray(grade, dtype=float), -0.30, 0.30)
    if mode == "walk":                                          # Tobler (1993)
        v = flat_kmh * np.exp(-3.5 * np.abs(g + 0.05)) / _TOBLER_FLAT
        return np.maximum(v, 0.20 * flat_kmh)                   # floor: very steep = slow walk
    factor = np.where(g >= 0, 1 - 6.66 * g, 1 - 3.96 * g)       # Parkin & Rotheram (2010)
    return flat_kmh * np.clip(factor, 0.20, 2.5)                # floor: dismount/push; cap: braking


def build_pandana_network(graphml_path, mode, speed_kmh):
    G = ox.load_graphml(graphml_path)
    try:
        G = ox.elevation.add_node_elevations_raster(G, str(cfg.SRTM_TIF), cpus=1)
        G = ox.elevation.add_edge_grades(G)
    except Exception as e:
        warnings.warn(f"Elevation failed ({e}); flat terrain.")
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()
    grade = edges["grade"].fillna(0.0) if "grade" in edges.columns else pd.Series(0.0, index=edges.index)
    v_m_per_min = slope_speed_kmh(mode, grade, speed_kmh) * 1000.0 / 60.0
    edges["minutes"] = edges["length"] / v_m_per_min
    edges = edges[edges["minutes"] > 0].copy()
    nodes.index = nodes.index.astype("int64")
    edges["u"] = edges["u"].astype("int64"); edges["v"] = edges["v"].astype("int64")
    net = pandana.Network(nodes["x"], nodes["y"], edges["u"], edges["v"], edges[["minutes"]])
    net.precompute(max(RINGS))
    return net


def pandana_rich(mode, graphml, speed):
    if _exists(mode):
        print(f"  [{mode}] checkpoint found -> loading")
        return _load(mode)
    net = build_pandana_network(graphml, mode, speed)
    o_nodes = np.asarray(net.get_node_ids(origins["lon_o"].values, origins["lat_o"].values))
    data = {}   # build all columns at once (avoids DataFrame fragmentation)
    for ck in ALL_KEYS:
        sub = amen[amen["ck"] == ck]
        nv = pd.Series(1.0, index=np.asarray(
            net.get_node_ids(sub["lon"].values, sub["lat"].values))).groupby(level=0).sum()
        net.set(nv.index.values, variable=nv.values, name="poi")
        for r in RINGS:
            agg = net.aggregate(r, type="sum", decay="flat", imp_name="minutes", name="poi")
            data[f"{ck}__r{r}"] = agg.reindex(o_nodes).to_numpy()
    out = pd.DataFrame(data, index=origins["orig_id"].values)[RING_COLS]
    _save(out, mode)
    print(f"  [{mode}] computed + checkpointed")
    return out


rich = {}
print("\nWalk/bike rich counts (pandana)...")
rich["walk"] = pandana_rich("walk", cfg.WALK_GRAPHML, cfg.WALK_SPEED_KMH)
rich["bike"] = pandana_rich("bike", cfg.BIKE_GRAPHML, cfg.BIKE_SPEED_KMH)


# -------------------------------------------------------------------------
# 3. TRANSIT rich counts (r5py per origin-batch, ring counts per sub-type)
# -------------------------------------------------------------------------
# %%
def transit_batch_counts(chunk, transport_modes):
    """Return (chunk_origins x RING_COLS) reachable POI counts via r5py."""
    from r5py import TravelTimeMatrix
    o_gdf = gpd.GeoDataFrame(
        chunk[["orig_id"]].rename(columns={"orig_id": "id"}),
        geometry=gpd.points_from_xy(chunk["lon_o"], chunk["lat_o"]), crs=cfg.CRS_WGS84)
    d_gdf = gpd.GeoDataFrame(
        amen[["amen_id"]].rename(columns={"amen_id": "id"}),
        geometry=gpd.points_from_xy(amen["lon"], amen["lat"]), crs=cfg.CRS_WGS84)
    ttm = TravelTimeMatrix(TRANSPORT_NETWORK, origins=o_gdf, destinations=d_gdf,
                           departure=cfg.DEPARTURE_TIME, transport_modes=transport_modes,
                           max_time=timedelta(minutes=max(RINGS)))
    ttm = ttm.dropna(subset=["travel_time"]).merge(
        amen[["amen_id", "ck"]], left_on="to_id", right_on="amen_id", how="left")
    data = {}   # build all columns at once (avoids DataFrame fragmentation)
    for r in RINGS:
        g = (ttm[ttm["travel_time"] <= r].groupby(["from_id", "ck"]).size()
             .unstack("ck").reindex(index=chunk["orig_id"].values, columns=ALL_KEYS).fillna(0))
        for ck in ALL_KEYS:
            data[f"{ck}__r{r}"] = g[ck].to_numpy()
    return pd.DataFrame(data, index=chunk["orig_id"].values)[RING_COLS]


def transit_rich(name, transport_modes):
    if _exists(name):
        print(f"  [{name}] checkpoint found -> loading")
        return _load(name)
    bdir = CKPT_DIR / f"{SCOPE}_{name}_batches"
    bdir.mkdir(exist_ok=True)
    n = len(origins); nb = (n + ORIGIN_BATCH - 1) // ORIGIN_BATCH
    parts = []
    for b in range(nb):
        bp = bdir / f"batch_{b:04d}.parquet"
        if bp.exists():
            parts.append(pd.read_parquet(bp)); continue
        chunk = origins.iloc[b * ORIGIN_BATCH:(b + 1) * ORIGIN_BATCH]
        res = transit_batch_counts(chunk, transport_modes)
        res.index.name = "orig_id"; res = res.reset_index()
        res.to_parquet(bp, index=False)
        parts.append(res)
        print(f"    [{name}] batch {b + 1}/{nb} ({min((b + 1) * ORIGIN_BATCH, n):,}/{n:,}) checkpointed")
    out = pd.concat(parts, ignore_index=True).set_index("orig_id")
    out = out.reindex(index=origins["orig_id"].values, columns=RING_COLS).fillna(0)
    _save(out, name)
    return out


if RUN_TRANSIT:
    import shutil
    from r5py import TransportNetwork, TransportMode

    def ensure_zip(folder, zp):
        if not zp.exists():
            shutil.make_archive(str(zp.with_suffix("")), "zip", str(folder))
        return str(zp)

    mode_sets = {"bus": [TransportMode.WALK, TransportMode.BUS],
                 "metro": [TransportMode.WALK, TransportMode.SUBWAY],
                 "train": [TransportMode.WALK, TransportMode.RAIL]}
    if any(not _exists(m) for m in mode_sets):
        print("\nBuilding R5 transport network...")
        feeds = [ensure_zip(cfg.GTFS_BUS_SPTRANS, cfg.GTFS_DIR / "sptrans_bus.zip")]
        if cfg.GTFS_BUS_EMTU.exists():
            feeds.append(ensure_zip(cfg.GTFS_BUS_EMTU, cfg.GTFS_DIR / "emtu_bus.zip"))
        TRANSPORT_NETWORK = TransportNetwork(str(cfg.PBF_SP_CLIP), feeds)
    else:
        print("\nAll transit checkpoints present -> skipping R5 build.")
    for name, modes in mode_sets.items():
        print(f"  routing {name}...")
        rich[name] = transit_rich(name, modes)


# -------------------------------------------------------------------------
# 4. Derive coverage / binary / count indices per threshold
# -------------------------------------------------------------------------
# %%
MODES = PANDANA_MODES + (TRANSIT_MODES if RUN_TRANSIT else [])
W = cfg.CARBON_WEIGHTS if CARBON_WEIGHTED else {m: 1.0 for m in MODES}


def derive(threshold):
    """Return per-origin DataFrame with binary/coverage/count category cols + index."""
    # sub-type carbon score = max over modes of weight * (count>0)
    sub_score = {}
    sub_bestcount = {}
    for ck in ALL_KEYS:
        col = f"{ck}__r{threshold}"
        score = None; best = None
        for m in MODES:
            reach = (rich[m][col] > 0)
            s = reach.astype(float) * W[m]
            score = s if score is None else np.maximum(score, s)
            best = rich[m][col] if best is None else np.maximum(best, rich[m][col])
        sub_score[ck] = score
        sub_bestcount[ck] = best
    out = pd.DataFrame(index=origins["orig_id"].values)
    cov_cols, bin_cols, cnt_cols = [], [], []
    for cat in cfg.CATEGORIES:
        keys = KEYS_BY_CAT[cat]
        ss = pd.concat([sub_score[k] for k in keys], axis=1)
        coverage = ss.mean(axis=1)                       # carbon-weighted diversity
        binary = ss.max(axis=1)                          # best reachable sub-type (baseline)
        total = pd.concat([sub_bestcount[k] for k in keys], axis=1).sum(axis=1)
        p95 = total[total > 0].quantile(0.95) if (total > 0).any() else 1.0
        countn = (total / p95).clip(upper=1.0) if p95 > 0 else total * 0
        out[f"cov_{cat}"] = coverage
        cov_cols.append(f"cov_{cat}"); bin_cols.append(binary.rename(cat)); cnt_cols.append(countn.rename(cat))
    out[f"index_coverage_{threshold}"] = out[cov_cols].mean(axis=1)
    out[f"index_binary_{threshold}"] = pd.concat(bin_cols, axis=1).mean(axis=1)
    out[f"index_count_{threshold}"] = pd.concat(cnt_cols, axis=1).mean(axis=1)
    # per-category coverage: unsuffixed for 15 min (back-compat with mapping
    # scripts), threshold-suffixed (cov_<cat>_20 / _30) for the other thresholds
    if threshold != 15:
        out = out.rename(columns={c: f"{c}_{threshold}" for c in cov_cols})
    return out


print("\nDeriving indices (coverage / binary / count)...")
parts = [derive(t) for t in cfg.THRESHOLDS]
idx = pd.concat(parts, axis=1)
idx = idx.loc[:, ~idx.columns.duplicated()]

for t in cfg.THRESHOLDS:
    print(f"  {t} min:  coverage={idx[f'index_coverage_{t}'].mean():.3f}  "
          f"binary={idx[f'index_binary_{t}'].mean():.3f}  "
          f"count={idx[f'index_count_{t}'].mean():.3f}")


# -------------------------------------------------------------------------
# 5. Map back to trips + save + equity
# -------------------------------------------------------------------------
# %%
res = idx.reset_index(names="orig_id").merge(origins, on="orig_id", how="left")
keep = ["id_pess", "n_viag", "lat_o", "lon_o", "criteriobr", "classe_economica",
        "fe_pess", "fe_via", "motivo_o"]
keep = [c for c in keep if c in trips.columns]
result = trips[keep].merge(res.drop(columns="orig_id"), on=["lat_o", "lon_o"], how="inner")
tag = "_sample" if SAMPLE_N else ""
tag += "" if CARBON_WEIGHTED else "_noweight"
result.to_parquet(cfg.OUTPUT_DIR / f"pmc_index_rich{tag}.parquet", index=False)
result.to_csv(cfg.OUTPUT_DIR / f"pmc_index_rich{tag}.csv", index=False)
print(f"\nSaved {len(result):,} rows -> pmc_index_rich{tag}.parquet")

print("\n=== 15-min coverage index distribution ===")
print(result["index_coverage_15"].describe().round(3))


def wmean(s, w):
    m = s.notna() & w.notna()
    return np.average(s[m], weights=w[m]) if m.any() else np.nan


print("\n=== Mean 15-min index by class (expansion-weighted) ===")
print(f"{'class':<6}{'coverage':>10}{'binary':>9}{'count':>9}")
for cls in ["A", "B1", "B2", "C1", "C2", "D-E"]:
    s = result[result["classe_economica"] == cls]
    if len(s):
        print(f"{cls:<6}{wmean(s['index_coverage_15'], s['fe_via']):>10.3f}"
              f"{wmean(s['index_binary_15'], s['fe_via']):>9.3f}"
              f"{wmean(s['index_count_15'], s['fe_via']):>9.3f}")
