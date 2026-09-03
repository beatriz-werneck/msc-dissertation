# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_09_index — 15-Minute City Accessibility Index
# -------------------------------------------------------------------------
"""
Compute the carbon-weighted 15-Minute City Accessibility Index for São Paulo
OD-2023 trip origins (anchors + chained trips, adults 18-65, both ends in SP).

For each origin and each of the 10 amenity categories, we find the travel time
to the nearest amenity by each mode:
    walk, bike            -> pandana on the OSM network (SRTM slope-adjusted)
    bus, metro, train     -> r5py / R5 on real GTFS schedules
A category is "reachable" within threshold X (15/20/30 min) if any mode reaches
it in <= X minutes; the category score is the highest carbon weight among the
reaching modes. The index is the mean score over the 10 categories (in [0,1]).
One index value per origin, retaining socio-economic class for equity analysis.

Run a quick sanity check first with SAMPLE_N set to a few hundred origins, then
set SAMPLE_N = None for the full run.

Dependencies: pandana, osmnx, r5py (Java 21), geopandas, pandas, numpy.
"""
# %%
import os
# r5py needs a modern JVM; point it at the conda-installed JDK 21 BEFORE import.
os.environ.setdefault("JAVA_HOME", "/opt/anaconda3/lib/jvm")

import warnings
from datetime import timedelta

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pandana

import bcw_dis_00_config as cfg

# --- Run controls ---
SAMPLE_N = None         # set to None for the full ~29.5k unique origins
RUN_TRANSIT = True      # set False to compute walk/bike only (no r5py/Java)
GRADE_FACTOR = {"walk": 3.5, "bike": 5.0}   # slope penalty steepness per mode
ORIGIN_BATCH = 2000     # r5py origins per batch = checkpoint granularity for transit

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resilience: each mode's per-origin travel-time matrix is checkpointed to disk
# the moment it is computed (r5py modes batch-by-batch). A crash mid-run loses
# at most one unfinished batch; re-running loads finished work and only computes
# what's missing. The 15/20/30 indexes are derived instantly afterwards, so only
# the routing needs protecting. Checkpoints are keyed by run SCOPE (sample/full
# + origin count), so the sample and full runs never collide.


# -------------------------------------------------------------------------
# 1. LOAD INPUTS (cached artifacts only — no downloads)
# -------------------------------------------------------------------------
# %%
print("Loading inputs...")
# Origins: use the ENRICHED parquet which carries WGS84 lat_o/lon_o.
trips = pd.read_parquet(cfg.OD_ENRICHED_PARQUET)
trips["classe_economica"] = trips["criteriobr"].astype(float).map(cfg.ECONOMIC_CLASS)
print(f"  trips (anchors+chains): {len(trips):,}")

# Unique origins (dedup for routing; map back later).
origins = (
    trips[["lat_o", "lon_o"]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)
origins["orig_id"] = origins.index.astype(str)
if SAMPLE_N is not None:
    origins = origins.sample(n=min(SAMPLE_N, len(origins)), random_state=42).reset_index(drop=True)
    origins["orig_id"] = origins.index.astype(str)
print(f"  unique origins routed: {len(origins):,}"
      + (f" (SAMPLE_N={SAMPLE_N})" if SAMPLE_N else " (full)"))

# --- Checkpoint setup (keyed by run scope so sample/full never collide) ---
SCOPE = f"sample{SAMPLE_N}" if SAMPLE_N else f"full{len(origins)}"
CKPT_DIR = cfg.OUTPUT_DIR / "tt_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _ckpt(name):
    return CKPT_DIR / f"{SCOPE}_{name}.parquet"


def _save_tt(df, name):
    """Persist a travel-time matrix (orig_id x categories) to its checkpoint."""
    d = df.copy()
    d.index.name = "orig_id"
    d.reset_index().to_parquet(_ckpt(name), index=False)


def _load_tt(name):
    """Load a checkpointed matrix, aligned to the current origins/categories."""
    d = pd.read_parquet(_ckpt(name)).set_index("orig_id")
    return d.reindex(index=origins["orig_id"].values, columns=cfg.CATEGORIES)

# Amenities.
amenities = pd.read_pickle(cfg.AMENITIES_PKL)
amenities = amenities[amenities.geometry.notna()].copy()
amenities["amen_id"] = np.arange(len(amenities)).astype(str)
amenities["lon"] = amenities.geometry.x
amenities["lat"] = amenities.geometry.y
print(f"  amenities: {len(amenities):,} across {amenities['category'].nunique()} categories")


# -------------------------------------------------------------------------
# 2. WALK / BIKE  (pandana, SRTM slope-adjusted)
# -------------------------------------------------------------------------
# %%
def build_pandana_network(graphml_path, speed_kmh, grade_factor):
    """Load an OSM graph, add SRTM grades, return a pandana Network whose edge
    impedance is travel time in minutes (slope-adjusted)."""
    G = ox.load_graphml(graphml_path)
    # Elevation + grades from the SRTM raster (graph + raster both WGS84).
    # cpus=1 avoids osmnx's multiprocessing, which on macOS 'spawn' re-imports
    # this (un-guarded) script in child processes.
    try:
        G = ox.elevation.add_node_elevations_raster(G, str(cfg.SRTM_TIF), cpus=1)
        G = ox.elevation.add_edge_grades(G)
    except Exception as e:
        warnings.warn(f"Elevation step failed ({e}); assuming flat terrain.")

    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()   # bring u, v, key out of the MultiIndex
    if "length" not in edges.columns:
        edges["length"] = edges.geometry.length

    grade_abs = edges["grade_abs"].fillna(0) if "grade_abs" in edges.columns else 0.0
    speed_m_min = speed_kmh * 1000 / 60.0
    penalty = 1.0 + grade_factor * grade_abs          # steeper -> slower
    edges["minutes"] = (edges["length"] * penalty) / speed_m_min
    edges = edges[edges["minutes"] > 0].copy()

    # pandana keys nodes by the Series INDEX, which must be the integer osmid
    # that the edges' u/v reference — so keep the osmid index (do NOT reset it).
    nodes.index = nodes.index.astype("int64")
    edges["u"] = edges["u"].astype("int64")
    edges["v"] = edges["v"].astype("int64")
    net = pandana.Network(
        nodes["x"], nodes["y"],
        edges["u"], edges["v"], edges[["minutes"]],
    )
    net.precompute(cfg.MAX_TIME_MIN)
    return net


def nearest_amenity_minutes(net, origins_df):
    """Return DataFrame indexed by orig_id with one column per category: minutes
    to the nearest amenity of that category (NaN if beyond MAX_TIME_MIN)."""
    # get_node_ids returns a Series whose VALUES are the snapped node ids (in
    # origin order). Take a plain array so we can reindex `near` positionally —
    # wrapping it as a Series with a different index would silently NaN it out.
    node_ids = np.asarray(net.get_node_ids(origins_df["lon_o"].values,
                                           origins_df["lat_o"].values))
    out = pd.DataFrame(index=origins_df["orig_id"].values)
    for cat in cfg.CATEGORIES:
        cat_am = amenities[amenities["category"] == cat]
        if len(cat_am) == 0:
            out[cat] = np.nan
            continue
        net.set_pois(cat, cfg.MAX_TIME_MIN, 1, cat_am["lon"].values, cat_am["lat"].values)
        near = net.nearest_pois(cfg.MAX_TIME_MIN, cat, num_pois=1)[1]  # minutes, indexed by node id
        vals = near.reindex(node_ids).to_numpy()
        # pandana fills unreachable nodes with the search radius (MAX_TIME_MIN);
        # treat that as "not reached" so it can't satisfy the 30-min threshold.
        vals = np.where(vals >= cfg.MAX_TIME_MIN, np.nan, vals)
        out[cat] = vals
    return out


def compute_pandana_tt(name, graphml, speed_kmh, grade_factor):
    """Checkpoint-aware walk/bike travel times: load if present, else compute+save."""
    if _ckpt(name).exists():
        print(f"  [{name}] checkpoint found -> loading (skip routing)")
        return _load_tt(name)
    net = build_pandana_network(graphml, speed_kmh, grade_factor)
    tt = nearest_amenity_minutes(net, origins)
    _save_tt(tt, name)
    print(f"  [{name}] computed + checkpointed -> {_ckpt(name).name}")
    return tt


print("\nWalk network (pandana)...")
walk_tt = compute_pandana_tt("walk", cfg.WALK_GRAPHML, cfg.WALK_SPEED_KMH, GRADE_FACTOR["walk"])
print(f"  walk: {(walk_tt <= 15).any(axis=1).mean()*100:.1f}% of origins reach >=1 category <=15min")

print("Bike network (pandana)...")
bike_tt = compute_pandana_tt("bike", cfg.BIKE_GRAPHML, cfg.BIKE_SPEED_KMH, GRADE_FACTOR["bike"])
print(f"  bike: {(bike_tt <= 15).any(axis=1).mean()*100:.1f}% of origins reach >=1 category <=15min")


# -------------------------------------------------------------------------
# 3. TRANSIT  (r5py / R5 on real GTFS — bus, metro, train isolated by mode)
# -------------------------------------------------------------------------
# %%
def transit_nearest_minutes(origins_subset, transport_modes):
    """Per-category minutes to nearest reachable amenity for a SUBSET of origins
    via the given r5py transport_modes. Returns DataFrame indexed by orig_id."""
    from r5py import TravelTimeMatrix

    o_gdf = gpd.GeoDataFrame(
        origins_subset[["orig_id"]].rename(columns={"orig_id": "id"}),
        geometry=gpd.points_from_xy(origins_subset["lon_o"], origins_subset["lat_o"]),
        crs=cfg.CRS_WGS84,
    )
    d_gdf = gpd.GeoDataFrame(
        amenities[["amen_id"]].rename(columns={"amen_id": "id"}),
        geometry=gpd.points_from_xy(amenities["lon"], amenities["lat"]),
        crs=cfg.CRS_WGS84,
    )
    ttm = TravelTimeMatrix(
        TRANSPORT_NETWORK,
        origins=o_gdf,
        destinations=d_gdf,
        departure=cfg.DEPARTURE_TIME,
        transport_modes=transport_modes,
        max_time=timedelta(minutes=cfg.MAX_TIME_MIN),
    )
    # ttm: from_id, to_id, travel_time (minutes); only reachable pairs.
    ttm = ttm.dropna(subset=["travel_time"])
    ttm = ttm.merge(amenities[["amen_id", "category"]],
                    left_on="to_id", right_on="amen_id", how="left")
    nearest = ttm.groupby(["from_id", "category"])["travel_time"].min().unstack("category")
    return nearest.reindex(index=origins_subset["orig_id"].values, columns=cfg.CATEGORIES)


def transit_minutes_checkpointed(name, transport_modes):
    """Checkpoint-aware transit routing. Computes in ORIGIN_BATCH-sized chunks,
    saving each batch immediately; a crash loses at most one batch. Returns the
    full orig_id x category matrix and writes a consolidated mode checkpoint."""
    if _ckpt(name).exists():
        print(f"  [{name}] checkpoint found -> loading (skip routing)")
        return _load_tt(name)

    batch_dir = CKPT_DIR / f"{SCOPE}_{name}_batches"
    batch_dir.mkdir(exist_ok=True)
    n = len(origins)
    n_batches = (n + ORIGIN_BATCH - 1) // ORIGIN_BATCH
    parts = []
    for bidx in range(n_batches):
        bpath = batch_dir / f"batch_{bidx:04d}.parquet"
        if bpath.exists():
            parts.append(pd.read_parquet(bpath))
            continue
        chunk = origins.iloc[bidx * ORIGIN_BATCH:(bidx + 1) * ORIGIN_BATCH]
        res = transit_nearest_minutes(chunk, transport_modes)
        res.index.name = "orig_id"
        res = res.reset_index()
        res.to_parquet(bpath, index=False)   # checkpoint this batch immediately
        parts.append(res)
        print(f"    [{name}] batch {bidx + 1}/{n_batches} "
              f"({min((bidx + 1) * ORIGIN_BATCH, n):,}/{n:,} origins) checkpointed")

    out = pd.concat(parts, ignore_index=True).set_index("orig_id")
    out = out.reindex(index=origins["orig_id"].values, columns=cfg.CATEGORIES)
    _save_tt(out, name)   # consolidated mode checkpoint (batches can now be ignored)
    return out


transit_tt = {}
if RUN_TRANSIT:
    import shutil
    from r5py import TransportNetwork, TransportMode

    def ensure_gtfs_zip(folder, zip_path):
        """r5py needs GTFS as a .zip (txt files at the archive root)."""
        if not zip_path.exists():
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(folder))
        return str(zip_path)

    mode_sets = {
        "bus":   [TransportMode.WALK, TransportMode.BUS],
        "metro": [TransportMode.WALK, TransportMode.SUBWAY],
        "train": [TransportMode.WALK, TransportMode.RAIL],
    }

    # Build the (slow) R5 network only if at least one mode still needs routing.
    if any(not _ckpt(name).exists() for name in mode_sets):
        print("\nBuilding R5 transport network (street PBF + SPTrans GTFS: bus+metro+rail)...")
        # The full SPTrans feed bundles bus (route_type 3) + Metro (1) + CPTM (2);
        # r5py's transport_modes filter isolates each mode at query time.
        gtfs_feeds = [ensure_gtfs_zip(cfg.GTFS_BUS_SPTRANS, cfg.GTFS_DIR / "sptrans_bus.zip")]
        if cfg.GTFS_BUS_EMTU.exists():
            gtfs_feeds.append(ensure_gtfs_zip(cfg.GTFS_BUS_EMTU, cfg.GTFS_DIR / "emtu_bus.zip"))
        TRANSPORT_NETWORK = TransportNetwork(str(cfg.PBF_SP_CLIP), gtfs_feeds)
    else:
        print("\nAll transit checkpoints present -> skipping R5 network build.")

    for name, modes in mode_sets.items():
        print(f"  routing {name} ({'+'.join(str(m).split('.')[-1] for m in modes)})...")
        transit_tt[name] = transit_minutes_checkpointed(name, modes)
        reach = (transit_tt[name] <= 15).any(axis=1).mean() * 100
        print(f"    {name}: {reach:.1f}% of origins reach >=1 category <=15min")
else:
    print("\n[RUN_TRANSIT=False] skipping r5py — walk/bike only.")


# -------------------------------------------------------------------------
# 4. CARBON-WEIGHTED SCORING + INDEX
# -------------------------------------------------------------------------
# %%
# Per-mode minute matrices (rows=orig_id, cols=categories).
mode_minutes = {"walk": walk_tt, "bike": bike_tt, **transit_tt}

# --- Diagnostic: per-category reachability share (<=15 min) by mode ---
# Shows which categories saturate and which mode drives accessibility.
print("\n=== Reachability share within 15 min (fraction of origins) ===")
diag_thr = 15
header = f"{'category':<15}" + "".join(f"{m:>8}" for m in mode_minutes) + f"{'ANY':>8}"
print(header)
any_reach = pd.DataFrame(False, index=walk_tt.index, columns=cfg.CATEGORIES)
for cat in cfg.CATEGORIES:
    row = f"{cat:<15}"
    for mode, mat in mode_minutes.items():
        share = (mat[cat] <= diag_thr).mean()
        row += f"{share*100:>7.0f}%"
        any_reach[cat] = any_reach[cat] | (mat[cat] <= diag_thr).fillna(False)
    row += f"{any_reach[cat].mean()*100:>7.0f}%"
    print(row)

# Nested index variants (each carbon-weighted, then averaged over 10 categories):
#   walk   : pedestrian only  -> strict 15-Minute City test (most discriminating)
#   active : walk + bike      -> Moreno's original active-travel concept
#   all    : walk+bike+transit-> full low-carbon multimodal index (primary)
INDEX_VARIANTS = {
    "walk":   ["walk"],
    "active": ["walk", "bike"],
    "all":    ["walk", "bike", "bus", "metro", "train"],
}


def compute_index(modes, thr):
    """Mean over 10 categories of the max carbon weight among `modes` reaching
    each category within `thr` minutes."""
    cat_scores = []
    for cat in cfg.CATEGORIES:
        per_mode = [(mode_minutes[m][cat] <= thr).astype(float) * cfg.CARBON_WEIGHTS[m]
                    for m in modes if m in mode_minutes]
        cat_scores.append(pd.concat(per_mode, axis=1).max(axis=1).rename(cat))
    return pd.concat(cat_scores, axis=1).mean(axis=1)


print("\nComputing carbon-weighted index (walk / active / all)...")
index_cols = {}
for variant, modes in INDEX_VARIANTS.items():
    for thr in cfg.THRESHOLDS:
        # keep the full multimodal index as plain `index_<thr>` (primary);
        # nested variants get a suffix.
        col = f"index_{thr}" if variant == "all" else f"index_{variant}_{thr}"
        index_cols[col] = compute_index(modes, thr)
        print(f"  {variant:<6} {thr:>2} min: mean = {index_cols[col].mean():.3f}")

index_df = pd.DataFrame(index_cols)
index_df.index.name = "orig_id"
index_df = index_df.reset_index()

# Attach origin coords for the map-back join.
index_df = index_df.merge(origins[["orig_id", "lat_o", "lon_o"]], on="orig_id", how="left")


# -------------------------------------------------------------------------
# 5. MAP BACK TO TRIPS + SAVE
# -------------------------------------------------------------------------
# %%
keep = ["id_pess", "n_viag", "lat_o", "lon_o", "criteriobr", "classe_economica",
        "fe_pess", "fe_via", "motivo_o"]
keep = [c for c in keep if c in trips.columns]
result = trips[keep].merge(
    index_df.drop(columns=["orig_id"]), on=["lat_o", "lon_o"], how="inner",
)
print(f"\nResult rows (trips with an index): {len(result):,}")

out_csv = cfg.OUTPUT_DIR / (f"pmc_index{'_sample' if SAMPLE_N else ''}.csv")
out_parquet = cfg.OUTPUT_DIR / (f"pmc_index{'_sample' if SAMPLE_N else ''}.parquet")
result.to_csv(out_csv, index=False)
result.to_parquet(out_parquet, index=False)
print(f"Saved: {out_csv}")


# -------------------------------------------------------------------------
# 6. EQUITY SUMMARY (expansion-weighted by economic class)
# -------------------------------------------------------------------------
# %%
def weighted_mean(g, value, weight):
    w = g[weight].to_numpy(dtype=float)
    v = g[value].to_numpy(dtype=float)
    m = np.isfinite(v) & np.isfinite(w)
    return np.average(v[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan


VARIANT_15 = {"walk": "index_walk_15", "active": "index_active_15", "all": "index_15"}
print("\n=== Mean 15-min index by economic class (expansion-weighted) ===")
if "classe_economica" in result.columns and "fe_via" in result.columns:
    order = ["A", "B1", "B2", "C1", "C2", "D-E"]
    print(f"{'class':<6}{'walk':>9}{'active':>9}{'all':>9}{'n':>9}")
    for cls in order:
        sub = result[result["classe_economica"] == cls]
        if len(sub):
            vals = [weighted_mean(sub, c, "fe_via") for c in VARIANT_15.values()]
            print(f"{cls:<6}" + "".join(f"{v:>9.3f}" for v in vals) + f"{len(sub):>9,}")

print("\n--- 15-min index distributions (walk / active / all) ---")
print(result[list(VARIANT_15.values())].describe().round(3))
