# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_02_amenities — SP amenities from OpenStreetMap
# -------------------------------------------------------------------------
"""
Download / load and explore the São Paulo amenity POIs (10 Zhang et al. 2025
categories) from OpenStreetMap.

Primary path (fast, offline, reproducible): pre-filter the local Geofabrik
Sudeste .pbf down to only objects carrying a wanted amenity tag (a ~4 MB file
that pyrosm reads in ~1 s), classify locally, convert to points, clip to the
municipality, cache to sp_amenities.pkl.

Fallback path: Overpass API, one resumable per-category query with checkpoints.

Dependencies: pyrosm, osmium (pyosmium), geopandas, pandas, matplotlib.
            osmnx only needed for the Overpass fallback.
"""
# %%
import os
import time
from collections import defaultdict

import geopandas as gpd
import pandas as pd

import bcw_dis_00_config as cfg

# Source: "pyrosm" (fast, offline) or "overpass" (online fallback).
AMENITIES_SOURCE = "pyrosm"

cfg.AMENITIES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = cfg.AMENITIES_DIR / "by_category"   # Overpass per-category checkpoints
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Classification + extraction helpers
# -------------------------------------------------------------------------
def classify_amenities(gdf_raw):
    """Tag each feature with a single category from cfg.AMENITY_TAGS.

    First category (in AMENITY_TAGS order) to match wins; unmatched dropped.
    """
    gdf_raw = gdf_raw.copy()
    gdf_raw["category"] = pd.NA
    for category, pairs in cfg.AMENITY_TAGS.items():
        for key, value in pairs:
            if key not in gdf_raw.columns:
                continue
            mask = gdf_raw["category"].isna() & (gdf_raw[key] == value)
            gdf_raw.loc[mask, "category"] = category
    return gdf_raw[gdf_raw["category"].notna()].copy()


def build_poi_pbf(src_pbf, out_pbf):
    """Filter `src_pbf` to only objects carrying a wanted amenity tag.

    Uses pyosmium's BackReferenceWriter so kept ways/relations also get the
    nodes needed to reconstruct their geometry. Produces a tiny,
    reference-complete .pbf that pyrosm reads in ~1 s.
    """
    import osmium

    wanted = defaultdict(set)
    for pairs in cfg.AMENITY_TAGS.values():
        for key, value in pairs:
            wanted[key].add(value)
    wanted = dict(wanted)

    def has_wanted_tag(obj):
        t = obj.tags
        for key, vals in wanted.items():
            if key in t and t[key] in vals:
                return True
        return False

    print(f"Building tag-filtered POI extract from {os.path.basename(src_pbf)} "
          "(one-time; ~15-20 min for the full Sudeste file)...")
    t0 = time.time()
    n_kept = 0
    with osmium.BackReferenceWriter(str(out_pbf), ref_src=str(src_pbf), overwrite=True) as writer:
        for obj in osmium.FileProcessor(str(src_pbf)):
            if obj.is_node() and has_wanted_tag(obj):
                writer.add_node(obj); n_kept += 1
            elif obj.is_way() and has_wanted_tag(obj):
                writer.add_way(obj); n_kept += 1
            elif obj.is_relation() and has_wanted_tag(obj):
                writer.add_relation(obj); n_kept += 1
    size_mb = os.path.getsize(out_pbf) / 1e6
    print(f"  kept {n_kept:,} tagged objects -> {out_pbf} ({size_mb:.1f} MB) "
          f"in {time.time() - t0:.0f}s")


def download_amenities_pyrosm(sp_boundary):
    """Fast offline extraction. Returns (raw_gdf, failed=[])."""
    import pyrosm

    # 1) Ensure the tiny tag-filtered POI extract exists.
    if not cfg.PBF_POIS.exists():
        if not cfg.PBF_FULL.exists():
            raise FileNotFoundError(
                f"Regional extract not found: {cfg.PBF_FULL}\n"
                "Download the Geofabrik 'Sudeste' .osm.pbf from "
                "https://download.geofabrik.de/south-america/brazil/sudeste.html"
            )
        build_poi_pbf(cfg.PBF_FULL, cfg.PBF_POIS)

    # 2) Single custom filter covering every tag across all categories.
    custom_filter = defaultdict(set)
    for pairs in cfg.AMENITY_TAGS.values():
        for key, value in pairs:
            custom_filter[key].add(value)
    custom_filter = {k: sorted(v) for k, v in custom_filter.items()}
    keep_cols = sorted(custom_filter)

    # 3) Read the tiny POI extract (fast); bbox + later clip restrict to SP.
    osm = pyrosm.OSM(str(cfg.PBF_POIS), bounding_box=list(sp_boundary.total_bounds))
    print(f"Extracting {sum(len(v) for v in custom_filter.values())} tag values "
          "from POI extract...")
    raw = osm.get_data_by_custom_criteria(
        custom_filter=custom_filter, filter_type="keep", tags_as_columns=keep_cols,
        keep_nodes=True, keep_ways=True, keep_relations=True,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("pyrosm returned no features — check the extract and bbox.")

    raw = raw.to_crs(cfg.CRS_WGS84)
    return classify_amenities(raw), []


def download_amenities_overpass(sp_boundary, bbox_osmnx):
    """Resumable per-category Overpass download. Returns (raw_gdf, failed)."""
    import osmnx as ox
    ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
    ox.settings.requests_timeout = 300

    print("Downloading OSM amenities via Overpass (resumable per category)...")
    all_batches, failed = [], []
    for i, category in enumerate(cfg.CATEGORIES, start=1):
        ckpt = CHECKPOINT_DIR / f"{category}.pkl"
        if ckpt.exists():
            batch = pd.read_pickle(ckpt)
            all_batches.append(batch)
            print(f"  [{i}/{len(cfg.CATEGORIES)}] '{category}' — checkpoint ({len(batch):,}), skip")
            continue
        tags = cfg.category_query_tags[category]
        print(f"  [{i}/{len(cfg.CATEGORIES)}] fetching '{category}'...")
        for attempt in range(1, 4):
            try:
                batch = ox.features_from_bbox(bbox_osmnx, tags)
                batch["category"] = category
                batch.to_pickle(ckpt)
                all_batches.append(batch)
                print(f"      OK — {len(batch):,} features (checkpointed)")
                break
            except Exception as e:
                print(f"      Attempt {attempt} failed: {str(e)[:100]}")
                if attempt == 3:
                    failed.append(category)
                else:
                    time.sleep(5 * attempt)
        time.sleep(2)

    if not all_batches:
        raise RuntimeError("All Overpass queries failed.")
    raw = pd.concat(all_batches, ignore_index=True)
    dup_cols = [c for c in raw.columns if c not in ("geometry", "category")]
    raw = raw.sort_values("category", kind="stable").drop_duplicates(subset=dup_cols, keep="first")
    return raw, failed


# -------------------------------------------------------------------------
# Load (cache-first) + post-process
# -------------------------------------------------------------------------
# %%
sp_boundary = cfg.load_sp_boundary()

if cfg.AMENITIES_PKL.exists():
    print("Loading cached amenities...")
    gdf = pd.read_pickle(cfg.AMENITIES_PKL)
    failed = []
else:
    if AMENITIES_SOURCE == "pyrosm":
        gdf, failed = download_amenities_pyrosm(sp_boundary)
    elif AMENITIES_SOURCE == "overpass":
        gdf, failed = download_amenities_overpass(sp_boundary, cfg.sp_bbox_osmnx())
    else:
        raise ValueError(f"Unknown AMENITIES_SOURCE: {AMENITIES_SOURCE!r}")

    print(f"\nTotal raw (pre-clip): {len(gdf):,}")

    # Polygons -> centroids (project to metric CRS for a correct centroid).
    pts = gdf[gdf.geom_type == "Point"].copy()
    polys = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(polys) > 0:
        polys["geometry"] = polys.geometry.to_crs(cfg.CRS_OD).centroid.to_crs(cfg.CRS_WGS84)
    gdf = pd.concat([pts, polys], ignore_index=True)
    gdf = gdf[gdf.geom_type == "Point"].copy()

    # Clip to the municipality boundary (drops neighbours inside the bbox).
    gdf = gdf.sjoin(sp_boundary[["geometry"]], predicate="within", how="inner")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    print(f"Total after clipping to municipality: {len(gdf):,}")

    if failed:
        print(f"NOT caching — {len(failed)} categories still pending: {failed}")
    else:
        gdf.to_pickle(cfg.AMENITIES_PKL)
        print(f"Saved {len(gdf):,} amenities to: {cfg.AMENITIES_PKL}")


# %% [markdown]
## Data exploration
# %%
print("=" * 60)
print(f"Total amenities: {len(gdf):,}   CRS: {gdf.crs}")
print("=" * 60)

print("\n--- Per-category counts ---")
print(gdf["category"].value_counts())

print("\n--- Geometry types ---")
print(gdf.geom_type.value_counts())

print("\n--- Spatial bounds (W,S,E,N) ---")
print(gdf.total_bounds)

print("\n--- Sample rows ---")
peek = [c for c in ["name", "category", "geometry"] if c in gdf.columns]
print(gdf[peek].head())

# Missing names per category (data-quality signal).
if "name" in gdf.columns:
    print("\n--- Share of unnamed POIs per category ---")
    unnamed = gdf.assign(_unnamed=gdf["name"].isna()).groupby("category")["_unnamed"].mean()
    print((unnamed * 100).round(1).astype(str) + " %")

# %%
# Map: amenities coloured by category over the municipality boundary.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
sp_boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
gdf.plot(ax=ax, column="category", markersize=2, legend=True, alpha=0.5, cmap="tab10")
ax.set_title(f"São Paulo OSM amenities by category (n={len(gdf):,})")
ax.set_axis_off()
cfg.save_fig("02_amenities_by_category", fig)
