# %% [markdown]
# -------------------------------------------------------------------------
## 0. DESCRIPTION
# -------------------------------------------------------------------------
"""
Computes the 15-Minute City Accessibility Index for the São Paulo OD 2023 data.
Stores raw travel times up to 30 minutes, then derives scores for 15, 20, and 30 min.
Preserves one row per individual anchor (no deduplication).

Dependencies:
    pip install: pandas, geopandas, osmnx, pyrosm, osmium, pandana, r5py, rasterio, elevation, gdal
    Java must be installed and on PATH for r5py.
    R5 will download its JAR automatically on first run.
"""
# %% [markdown]
# -------------------------------------------------------------------------
## 1. LOAD LIBRARIES
# -------------------------------------------------------------------------
# %%
import os
import elevation
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import time

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pandana


# %% [markdown]
# -------------------------------------------------------------------------
## 2. CONFIGURATION
# -------------------------------------------------------------------------
# %%
# Create a dict with all amenities tags 
AMENITY_TAGS = {
    "civic_religion": [
        ("amenity", "place_of_worship"), 
        ("amenity", "townhall"),
        ("amenity", "courthouse"), 
        ("amenity", "police"),
        ("amenity", "fire_station"), 
        ("amenity", "post_office"),
        ("amenity", "community_centre"), 
        ("amenity", "social_facility"),
        ("amenity", "public_building"),
    ],
    "culture": [
        ("amenity", "theatre"), 
        ("amenity", "cinema"),
        ("amenity", "arts_centre"), 
        ("amenity", "museum"),
        ("amenity", "library"), 
        ("tourism", "museum"),
        ("tourism", "gallery"),
    ],
    "dining": [
        ("amenity", "restaurant"), 
        ("amenity", "fast_food"),
        ("amenity", "cafe"), 
        ("amenity", "bar"),
        ("amenity", "pub"), 
        ("amenity", "food_court"),
        ("amenity", "biergarten"),
    ],
    "education": [
        ("amenity", "school"), 
        ("amenity", "university"),
        ("amenity", "college"), 
        ("amenity", "kindergarten"),
        ("amenity", "language_school"),
    ],
    "fitness": [
        ("amenity", "gym"), 
        ("amenity", "sports_centre"),
        ("leisure", "fitness_centre"), 
        ("leisure", "sports_centre"),
        ("leisure", "swimming_pool"), 
        ("leisure", "track"),
        ("sport", "swimming"),
    ],
    "groceries": [
        ("shop", "supermarket"), 
        ("shop", "convenience"),
        ("shop", "grocery"), 
        ("shop", "greengrocer"),
        ("shop", "butcher"), 
        ("shop", "bakery"),
        ("shop", "deli"), 
        ("shop", "health_food"),
    ],
    "healthcare": [
        ("amenity", "hospital"), 
        ("amenity", "clinic"),
        ("amenity", "doctors"), 
        ("amenity", "dentist"),
        ("amenity", "pharmacy"), 
        ("healthcare", "hospital"),
        ("healthcare", "clinic"), 
        ("healthcare", "centre"),
    ],
    "transport": [
        ("amenity", "bus_station"), 
        ("amenity", "taxi"),
        ("public_transport", "station"), 
        ("railway", "station"),
        ("railway", "subway_entrance"), 
        ("highway", "bus_stop"),
    ],
    "retail": [
        ("shop", "mall"), 
        ("shop", "department_store"),
        ("shop", "clothes"), 
        ("shop", "electronics"),
        ("shop", "shoes"), 
        ("shop", "general"),
        ("shop", "hardware"), 
        ("shop", "books"),
        ("shop", "toys"), 
        ("shop", "furniture"),
        ("shop", "pet"), 
        ("shop", "florist"),
        ("shop", "stationery"), 
        ("shop", "optician"),
        ("shop", "jewelry"),
    ],
    "services": [
        ("shop", "hairdresser"), 
        ("shop", "beauty"),
        ("shop", "dry_cleaning"), 
        ("shop", "laundry"),
        ("shop", "tailor"), 
        ("amenity", "bank"),
        ("amenity", "atm"), 
        ("amenity", "bureau_de_change"),
        ("amenity", "car_wash"), 
        ("amenity", "charging_station"),
        ("amenity", "state_agent"), 
        ("amenity", "car_rental"),
        ("office", "insurance"), 
        ("office", "lawyer"),
    ],
}

# Build osmnx-friendly tag dicts per category
category_query_tags = {}
tag_to_category = {}

for category, pairs in AMENITY_TAGS.items():
    tags = defaultdict(set)
    for key, value in pairs:
        tags[key].add(value)
        tag_to_category[(key, value)] = category
    # osmnx expects lists, not sets
    category_query_tags[category] = {k: sorted(list(v)) for k, v in tags.items()}

# %% [markdown]
# -------------------------------------------------------------------------
## 3. LOAD DATA 
# -------------------------------------------------------------------------

## 3.1. Load 2023 OD SP cleaned dataset
# %%
od_sp_2023 = pd.read_csv("../data/clean/od_2023_sp_clean.csv", low_memory=False)


# %% [markdown]
## 3.2. Load SP amenities data from OSM
# %%
# Overpass settings
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
ox.settings.requests_timeout = 300

# --- Load São Paulo municipality boundary ---
munis = gpd.read_file("../data/raw/002_Site Metro Mapas_190225/Shape/Municipios_2023.shp")
sp_boundary = munis[munis["NumeroMuni"] == 36].copy()
sp_boundary = sp_boundary.to_crs("EPSG:4326")
bounds = sp_boundary.total_bounds  # (west, south, east, north)

# osmnx 2.x needs bbox as ONE tuple: (north, south, east, west)
bbox_osmnx = (bounds[3], bounds[1], bounds[2], bounds[0])

# Cache setup
cache_dir = os.path.abspath("../outputs/data/cache/documents/osm_amenities")
os.makedirs(cache_dir, exist_ok=True)
pickle_path = os.path.join(cache_dir, "sp_amenities.pkl")

# Per-category checkpoint folder — each successful fetch is saved here
# immediately, so a later failure never forces a re-download of a
# category that already succeeded (only used by the Overpass path).
checkpoint_dir = os.path.join(cache_dir, "by_category")
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Amenities source selection ---
#   "pyrosm"  : FAST + offline. Queries a local Geofabrik .osm.pbf extract.
#               One read, no rate limits, fully reproducible. Recommended.
#   "overpass": online Overpass API (slow, rate-limited). Kept as a fallback;
#               resumable via per-category checkpoints.
AMENITIES_SOURCE = "pyrosm"

# Local Geofabrik extract used by the pyrosm path. The "Sudeste" region
# covers São Paulo state. If the file below is missing it is auto-downloaded.
pbf_region = "Sudeste"
pbf_path = os.path.abspath("../data/raw/osm/sudeste-260619.osm.pbf")

# Tiny tag-filtered extract built once from `pbf_path`. Reading the full
# 850 MB Sudeste file with pyrosm takes ~100 min because it parses every
# node/way in the dense metro region. We instead pre-filter the .pbf down
# to only objects carrying one of our amenity tags (plus the nodes needed
# to build their geometry). That yields a ~4 MB file that pyrosm reads in
# ~1 second. The filter pass itself is a one-time ~15-20 min cost.
poi_pbf_path = os.path.abspath("../data/raw/osm/sp_pois.osm.pbf")


def classify_amenities(gdf_raw):
    """Tag each feature with a single category from tag_to_category.

    A feature can match several tags; the first category in AMENITY_TAGS
    order wins (same precedence as the Overpass path's stable sort).
    Features matching no known tag are dropped.
    """
    gdf_raw = gdf_raw.copy()
    gdf_raw["category"] = pd.NA
    for category, pairs in AMENITY_TAGS.items():
        for key, value in pairs:
            if key not in gdf_raw.columns:
                continue
            mask = gdf_raw["category"].isna() & (gdf_raw[key] == value)
            gdf_raw.loc[mask, "category"] = category
    return gdf_raw[gdf_raw["category"].notna()].copy()


def download_amenities_overpass():
    """Resumable per-category Overpass download. Returns (raw_gdf, failed)."""
    print("Downloading OSM amenities via Overpass (resumable, one bbox query each)...")
    all_batches = []
    failed = []
    categories = list(category_query_tags.keys())

    for i, category in enumerate(categories, start=1):
        ckpt_path = os.path.join(checkpoint_dir, f"{category}.pkl")

        # --- Checkpoint hit: skip the request entirely ---
        if os.path.exists(ckpt_path):
            batch_gdf = pd.read_pickle(ckpt_path)
            all_batches.append(batch_gdf)
            print(f"  [{i}/{len(categories)}] '{category}' — checkpoint found ({len(batch_gdf):,} features), skipping fetch")
            continue

        tags = category_query_tags[category]
        n_tags = sum(len(v) for v in tags.values())
        print(f"  [{i}/{len(categories)}] fetching '{category}' ({n_tags} tag values)...")

        for attempt in range(1, 4):
            try:
                # CORRECT osmnx 2.x call — bbox is a tuple, tags is positional
                batch_gdf = ox.features_from_bbox(bbox_osmnx, tags)

                if len(batch_gdf) > 0:
                    batch_gdf["category"] = category
                    # --- Write checkpoint immediately on success ---
                    batch_gdf.to_pickle(ckpt_path)
                    all_batches.append(batch_gdf)
                    print(f"      OK — got {len(batch_gdf):,} features (checkpointed)")
                else:
                    # Save an empty checkpoint so we don't re-query an
                    # empty category on the next run.
                    batch_gdf["category"] = category
                    batch_gdf.to_pickle(ckpt_path)
                    print(f"      Warning — 0 features returned (checkpointed empty)")
                break  # success

            except Exception as e:
                msg = str(e)[:100]
                print(f"      Attempt {attempt} failed: {msg}")
                if attempt == 3:
                    print(f"      SKIPPING '{category}' after 3 failures (no checkpoint written — will retry on next run).")
                    failed.append(category)
                else:
                    time.sleep(5 * attempt)

        time.sleep(2)  # polite pause

    if failed:
        print(f"\n{len(failed)} categories failed: {failed}")
        print("Re-run the script to retry only the failed categories "
              "(successful ones are loaded from checkpoints).")

    if not all_batches:
        raise RuntimeError("All Overpass queries failed — check internet or try another mirror.")

    raw = pd.concat(all_batches, ignore_index=True)

    # Drop duplicates (same POI matched two categories)
    dup_cols = [c for c in raw.columns if c not in ("geometry", "category")]
    raw = raw.sort_values("category", kind="stable")
    raw = raw.drop_duplicates(subset=dup_cols, keep="first")
    return raw, failed


def build_poi_pbf(src_pbf, out_pbf):
    """Filter `src_pbf` down to only objects carrying a wanted amenity tag.

    Uses pyosmium's BackReferenceWriter so every kept way/relation also
    gets the nodes needed to reconstruct its geometry (those helper nodes
    have their tags stripped). The result is a tiny, reference-complete
    .pbf that pyrosm can read in ~1 second.
    """
    import osmium

    # Wanted (key -> set of values), built from AMENITY_TAGS.
    wanted = defaultdict(set)
    for pairs in AMENITY_TAGS.values():
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
    with osmium.BackReferenceWriter(out_pbf, ref_src=src_pbf, overwrite=True) as writer:
        for obj in osmium.FileProcessor(src_pbf):
            if obj.is_node() and has_wanted_tag(obj):
                writer.add_node(obj); n_kept += 1
            elif obj.is_way() and has_wanted_tag(obj):
                writer.add_way(obj); n_kept += 1
            elif obj.is_relation() and has_wanted_tag(obj):
                writer.add_relation(obj); n_kept += 1
    size_mb = os.path.getsize(out_pbf) / 1e6
    print(f"  kept {n_kept:,} tagged objects -> {out_pbf} ({size_mb:.1f} MB) "
          f"in {time.time() - t0:.0f}s")


def download_amenities_pyrosm():
    """Fast offline extraction from a local Geofabrik .osm.pbf.

    Returns (raw_gdf, failed=[]). On first run it (a) ensures the regional
    extract exists — auto-downloading it if needed — and (b) builds a tiny
    tag-filtered .pbf. Subsequent reads of that tiny file take ~1 second.
    """
    import pyrosm

    # 1) Ensure the tiny tag-filtered POI extract exists.
    if not os.path.exists(poi_pbf_path):
        # Resolve the full regional .pbf — use the local file if present,
        # else let pyrosm download the Geofabrik extract.
        global pbf_path
        if not os.path.exists(pbf_path):
            print(f"Regional extract not found. Downloading Geofabrik '{pbf_region}' via pyrosm "
                  "(one-time, several hundred MB)...")
            try:
                pbf_path = pyrosm.get_data(pbf_region, directory=cache_dir)
                print(f"Extract ready: {pbf_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not auto-download '{pbf_region}' ({e}).\n"
                    "Download it manually from "
                    "https://download.geofabrik.de/south-america/brazil/sudeste.html "
                    f"and save the .osm.pbf to:\n  {pbf_path}"
                )
        build_poi_pbf(pbf_path, poi_pbf_path)

    # 2) Build a single custom filter covering every tag across all categories.
    custom_filter = defaultdict(set)
    for pairs in AMENITY_TAGS.values():
        for key, value in pairs:
            custom_filter[key].add(value)
    custom_filter = {k: sorted(v) for k, v in custom_filter.items()}
    keep_cols = sorted(custom_filter.keys())

    # 3) Read the tiny POI extract (fast). bounding_box + the later sjoin
    #    restrict results to the São Paulo municipality.
    osm = pyrosm.OSM(poi_pbf_path, bounding_box=list(sp_boundary.total_bounds))
    print(f"Extracting {sum(len(v) for v in custom_filter.values())} tag values "
          "from POI extract...")
    raw = osm.get_data_by_custom_criteria(
        custom_filter=custom_filter,
        filter_type="keep",
        tags_as_columns=keep_cols,
        keep_nodes=True,
        keep_ways=True,
        keep_relations=True,
    )

    if raw is None or len(raw) == 0:
        raise RuntimeError("pyrosm returned no features — check the extract and bbox.")

    raw = raw.to_crs("EPSG:4326")
    raw = classify_amenities(raw)
    return raw, []


if os.path.exists(pickle_path):
    print("Loading cached amenities...")
    gdf = pd.read_pickle(pickle_path)

else:
    if AMENITIES_SOURCE == "pyrosm":
        gdf, failed = download_amenities_pyrosm()
    elif AMENITIES_SOURCE == "overpass":
        gdf, failed = download_amenities_overpass()
    else:
        raise ValueError(f"Unknown AMENITIES_SOURCE: {AMENITIES_SOURCE!r}")

    print(f"\nTotal raw (pre-clip): {len(gdf):,}")

    # Convert polygons → centroids
    pts = gdf[gdf.geom_type == "Point"].copy()
    polys = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(polys) > 0:
        polys["geometry"] = polys.geometry.centroid
    gdf = pd.concat([pts, polys], ignore_index=True)
    gdf = gdf[gdf.geom_type == "Point"].copy()

    # Clip to actual municipality boundary (drops neighbours inside bbox)
    gdf = gdf.sjoin(sp_boundary[["geometry"]], predicate="within", how="inner")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")

    print(f"Total after clipping to municipality: {len(gdf):,}")

    # Cache the final combined result ONLY if nothing is still pending.
    # (Overpass may leave some categories failed; pyrosm never does.)
    if failed:
        print(f"NOT writing final {pickle_path} — {len(failed)} categories still pending.")
    else:
        gdf.to_pickle(pickle_path)
        print(f"Saved {len(gdf):,} classified amenities to: {pickle_path}")

# Sanity check
print(f"\nTotal amenities: {len(gdf):,}")
print(gdf["category"].value_counts())

peek = [c for c in ["name", "category", "geometry"] if c in gdf.columns]
print("\nSample:")
print(gdf[peek].head())


# %% [markdown]
## 3.3. Load OSM walk network for SP
# %%
# Set a writable cache folder
walk_dir = os.path.abspath("../outputs/data/cache/documents/osm_walk")
os.makedirs(walk_dir, exist_ok=True)
walk_path = os.path.join(walk_dir, "sao_paulo_walk.graphml")

# Define the study area — São Paulo municipality boundary.
# OSMnx resolves this place name from OpenStreetMap Nominatim.
place_name = "São Paulo, São Paulo, Brazil"

# Download the pedestrian network (footpaths, sidewalks, pedestrian crossings, etc.)
if os.path.exists(walk_path):
    print("Loading cached walk network...")
    G_walk = ox.load_graphml(walk_path)
else:
    print("Downloading OSM walk network for São Paulo (this may take a few minutes)...")
    G_walk = ox.graph_from_place(
        place_name,
        network_type="walk",      # filters for foot-accessible ways
        simplify=True,           # removes degree-2 nodes for cleaner topology
        retain_all=True,         # keeps disconnected components (e.g., isolated paths)
    )
    ox.save_graphml(G_walk, walk_path)
    print(f"Saved walk network to: {walk_path}")

# Convert to GeoDataFrames if you want to inspect or merge later
nodes_walk, edges_walk = ox.graph_to_gdfs(G_walk)

print(f"Walk network: {len(nodes_walk):,} nodes, {len(edges_walk):,} edges")


# %% [markdown]
## 3.4. Load OSM bike network for SP
# %%
# Set a writable cache folder
bike_dir = os.path.abspath("../outputs/data/cache/documents/osm_bike")
os.makedirs(bike_dir, exist_ok=True)
bike_path = os.path.join(bike_dir, "sao_paulo_bike.graphml")

# Download the cycling network.
# OSMnx 'bike' keeps ways with bicycle=yes, cycleways, and roads where cycling is permitted.
# It excludes motorways and other bike-prohibited ways.
if os.path.exists(bike_path):
    print("Loading cached bike network...")
    G_bike = ox.load_graphml(bike_path)
else:
    print("Downloading OSM bike network for São Paulo (this may take a few minutes)...")
    G_bike = ox.graph_from_place(
        place_name,
        network_type="bike",
        simplify=True,
        retain_all=True,
    )
    ox.save_graphml(G_bike, bike_path)
    print(f"Saved bike network to: {bike_path}")

# Convert to GeoDataFrames for later use (e.g., plotting or joining with amenities)
nodes_bike, edges_bike = ox.graph_to_gdfs(G_bike)

print(f"Bike network: {len(nodes_bike):,} nodes, {len(edges_bike):,} edges")


# %% [markdown]
## 3.5. Load GTFS Public Transportation network for SP

### 3.5.1 Load GTFS Metro network for SP
# %%
# Since Metro does not publish an open static GTFS, 
# grab the tube lines and stations directly from OpenStreetMap.
metro_dir = os.path.abspath("../outputs/data/cache/documents/osm_metro")
os.makedirs(metro_dir, exist_ok=True)

# Query subway tracks (railway=subway) and subway stations (station=subway).
# The try/except covers both modern osmnx (features_from_place) and legacy
# versions (geometries_from_place).
try:
    metro_tracks = ox.features_from_place(place_name, tags={"railway": "subway"})
    metro_stations_raw = ox.features_from_place(place_name, tags={"station": "subway"})
except AttributeError:
    metro_tracks = ox.geometries_from_place(place_name, tags={"railway": "subway"})
    metro_stations_raw = ox.geometries_from_place(place_name, tags={"station": "subway"})

# Separate lines from points. Tracks are LineStrings; stations are Points.
metro_lines = metro_tracks[metro_tracks.geom_type.isin(["LineString", "MultiLineString"])].copy()
metro_stations = metro_stations_raw[metro_stations_raw.geom_type == "Point"].copy()

# Reset the MultiIndex so the DataFrame is easier to work with later
metro_lines = metro_lines.reset_index(drop=True)
metro_stations = metro_stations.reset_index(drop=True)

# Cache as pickle — keeps all OSM tag columns without GeoPackage schema issues
metro_lines.to_pickle(os.path.join(metro_dir, "metro_lines.pkl"))
metro_stations.to_pickle(os.path.join(metro_dir, "metro_stations.pkl"))

print(f"Metro lines:    {len(metro_lines):,} features")
print(f"Metro stations: {len(metro_stations):,} features")

if "name" in metro_stations.columns:
    print("\nStation sample:")
    print(metro_stations[["name", "ref"]].drop_duplicates().head())


# %% [markdown]
### 3.5.2 Load GTFS CPTM Train network for SP
# %%
# CPTM also lacks a public static GTFS feed, so we use OSM railway data.
# Note: railway=rail pulls *all* heavy rail (CPTM commuter lines + freight).
# Within São Paulo municipality most passenger rail is CPTM
cptm_dir = os.path.abspath("../outputs/data/cache/documents/osm_cptm")
os.makedirs(cptm_dir, exist_ok=True)

try:
    cptm_tracks = ox.features_from_place(place_name, tags={"railway": "rail"})
    cptm_stations_raw = ox.features_from_place(place_name, tags={"railway": "station"})
except AttributeError:
    cptm_tracks = ox.geometries_from_place(place_name, tags={"railway": "rail"})
    cptm_stations_raw = ox.geometries_from_place(place_name, tags={"railway": "station"})

cptm_lines = cptm_tracks[cptm_tracks.geom_type.isin(["LineString", "MultiLineString"])].copy()
cptm_stations = cptm_stations_raw[cptm_stations_raw.geom_type == "Point"].copy()

cptm_lines = cptm_lines.reset_index(drop=True)
cptm_stations = cptm_stations.reset_index(drop=True)

cptm_lines.to_pickle(os.path.join(cptm_dir, "cptm_lines.pkl"))
cptm_stations.to_pickle(os.path.join(cptm_dir, "cptm_stations.pkl"))

print(f"CPTM lines:    {len(cptm_lines):,} features")
print(f"CPTM stations: {len(cptm_stations):,} features")

if "name" in cptm_stations.columns:
    print("\nStation sample:")
    print(cptm_stations[["name", "ref"]].drop_duplicates().head())


# %% [markdown]
### 3.5.3 Load GTFS Bus network for SP
# %%
# SPTrans offers the municipal bus GTFS via its developer portal.
# However login is needed, so login, download it manually and load it here.

gtfs_bus_dir = os.path.abspath("../outputs/data/cache/documents/gtfs/sptrans_bus")

# Load the core GTFS tables into pandas DataFrames
b_agency     = pd.read_csv(os.path.join(gtfs_bus_dir, "agency.txt"))
b_routes     = pd.read_csv(os.path.join(gtfs_bus_dir, "routes.txt"))
b_trips      = pd.read_csv(os.path.join(gtfs_bus_dir, "trips.txt"))
b_stops      = pd.read_csv(os.path.join(gtfs_bus_dir, "stops.txt"))
b_stop_times = pd.read_csv(os.path.join(gtfs_bus_dir, "stop_times.txt"))
b_shapes     = pd.read_csv(os.path.join(gtfs_bus_dir, "shapes.txt"))

# Quick sanity check of what we loaded
print(f"\nAgency:      {len(b_agency):,} operator(s)")
print(f"Routes:      {len(b_routes):,} routes")
print(f"Trips:       {len(b_trips):,} trips")
print(f"Stops:       {len(b_stops):,} stops")
print(f"Stop times:  {len(b_stop_times):,} records")
print(f"Shapes:  {len(b_shapes):,} records")

# Show a peek at the stops table to confirm latitude/longitude columns exist
print("\nStops sample:")
print(b_stops.head(3))

# Verify coordinate columns are numeric (common parsing issue)
assert pd.api.types.is_numeric_dtype(b_stops["stop_lat"]), "stop_lat must be numeric"
assert pd.api.types.is_numeric_dtype(b_stops["stop_lon"]), "stop_lon must be numeric"


# %% [markdown]
## 3.6. Load SRTM elevation raster (GeoTIFF) for slope adjustment
# %%
# Create the output folder to save srtm
output_dir = os.path.abspath("../outputs/data/cache/documents/srtm")
os.makedirs(output_dir, exist_ok=True)

# Define SP bounds for the tile selection in WGS84
bounds = (-46.90, -24.00, -46.37, -23.35) 
# Define path where the SRTM Digital Elevation Model (DEM) raster will be saved
dem_path = os.path.join(output_dir, "sao_paulo_srtm.tif")

# Only download if the file doesn't exist yet
if not os.path.exists(dem_path):
    print("Downloading SRTM 30 m tiles...")
    elevation.clip(bounds=bounds, output=dem_path, product="SRTM1")
    elevation.clean()
else:
    print(f"Using existing DEM: {dem_path}")
    
# %%
