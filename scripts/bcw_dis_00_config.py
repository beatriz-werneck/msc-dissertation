# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_00_config — Shared configuration
# -------------------------------------------------------------------------
"""
Single source of truth for the 15-Minute City pipeline.

Importing this module has NO side effects (no downloads, no file writes) —
it only defines paths, constants, the amenity taxonomy, and small helpers.
Every other `bcw_dis_*` script imports from here so that paths, the SP
boundary, the amenity tags, and the index parameters never drift apart.

Usage:
    from bcw_dis_00_config import (
        CACHE_DIR, AMENITY_TAGS, CATEGORIES, load_sp_boundary, save_fig, ...
    )
"""
# %%
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------------
# 1. PATHS  (resolved from this file, so they work regardless of cwd)
# -------------------------------------------------------------------------
# scripts/ -> 05-Code/
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = OUTPUTS_DIR / "data" / "cache" / "documents"
FIG_DIR = OUTPUTS_DIR / "figures"
OUTPUT_DIR = CACHE_DIR / "output_indexes"          # final index results

# --- Inputs ---
OD_CLEAN_CSV = CLEAN_DIR / "od_2023_sp_clean.csv"
OD_ENRICHED_PARQUET = CLEAN_DIR / "trips_sp_enriched.parquet"   # has lat_o/lon_o (WGS84)
MUNI_SHP = RAW_DIR / "002_Site Metro Mapas_190225" / "Shape" / "Municipios_2023.shp"
# Urban zone of São Paulo (Plano Diretor 2014) — base layer for maps.
URBAN_AREA_SHP = RAW_DIR / "urban_area" / "SHP" / "01A_Zona_Urbana.shp"

# --- OSM PBF extracts (see bcw_dis_02_amenities) ---
OSM_DIR = RAW_DIR / "osm"
PBF_FULL = OSM_DIR / "sudeste-260619.osm.pbf"      # raw Geofabrik Sudeste extract
PBF_SP_CLIP = OSM_DIR / "sp_clip.osm.pbf"          # SP-bbox clip — used as r5py street network
PBF_POIS = OSM_DIR / "sp_pois.osm.pbf"             # tiny tag-filtered POI extract

# --- Cache subfolders (derived/processed artifacts) ---
AMENITIES_DIR = CACHE_DIR / "osm_amenities"
WALK_DIR = CACHE_DIR / "osm_walk"
BIKE_DIR = CACHE_DIR / "osm_bike"
METRO_DIR = CACHE_DIR / "osm_metro"
CPTM_DIR = CACHE_DIR / "osm_cptm"
SRTM_DIR = CACHE_DIR / "srtm"
GTFS_DIR = CACHE_DIR / "gtfs"

# --- Specific cached files ---
AMENITIES_PKL = AMENITIES_DIR / "sp_amenities.pkl"
WALK_GRAPHML = WALK_DIR / "sao_paulo_walk.graphml"
BIKE_GRAPHML = BIKE_DIR / "sao_paulo_bike.graphml"
SRTM_TIF = SRTM_DIR / "sao_paulo_srtm.tif"

# --- GTFS feeds ---
# The SPTrans GTFS is the integrated São Paulo feed: it already bundles the
# Metro (route_type 1, incl. the L15 monorail) and CPTM rail (route_type 2)
# networks WITH frequencies, alongside the municipal buses (route_type 3).
# So metro/CPTM use real schedules — no synthetic GTFS needed.
GTFS_BUS_SPTRANS = GTFS_DIR / "sptrans_bus"        # real SPTrans GTFS (manual download)
GTFS_BUS_EMTU = GTFS_DIR / "emtu_bus"              # real EMTU GTFS (optional; to source)
GTFS_METRO = GTFS_DIR / "metro.zip"                # Metro+monorail subset extracted from SPTrans
GTFS_CPTM = GTFS_DIR / "cptm.zip"                  # CPTM rail subset extracted from SPTrans

# GTFS route_type -> carbon-weight mode bucket (for the index).
ROUTE_TYPE_TO_MODE = {1: "metro", 2: "train", 3: "bus", 0: "bus", 5: "bus"}

# -------------------------------------------------------------------------
# 2. CRS + STUDY AREA
# -------------------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"        # OSM / r5py / lat-lon
CRS_OD = "EPSG:22523"          # OD survey metric coords (co_o_x/co_o_y)
SP_MUNI_CODE = 36              # São Paulo municipality (NumeroMuni)
PLACE_NAME = "São Paulo, São Paulo, Brazil"

# -------------------------------------------------------------------------
# 3. AMENITY TAXONOMY — 10 categories (Zhang et al., 2025)
# -------------------------------------------------------------------------
# (key, value) OSM tag pairs per category. Corrected spelling and de-duplicated
# (police/fire_station belong to civic_religion only, not services).
AMENITY_TAGS = {
    "civic_religion": [
        ("amenity", "place_of_worship"), ("amenity", "townhall"),
        ("amenity", "courthouse"), ("amenity", "police"),
        ("amenity", "fire_station"), ("amenity", "post_office"),
        ("amenity", "community_centre"), ("amenity", "social_facility"),
        ("amenity", "public_building"),
    ],
    "culture": [
        ("amenity", "theatre"), ("amenity", "cinema"),
        ("amenity", "arts_centre"), ("amenity", "museum"),
        ("amenity", "library"), ("tourism", "museum"),
        ("tourism", "gallery"),
    ],
    "dining": [
        ("amenity", "restaurant"), ("amenity", "fast_food"),
        ("amenity", "cafe"), ("amenity", "bar"),
        ("amenity", "pub"), ("amenity", "food_court"),
        ("amenity", "biergarten"),
    ],
    "education": [
        ("amenity", "school"), ("amenity", "university"),
        ("amenity", "college"), ("amenity", "kindergarten"),
        ("amenity", "language_school"),
    ],
    "fitness": [
        ("amenity", "gym"), ("amenity", "sports_centre"),
        ("leisure", "fitness_centre"), ("leisure", "sports_centre"),
        ("leisure", "swimming_pool"), ("leisure", "track"),
        ("sport", "swimming"),
    ],
    "groceries": [
        ("shop", "supermarket"), ("shop", "convenience"),
        ("shop", "grocery"), ("shop", "greengrocer"),
        ("shop", "butcher"), ("shop", "bakery"),
        ("shop", "deli"), ("shop", "health_food"),
    ],
    "healthcare": [
        ("amenity", "hospital"), ("amenity", "clinic"),
        ("amenity", "doctors"), ("amenity", "dentist"),
        ("amenity", "pharmacy"), ("healthcare", "hospital"),
        ("healthcare", "clinic"), ("healthcare", "centre"),
    ],
    "transport": [
        ("amenity", "bus_station"), ("amenity", "taxi"),
        ("public_transport", "station"), ("railway", "station"),
        ("railway", "subway_entrance"), ("highway", "bus_stop"),
    ],
    "retail": [
        ("shop", "mall"), ("shop", "department_store"),
        ("shop", "clothes"), ("shop", "electronics"),
        ("shop", "shoes"), ("shop", "general"),
        ("shop", "hardware"), ("shop", "books"),
        ("shop", "toys"), ("shop", "furniture"),
        ("shop", "pet"), ("shop", "florist"),
        ("shop", "stationery"), ("shop", "optician"),
        ("shop", "jewelry"),
    ],
    "services": [
        ("shop", "hairdresser"), ("shop", "beauty"),
        ("shop", "dry_cleaning"), ("shop", "laundry"),
        ("shop", "tailor"), ("amenity", "bank"),
        ("amenity", "atm"), ("amenity", "bureau_de_change"),
        ("amenity", "car_wash"), ("amenity", "charging_station"),
        ("amenity", "state_agent"), ("amenity", "car_rental"),
        ("office", "insurance"), ("office", "lawyer"),
    ],
}

CATEGORIES = list(AMENITY_TAGS.keys())

# Derived lookups (built once at import).
tag_to_category = {}
category_query_tags = {}
for _category, _pairs in AMENITY_TAGS.items():
    _tags = defaultdict(set)
    for _key, _value in _pairs:
        _tags[_key].add(_value)
        tag_to_category[(_key, _value)] = _category
    # osmnx expects lists, not sets
    category_query_tags[_category] = {k: sorted(v) for k, v in _tags.items()}

# -------------------------------------------------------------------------
# 4. TRIP PURPOSES + SOCIO-ECONOMIC CLASS (OD 2023)
# -------------------------------------------------------------------------
PURPOSE_LABELS = {
    1: "Work in industry", 2: "Work in commerce", 3: "Work in services",
    4: "School/Education", 5: "Shopping", 6: "Doctor/Dentist/Health",
    7: "Recreation/Visits/Leisure", 8: "Residence", 9: "Job search",
    10: "Personal Matters", 11: "Dining",
}
# Long-permanence anchors: residence, work, study.
ANCHOR_PURPOSES = {1, 2, 3, 4, 8}

# criteriobr (Critério Brasil) coding in the OD 2023 microdata.
ECONOMIC_CLASS = {1: "A", 2: "B1", 3: "B2", 4: "C1", 5: "C2", 6: "D-E"}

# -------------------------------------------------------------------------
# 5. INDEX PARAMETERS
# -------------------------------------------------------------------------
THRESHOLDS = [15, 20, 30]          # minutes; 15 is the primary 15-Minute City cutoff
MAX_TIME_MIN = 30                  # raw travel-time ceiling (= max threshold)
WALK_SPEED_KMH = 5.0
BIKE_SPEED_KMH = 15.0
# Typical weekday morning peak. NOTE: the available SPTrans GTFS is valid from
# 2023-10-01, so we use a representative weekday WITHIN the feed's validity
# (Wed 2023-10-18) — the closest schedule to the OD-2023 survey period. Using a
# date outside the feed window yields zero transit service. (Methodological
# caveat to document: schedule period vs. survey period.)
DEPARTURE_TIME = datetime(2023, 10, 18, 7, 30)

# Carbon weights = 1 / (life-cycle gCO2e per passenger-km), capped at 1.0.
# Life-cycle = operational (CPTM 2024) x embodied multiplier (Chester & Horvath
# 2009, ERL 4:024008), which adds vehicle manufacturing + infrastructure +
# fuel/supply chains. This is a change from the earlier operational-only figures
# (bike was 1.00, metro 0.33, train 0.30, bus 0.01).
#
#   mode   operational g   x multiplier        = lifecycle g   weight = 1/g
#   walk   ~0              --                     ~0             1.00
#   bike   5 (mfg only)    -- (ECF 2011/TNO)      5             0.20
#   metro  3.0             x2.5 (rail)            7.5           0.133
#   train  3.3             x2.5 (rail)            8.25          0.121
#   bus    90.6            x1.6 (on-road)         145           0.0069
#
# Verified multipliers (Chester & Horvath 2009): on-road +63% (abstract) /
# 1.4-1.6x (sect. 3.2); rail +155% (abstract) / 1.8-2.5x (sect. 3.2).
# Bike: 5 g/km is bicycle manufacturing+maintenance only (ECF 2011, p.9-10,
# citing TNO 2010 / Eco-invent); infrastructure (bike lanes) is EXCLUDED — both
# because ECF excludes it for lack of data and because cyclists ride mostly on
# roadways whose embodied carbon is already charged to the motorised modes.
#
# SENSITIVITY (document/run as robustness checks):
#   - bus on-road multiplier 1.4-1.6x  -> bus weight 0.0069-0.0079
#   - rail multiplier 1.8-2.5x         -> metro 0.133-0.185, train 0.121-0.168
#   - bike WITH rider food 21 g/km     -> bike weight 0.05
# Caveat: Chester & Horvath multipliers are US-derived (could validate metro/
# CPTM against the Rio de Janeiro Metro LCA, 2016).
CARBON_WEIGHTS = {
    "walk": 1.00,
    "bike": 0.20,     # 5 g/km, manufacturing only (ECF 2011 / TNO 2010)
    "metro": 0.133,   # underground + monorail (Metrô); 3.0 x2.5 rail
    "train": 0.121,   # CPTM commuter rail; 3.3 x2.5 rail
    "bus": 0.0069,    # SPTrans + EMTU; 90.6 x1.6 on-road
}

# Provider/feed -> carbon-weight mode bucket.
MODE_TO_WEIGHT = {
    "walk": "walk", "bike": "bike",
    "metro": "metro", "monorail": "metro",
    "train": "train", "cptm": "train",
    "bus": "bus", "sptrans": "bus", "emtu": "bus",
}

# -------------------------------------------------------------------------
# 6. HELPERS
# -------------------------------------------------------------------------
def load_sp_boundary():
    """Return the São Paulo municipality boundary as a GeoDataFrame (WGS84).

    Replaces the boundary-loading block that was duplicated across scripts.
    """
    import geopandas as gpd
    munis = gpd.read_file(MUNI_SHP)
    sp = munis[munis["NumeroMuni"] == SP_MUNI_CODE].copy()
    return sp.to_crs(CRS_WGS84)


def load_urban_area():
    """Return the São Paulo urban zone (Plano Diretor 2014) as a GeoDataFrame
    (WGS84). Used as a grey base layer so non-urban gaps (parks, reservoirs,
    rural fringe) are distinguishable from urban areas that simply lack anchors."""
    import geopandas as gpd
    return gpd.read_file(URBAN_AREA_SHP).to_crs(CRS_WGS84)


def sp_bbox_osmnx():
    """Return the SP bounding box as the (north, south, east, west) tuple
    that osmnx 2.x expects."""
    west, south, east, north = load_sp_boundary().total_bounds
    return (north, south, east, west)


def save_fig(name, fig=None, dpi=150, show=True):
    """Save a matplotlib figure to FIG_DIR/<name>.png and optionally show it."""
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")
    # Only call show() on an interactive backend (avoids a noisy warning when
    # scripts are run head-less, e.g. MPLBACKEND=Agg).
    if show and not plt.get_backend().lower().endswith("agg"):
        plt.show()
    return path


# Convenience: ensure the common output dirs exist (call explicitly; not on import).
def ensure_dirs():
    for d in (AMENITIES_DIR, WALK_DIR, BIKE_DIR, METRO_DIR, CPTM_DIR,
              SRTM_DIR, GTFS_DIR, FIG_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick self-check (safe: only prints + mkdir).
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")
    print(f"Total tag pairs: {sum(len(v) for v in AMENITY_TAGS.values())}")
    print(f"Thresholds: {THRESHOLDS} | departure: {DEPARTURE_TIME}")
    print(f"OD clean exists: {OD_CLEAN_CSV.exists()} | enriched: {OD_ENRICHED_PARQUET.exists()}")
    print(f"SP PBF clip exists: {PBF_SP_CLIP.exists()} | POIs: {PBF_POIS.exists()}")
