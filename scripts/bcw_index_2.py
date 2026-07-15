
# %% [markdown]
## 2.2. Load OSM amenities for SP
# %%

# -------------------------------------------------------------------------
# CONFIGURATION 
# -------------------------------------------------------------------------
CONFIG = {
    # Input: cleaned OD 2023 data 
    "path_clean_od": "../data/clean/od_2023_sp_clean.csv",
    
    # Input: SRTM elevation raster (GeoTIFF) for slope adjustment
    "path_srtm": "/opt/data/cache/documents/srtm/sao_paulo_srtm.tif",
    
    # Input: GTFS feeds for R5 (list of zip files or directories)
    "gtfs_paths": [
        "/opt/data/cache/documents/gtfs/sptrans.zip",
        "/opt/data/cache/documents/gtfs/emtu.zip",
        "/opt/data/cache/documents/gtfs/metro.zip",
        "/opt/data/cache/documents/gtfs/cptm.zip",
    ],
    
    # Output directory
    "output_dir": "/opt/data/cache/documents/output_indexes",
    
    # Parameters
    "max_time_min": 30,              # Raw travel time ceiling
    "walk_speed_kmh": 5.0,           # Commercial walk speed
    "bike_speed_kmh": 15.0,          # Commercial bike speed
    "departure_time": datetime(2023, 5, 17, 7, 30),  # Typical weekday morning peak
}


# -------------------------------------------------------------------------
# AMENITY CATEGORIES — Zhang et al. (2025) mapping to OSM tags
# -------------------------------------------------------------------------
# For each category we list (key, value) tuples used in OSMnx queries.
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
        ("shop", "eletronics"),
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
        ("shop", "laudry"),
        ("shop", "tailor"),
        ("amenity", "bank"),
        ("amenity", "atm"),
        ("amenity", "bureau_de_change"),
        ("amenity", "car_wash"),
        ("amenity", "charging_station"),
        ("amenity", "state_agent"),
        ("amenity", "police"),
        ("amenity", "fire_station"),
        ("amenity", "car_rental"),
        ("office", "insurance"),
        ("office", "lawyer"),
    ],
}

CATEGORIES = list(AMENITY_TAGS.keys())

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

if os.path.exists(pickle_path):
    print("Loading cached amenities...")
    gdf = pd.read_pickle(pickle_path)

else:
    print("Downloading OSM amenities by category (10 small bbox queries)...")
    all_batches = []
    categories = list(category_query_tags.keys())

    for i, category in enumerate(categories, start=1):
        tags = category_query_tags[category]
        n_tags = sum(len(v) for v in tags.values())
        print(f"  [{i}/10] fetching '{category}' ({n_tags} tag values)...")
        
        for attempt in range(1, 4):
            try:
                # CORRECT osmnx 2.x call — bbox is a tuple, tags is positional
                batch_gdf = ox.features_from_bbox(bbox_osmnx, tags)
                
                if len(batch_gdf) > 0:
                    batch_gdf["category"] = category
                    all_batches.append(batch_gdf)
                    print(f"      OK — got {len(batch_gdf):,} features")
                else:
                    print(f"      Warning — 0 features returned")
                break  # success
                
            except Exception as e:
                msg = str(e)[:100]
                print(f"      Attempt {attempt} failed: {msg}")
                if attempt == 3:
                    print(f"      SKIPPING '{category}' after 3 failures.")
                else:
                    time.sleep(5 * attempt)
        
        time.sleep(2)  # polite pause

    if not all_batches:
        raise RuntimeError("All Overpass queries failed — check internet or try another mirror.")

    # Combine
    gdf = pd.concat(all_batches, ignore_index=True)
    print(f"\nTotal raw (pre-clip): {len(gdf):,}")

    # Drop duplicates (same POI matched two categories)
    dup_cols = [c for c in gdf.columns if c not in ("geometry", "category")]
    gdf = gdf.sort_values("category", kind="stable")
    gdf = gdf.drop_duplicates(subset=dup_cols, keep="first")

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

    # Cache
    gdf.to_pickle(pickle_path)
    print(f"Saved {len(gdf):,} classified amenities to: {pickle_path}")

# Sanity check
print(f"\nTotal amenities: {len(gdf):,}")
print(gdf["category"].value_counts())

peek = [c for c in ["name", "category", "geometry"] if c in gdf.columns]
print("\nSample:")
print(gdf[peek].head())

# -------------------------------------------------------------------------
# CARBON WEIGHTS — highest weight wins per category
# -------------------------------------------------------------------------
CARBON_WEIGHTS = {
    "walk":    1.00,
    "bike":    1.00,
    "metro":   0.33,   # Underground
    "train":   0.30,   # CPTM commuter rail
    "bus":     0.01,
}


# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def ensure_output_dir():
    """Create output directory if it does not exist."""
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


def load_clean_od_data(path: str) -> pd.DataFrame:
    """
    Load the pre-cleaned OD 2023 DataFrame.
    """
    df = pd.read_csv(path, low_memory=False)
    # Ensure coordinate columns are float
    for col in ["coord_x_o", "coord_y_o"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Loaded cleaned OD data: {len(df):,} trips")
    return df


def fetch_amenities_from_osm() -> gpd.GeoDataFrame:
    """
    Fetch amenities from OpenStreetMap and classify into the 10 categories.
    Uses a local PBF if available, otherwise downloads via OSMnx.
    Returns a GeoDataFrame with columns: category, osmid, geometry.
    """
    print("Fetching OSM amenities...")
    all_amenities = []
    
    # Use the municipality boundary to clip the query
    if os.path.exists(CONFIG["path_osm_pbf"]):
        # If you have a PBF, use pyrosm or osmnx to extract amenities
        # Here we show osmnx approach with a place query
        print(f"Using OSM PBF path: {CONFIG['path_osm_pbf']}")
        # For PBF files, osmnx supports graph_from_xml but not direct features_from_xml
        # We fall back to a bounding box query using the São Paulo place name
        tags_override = {"amenity": True, "shop": True, "leisure": True, "public_transport": True, "railway": True}
    else:
        tags_override = {"amenity": True, "shop": True, "leisure": True, "public_transport": True, "railway": True}

    # Query OSM for all relevant features in São Paulo municipality
    # This returns a GeoDataFrame with multi-index (osmid, element_type)
    gdf_osm = ox.features.features_from_place(
        "São Paulo, São Paulo, Brazil",
        tags=tags_override
    ).reset_index()

    print(f"Total OSM features retrieved: {len(gdf_osm):,}")

    # Classify each feature into our 10 categories
    for category, tag_list in AMENITY_TAGS.items():
        mask = np.zeros(len(gdf_osm), dtype=bool)
        for key, value in tag_list:
            col = f"{key}"
            if col in gdf_osm.columns:
                # OSMnx returns tags as string or list; handle both
                if value is True:
                    mask |= gdf_osm[col].notna()
                else:
                    # exact match or partial match in semicolon-separated values
                    matches = gdf_osm[col].astype(str).str.contains(
                        rf"(^|;\s*){re.escape(str(value))}(\s*;|$)",
                        na=False,
                        case=False,
                    )
                    mask |= matches
        cat_gdf = gdf_osm[mask].copy()
        cat_gdf["category"] = category
        # Keep only geometry and minimal metadata
        cat_gdf = cat_gdf[["category", "osmid", "geometry"]].copy()
        all_amenities.append(cat_gdf)

    amenities = pd.concat(all_amenities, ignore_index=True)
    amenities = amenities.drop_duplicates(subset=["osmid", "category"])
    # Remove any rows with null geometry
    amenities = amenities[amenities.geometry.notna()].copy()
    # Ensure CRS is WGS84 (EPSG:4326)
    if amenities.crs is None:
        amenities.set_crs("EPSG:4326", inplace=True)
    else:
        amenities = amenities.to_crs("EPSG:4326")

    print(f"Classified amenities: {len(amenities):,} total")
    for cat in CATEGORIES:
        n = len(amenities[amenities["category"] == cat])
        print(f"  {cat}: {n:,}")
    return amenities


def build_walk_bike_network() -> tuple:
    """
    Build the pedestrian and cycling street network from OSM,
    add elevation grades, and compute impedance (travel time per edge).
    
    Returns
    -------
    network : pandana.Network
        Network ready for nearest-POI queries.
    nodes : pd.DataFrame
        Node coordinates and elevations.
    edges : pd.DataFrame
        Edges with 'walk_time' and 'bike_time' columns (in minutes).
    """
    print("Building OSM walk network...")
    if os.path.exists(CONFIG["path_osm_pbf"]):
        G = ox.graph_from_xml(CONFIG["path_osm_pbf"], simplify=True, network_type="walk")
    else:
        G = ox.graph_from_place(
            "São Paulo, São Paulo, Brazil",
            network_type="walk",
            simplify=True,
        )

    # Add elevation from SRTM raster if available
    if os.path.exists(CONFIG["path_srtm"]):
        print("Adding elevation grades from SRTM...")
        G = ox.elevation.add_node_elevations_raster(G, CONFIG["path_srtm"])
        G = ox.elevation.add_edge_grades(G)
    else:
        warnings.warn("SRTM raster not found. Proceeding with flat terrain assumption.")
        for _, _, d in G.edges(data=True):
            d["grade"] = 0.0

    # Convert to nodes/edges DataFrames for Pandana
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=False, fill_edge_geometry=False)

    # Compute length in metres if not present
    if "length" not in edges.columns:
        edges["length"] = edges.geometry.length

    # Grade-based speed factors (simple approach)
    # Walk: 5 km/h baseline, reduced by steep grades
    # Bike: 15 km/h baseline
    edges["grade_abs"] = edges["grade"].abs().fillna(0)

    # Walk time in minutes: length (m) / (speed (m/min) * grade_penalty)
    walk_speed_m_min = CONFIG["walk_speed_kmh"] * 1000 / 60.0  # ~83.33 m/min
    # Tobler-style penalty: reduce speed on steep grades
    edges["walk_penalty"] = 1.0 + 2.5 * edges["grade_abs"]
    edges["walk_time"] = (edges["length"] / (walk_speed_m_min / edges["walk_penalty"])) / 60.0  # hours? No, we want minutes
    # Actually let's keep consistent units:
    edges["walk_time"] = edges["length"] / (walk_speed_m_min / edges["walk_penalty"])  # minutes

    # Bike time in minutes
    bike_speed_m_min = CONFIG["bike_speed_kmh"] * 1000 / 60.0  # 250 m/min
    # Cycling penalty: steeper hills reduce speed more aggressively
    edges["bike_penalty"] = 1.0 + 3.0 * edges["grade_abs"]
    edges["bike_time"] = edges["length"] / (bike_speed_m_min / edges["bike_penalty"])  # minutes

    # Drop edges with zero or negative travel time (data errors)
    edges = edges[(edges["walk_time"] > 0) & (edges["bike_time"] > 0)].copy()

    # Build Pandana network
    # Pandana needs integer node IDs. Use osmid.
    nodes = nodes.reset_index().rename(columns={"osmid": "node_id"})
    edges = edges.reset_index()
    edges = edges.rename(columns={"u": "from", "v": "to"})

    # Ensure node_id is integer
    nodes["node_id"] = nodes["node_id"].astype(np.int64)
    edges["from"] = edges["from"].astype(np.int64)
    edges["to"] = edges["to"].astype(np.int64)

    network = pandana.Network(
        node_x=nodes["x"],
        node_y=nodes["y"],
        edge_from=edges["from"],
        edge_to=edges["to"],
        edge_weights=edges[["walk_time", "bike_time"]],
    )

    print(f"Network ready: {len(nodes):,} nodes, {len(edges):,} edges")
    return network, nodes, edges
def map_origins_to_network_nodes(df: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Snap each trip origin to its nearest network node.
    Adds 'node_id' column to the DataFrame.
    """
    print("Snapping origins to network nodes...")
    # Build a KD-tree or use Pandana's nearest node function
    # Pandana has get_node_ids(x, y)
    df["node_id"] = network.get_node_ids(df["coord_x_o"].values, df["coord_y_o"].values)
    print(f"Origins snapped: {df['node_id'].notna().sum():,} / {len(df):,}")
    return df


def compute_walk_bike_times(
    network: pandana.Network,
    df: pd.DataFrame,
    amenities: gpd.GeoDataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute travel time to the nearest amenity of each category by walk and bike.
    Adds columns: tt_walk_<category>, tt_bike_<category>.
    """
    print("Computing walk and bike travel times...")

    # For each category, set POIs on the Pandana network and query nearest
    for category in CATEGORIES:
        cat_amenities = amenities[amenities["category"] == category].copy()
        if len(cat_amenities) == 0:
            warnings.warn(f"No amenities found for category: {category}")
            df[f"tt_walk_{category}"] = np.nan
            df[f"tt_bike_{category}"] = np.nan
            continue

        # Get node IDs for amenities
        amenity_node_ids = network.get_node_ids(
            cat_amenities.geometry.x.values,
            cat_amenities.geometry.y.values,
        )
        # Remove -1 (no snap)
        amenity_node_ids = amenity_node_ids[amenity_node_ids != -1]

        if len(amenity_node_ids) == 0:
            warnings.warn(f"No amenities snapped to network for category: {category}")
            df[f"tt_walk_{category}"] = np.nan
            df[f"tt_bike_{category}"] = np.nan
            continue

        # Distance threshold: use max_time converted to approximate metres
        # At 5 km/h, 45 min = 3.75 km. We set max_distance slightly above.
        max_dist_m = (CONFIG["max_time_min"] / 60.0) * CONFIG["walk_speed_kmh"] * 1000 * 1.5

        # Set POIs on network
        network.set_pois(
            category,
            maxdist=max_dist_m,
            maxitems=1,
            x_col=cat_amenities.geometry.x.values,
            y_col=cat_amenities.geometry.y.values,
        )

        # Query nearest by walk time
        walk_nearest = network.nearest_pois(
            distance=max_dist_m,
            category=category,
            num_pois=1,
            imp_name="walk_time",
        )
        walk_nearest = walk_nearest[1]  # column for 1st nearest POI

        # Query nearest by bike time
        bike_nearest = network.nearest_pois(
            distance=max_dist_m,
            category=category,
            num_pois=1,
            imp_name="bike_time",
        )
        bike_nearest = bike_nearest[1]

        # Map back to origins by node_id
        node_to_walk_time = walk_nearest.to_dict()
        node_to_bike_time = bike_nearest.to_dict()

        df[f"tt_walk_{category}"] = df["node_id"].map(node_to_walk_time)
        df[f"tt_bike_{category}"] = df["node_id"].map(node_to_bike_time)

        # Convert from whatever unit pandana returned to minutes
        # Since we set edge weights in minutes, the result is already in minutes.
        # Replace >45 min or inf with NaN
        df[f"tt_walk_{category}"] = df[f"tt_walk_{category}"].replace(
            [np.inf, -np.inf], np.nan
        )
        df[f"tt_bike_{category}"] = df[f"tt_bike_{category}"].replace(
            [np.inf, -np.inf], np.nan
        )

        print(f"  {category}: walk median={df[f'tt_walk_{category}'].median():.1f} min, "
              f"bike median={df[f'tt_bike_{category}'].median():.1f} min")

    return df


def compute_pt_times(df: pd.DataFrame, amenities: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute public-transport travel times using R5py.
    Runs three separate mode sets: walk+metro, walk+train, walk+bus.
    For each category, finds the nearest amenity reachable within max_time_min.
    Adds columns: tt_metro_<category>, tt_train_<category>, tt_bus_<category>.
    """
    print("Setting up R5 transport network...")
    try:
        from r5py import TransportNetwork, TravelTimeMatrix, TransportMode
    except ImportError:
        raise ImportError("r5py is required. Install it with: pip install r5py")

    # Verify OSM PBF exists (R5 needs it)
    osm_pbf = CONFIG["path_osm_pbf"]
    if not os.path.exists(osm_pbf):
        raise FileNotFoundError(f"OSM PBF not found: {osm_pbf}. R5 requires a PBF file.")

    gtfs_paths = [p for p in CONFIG["gtfs_paths"] if os.path.exists(p)]
    if len(gtfs_paths) == 0:
        raise FileNotFoundError("No GTFS feeds found. Please check gtfs_paths in CONFIG.")

    transport_network = TransportNetwork(osm_pbf, gtfs_paths)
    print("R5 transport network loaded.")

    # Build origins GeoDataFrame
    origins_gdf = gpd.GeoDataFrame(
        df[["id_pess", "n_viag", "coord_x_o", "coord_y_o"]].copy(),
        geometry=gpd.points_from_xy(df["coord_x_o"], df["coord_y_o"]),
        crs="EPSG:4326",
    )
    origins_gdf["from_id"] = origins_gdf["id_pess"].astype(str) + "_" + origins_gdf["n_viag"].astype(str)

    # R5 mode configurations to run separately
    pt_configs = {
        "metro": [TransportMode.WALK, TransportMode.SUBWAY],
        "train": [TransportMode.WALK, TransportMode.RAIL],
        "bus":   [TransportMode.WALK, TransportMode.BUS],
    }

    max_time_td = timedelta(minutes=CONFIG["max_time_min"])

    for pt_name, modes in pt_configs.items():
        print(f"  Running R5 for mode: {pt_name} ({', '.join(str(m) for m in modes)})")
        for category in CATEGORIES:
            cat_amenities = amenities[amenities["category"] == category].copy()
            if len(cat_amenities) == 0:
                df[f"tt_{pt_name}_{category}"] = np.nan
                continue

            # Build destinations GeoDataFrame
            dests_gdf = gpd.GeoDataFrame(
                cat_amenities[["osmid", "geometry"]].copy(),
                geometry=cat_amenities.geometry,
                crs="EPSG:4326",
            )
            dests_gdf["to_id"] = dests_gdf["osmid"].astype(str)
            dests_gdf = dests_gdf[["to_id", "geometry"]]

            try:
                # Compute travel time matrix
                tt_matrix = TravelTimeMatrix(
                    transport_network,
                    origins=origins_gdf,
                    destinations=dests_gdf,
                    departure=CONFIG["departure_time"],
                    transport_modes=modes,
                    max_time=max_time_td,
                    # speed_walking is in km/h; leave default or match our flat walk speed
                )

                # tt_matrix is a DataFrame with from_id, to_id, travel_time
                # Group by origin, take minimum travel time across all destinations
                min_times = tt_matrix.groupby("from_id")["travel_time"].min()
                # Map back to df
                df[f"tt_{pt_name}_{category}"] = origins_gdf["from_id"].map(min_times)
            except Exception as e:
                warnings.warn(f"R5 failed for {pt_name}/{category}: {e}")
                df[f"tt_{pt_name}_{category}"] = np.nan

            # R5 returns travel_time in minutes (or None if unreachable)
            df[f"tt_{pt_name}_{category}"] = pd.to_numeric(df[f"tt_{pt_name}_{category}"], errors="coerce")
            # R5 uses a sentinel for unreachable; force to NaN if > max_time
            df.loc[df[f"tt_{pt_name}_{category}"] > CONFIG["max_time_min"], f"tt_{pt_name}_{category}"] = np.nan

        print(f"  {pt_name} complete.")

    return df


def apply_carbon_weights_and_compute_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    From raw travel times, compute category scores and final index at 15, 30, 45 minutes.
    
    Logic:
    1. For each mode config and category, if tt <= threshold: score = carbon_weight
    2. Category score = max(score across all mode configs)
    3. Index = mean of 10 category scores
    """
    print("Applying carbon weights and computing indexes...")

    thresholds = [15, 30, 45]
    mode_configs = ["walk", "bike", "metro", "train", "bus"]

    for threshold in thresholds:
        # Build a temporary score matrix: for each mode config and category
        score_matrix = {}  # key: (mode, category), value: 0/1 * weight

        for mode in mode_configs:
            weight = CARBON_WEIGHTS[mode]
            for category in CATEGORIES:
                tt_col = f"tt_{mode}_{category}"
                if tt_col in df.columns:
                    # 1 if reachable within threshold, 0 otherwise
                    reachable = (df[tt_col] <= threshold).astype(float)
                    score_matrix[(mode, category)] = reachable * weight
                else:
                    score_matrix[(mode, category)] = pd.Series(0.0, index=df.index)

        # For each category, take max score across all modes
        for category in CATEGORIES:
            cat_scores = pd.concat([
                score_matrix[(m, category)] for m in mode_configs
            ], axis=1)
            df[f"score_{threshold}_{category}"] = cat_scores.max(axis=1)

        # Final index = mean of 10 category scores
        score_cols = [f"score_{threshold}_{c}" for c in CATEGORIES]
        df[f"pmc_index_{threshold}"] = df[score_cols].mean(axis=1)

        print(f"  Threshold {threshold} min: mean index = {df[f'pmc_index_{threshold}'].mean():.3f}")

    return df


def save_results(df: pd.DataFrame):
    """Save final DataFrame to CSV and Parquet."""
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pmc_indexes_all_thresholds.csv"
    parquet_path = out_dir / "pmc_indexes_all_thresholds.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Parquet: {parquet_path}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import re  # used in fetch_amenities_from_osm
    
    ensure_output_dir()
    
    # 1. Load data
    df = load_clean_od_data(CONFIG["path_clean_od"])
    
    # 2. Fetch amenities from OSM
    amenities = fetch_amenities_from_osm()
    
    # 3. Build walk/bike network with elevation
    global network  # needed for map_origins_to_network_nodes
    network, nodes, edges = build_walk_bike_network()
    
    # 4. Snap origins to network
    df = map_origins_to_network_nodes(df, nodes)
    
    # 5. Compute walk and bike times to nearest amenity per category
    df = compute_walk_bike_times(network, df, amenities, nodes)
    
    # 6. Compute PT times via R5
    df = compute_pt_times(df, amenities)
    
    # 7. Apply carbon weights and derive 15/30/45 min indexes
    df = apply_carbon_weights_and_compute_index(df)
    
    # 8. Save
    save_results(df)
    
    print("\nDone.")








# %%
