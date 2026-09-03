# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_05_metro_network — Metro (underground + monorail)
# -------------------------------------------------------------------------
"""
São Paulo Metro network (Metrô), including the L15-Prata monorail.

Two complementary data sources:
  1. OSM geometry (lines + stations) — for mapping / exploration.
  2. GTFS schedules — extracted from the integrated SPTrans feed, which already
     bundles the Metro lines as route_type 1 (with frequencies). This is the
     real schedule, so NO synthetic GTFS is needed. The Metro subset is written
     to gtfs/metro.zip for use by the index.

Dependencies: osmnx, geopandas, gtfs-kit, pandas, matplotlib.
"""
# %%
import geopandas as gpd
import gtfs_kit as gk
import pandas as pd

import bcw_dis_00_config as cfg

cfg.METRO_DIR.mkdir(parents=True, exist_ok=True)
METRO_LINES_PKL = cfg.METRO_DIR / "metro_lines.pkl"
METRO_STATIONS_PKL = cfg.METRO_DIR / "metro_stations.pkl"


# -------------------------------------------------------------------------
# 1. OSM geometry (cache-first)
# -------------------------------------------------------------------------
# %%
if METRO_LINES_PKL.exists() and METRO_STATIONS_PKL.exists():
    print("Loading cached OSM metro geometry...")
    metro_lines = pd.read_pickle(METRO_LINES_PKL)
    metro_stations = pd.read_pickle(METRO_STATIONS_PKL)
else:
    import osmnx as ox
    print("Downloading OSM metro geometry (railway=subway / station=subway)...")
    tracks = ox.features_from_place(cfg.PLACE_NAME, tags={"railway": "subway"})
    stations_raw = ox.features_from_place(cfg.PLACE_NAME, tags={"station": "subway"})
    metro_lines = tracks[tracks.geom_type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
    metro_stations = stations_raw[stations_raw.geom_type == "Point"].reset_index(drop=True)
    metro_lines.to_pickle(METRO_LINES_PKL)
    metro_stations.to_pickle(METRO_STATIONS_PKL)
    print(f"Saved OSM metro geometry to {cfg.METRO_DIR}")


# -------------------------------------------------------------------------
# 2. Metro GTFS subset (extract from SPTrans feed; route_type 1)
# -------------------------------------------------------------------------
# %%
print("\nExtracting Metro GTFS subset from SPTrans feed (route_type 1)...")
feed = gk.read_feed(str(cfg.GTFS_BUS_SPTRANS), dist_units="km")
metro_route_ids = feed.routes.loc[feed.routes["route_type"] == 1, "route_id"].tolist()
metro_feed = feed.restrict_to_routes(metro_route_ids)
metro_feed.to_file(str(cfg.GTFS_METRO))
print(f"  {len(metro_route_ids)} metro routes -> {cfg.GTFS_METRO}")


# %% [markdown]
## Data exploration
# %%
print("=" * 60)
print("OSM GEOMETRY")
print("=" * 60)
print(f"Metro lines:    {len(metro_lines):,} features   CRS: {metro_lines.crs}")
print(f"Metro stations: {len(metro_stations):,} features CRS: {metro_stations.crs}")
if "name" in metro_stations.columns:
    print("\n--- Station name sample ---")
    print(metro_stations["name"].dropna().drop_duplicates().head(10).to_string(index=False))

print("\n" + "=" * 60)
print("GTFS SCHEDULE (Metro subset)")
print("=" * 60)
print("Routes:")
print(metro_feed.routes[["route_id", "route_short_name", "route_long_name"]].to_string(index=False))
print(f"\nTrips: {len(metro_feed.trips):,} | stop_times: {len(metro_feed.stop_times):,} | "
      f"stops: {len(metro_feed.stops):,} | frequencies: "
      f"{0 if metro_feed.frequencies is None else len(metro_feed.frequencies):,}")

# Headway summary from frequencies (seconds -> minutes).
if metro_feed.frequencies is not None and len(metro_feed.frequencies):
    hw = metro_feed.frequencies["headway_secs"] / 60.0
    print(f"Headways (min): min={hw.min():.1f}, median={hw.median():.1f}, max={hw.max():.1f}")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
cfg.load_sp_boundary().boundary.plot(ax=ax, color="lightgrey", linewidth=0.8)
metro_lines.plot(ax=ax, color="#0072bc", linewidth=1.5)
metro_stations.plot(ax=ax, color="red", markersize=10)
ax.set_title(f"São Paulo Metro: {len(metro_lines):,} line features, "
             f"{len(metro_stations):,} stations")
ax.set_axis_off()
cfg.save_fig("05_metro_network", fig)
