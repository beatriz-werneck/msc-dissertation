# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_06_cptm_network — CPTM commuter rail
# -------------------------------------------------------------------------
"""
São Paulo CPTM commuter-rail network (Lines 7-13).

Two complementary data sources:
  1. OSM geometry (lines + stations) — for mapping / exploration.
  2. GTFS schedules — extracted from the integrated SPTrans feed, which already
     bundles the CPTM lines as route_type 2 (with frequencies). Real schedule,
     so NO synthetic GTFS is needed. The CPTM subset is written to gtfs/cptm.zip
     for use by the index.

Dependencies: osmnx, geopandas, gtfs-kit, pandas, matplotlib.
"""
# %%
import geopandas as gpd
import gtfs_kit as gk
import pandas as pd

import bcw_dis_00_config as cfg

cfg.CPTM_DIR.mkdir(parents=True, exist_ok=True)
CPTM_LINES_PKL = cfg.CPTM_DIR / "cptm_lines.pkl"
CPTM_STATIONS_PKL = cfg.CPTM_DIR / "cptm_stations.pkl"


# -------------------------------------------------------------------------
# 1. OSM geometry (cache-first)
# -------------------------------------------------------------------------
# %%
if CPTM_LINES_PKL.exists() and CPTM_STATIONS_PKL.exists():
    print("Loading cached OSM CPTM geometry...")
    cptm_lines = pd.read_pickle(CPTM_LINES_PKL)
    cptm_stations = pd.read_pickle(CPTM_STATIONS_PKL)
else:
    import osmnx as ox
    # railway=rail pulls all heavy rail; within SP most passenger rail is CPTM.
    print("Downloading OSM CPTM geometry (railway=rail / railway=station)...")
    tracks = ox.features_from_place(cfg.PLACE_NAME, tags={"railway": "rail"})
    stations_raw = ox.features_from_place(cfg.PLACE_NAME, tags={"railway": "station"})
    cptm_lines = tracks[tracks.geom_type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
    cptm_stations = stations_raw[stations_raw.geom_type == "Point"].reset_index(drop=True)
    cptm_lines.to_pickle(CPTM_LINES_PKL)
    cptm_stations.to_pickle(CPTM_STATIONS_PKL)
    print(f"Saved OSM CPTM geometry to {cfg.CPTM_DIR}")


# -------------------------------------------------------------------------
# 2. CPTM GTFS subset (extract from SPTrans feed; route_type 2)
# -------------------------------------------------------------------------
# %%
print("\nExtracting CPTM GTFS subset from SPTrans feed (route_type 2)...")
feed = gk.read_feed(str(cfg.GTFS_BUS_SPTRANS), dist_units="km")
cptm_route_ids = feed.routes.loc[feed.routes["route_type"] == 2, "route_id"].tolist()
cptm_feed = feed.restrict_to_routes(cptm_route_ids)
cptm_feed.to_file(str(cfg.GTFS_CPTM))
print(f"  {len(cptm_route_ids)} CPTM routes -> {cfg.GTFS_CPTM}")


# %% [markdown]
## Data exploration
# %%
print("=" * 60)
print("OSM GEOMETRY")
print("=" * 60)
print(f"CPTM lines:    {len(cptm_lines):,} features   CRS: {cptm_lines.crs}")
print(f"CPTM stations: {len(cptm_stations):,} features CRS: {cptm_stations.crs}")
if "name" in cptm_stations.columns:
    print("\n--- Station name sample ---")
    print(cptm_stations["name"].dropna().drop_duplicates().head(10).to_string(index=False))

print("\n" + "=" * 60)
print("GTFS SCHEDULE (CPTM subset)")
print("=" * 60)
print("Routes:")
print(cptm_feed.routes[["route_id", "route_short_name", "route_long_name"]].to_string(index=False))
print(f"\nTrips: {len(cptm_feed.trips):,} | stop_times: {len(cptm_feed.stop_times):,} | "
      f"stops: {len(cptm_feed.stops):,} | frequencies: "
      f"{0 if cptm_feed.frequencies is None else len(cptm_feed.frequencies):,}")

if cptm_feed.frequencies is not None and len(cptm_feed.frequencies):
    hw = cptm_feed.frequencies["headway_secs"] / 60.0
    print(f"Headways (min): min={hw.min():.1f}, median={hw.median():.1f}, max={hw.max():.1f}")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
cfg.load_sp_boundary().boundary.plot(ax=ax, color="lightgrey", linewidth=0.8)
cptm_lines.plot(ax=ax, color="#8a2be2", linewidth=1.0)
cptm_stations.plot(ax=ax, color="red", markersize=10)
ax.set_title(f"São Paulo CPTM: {len(cptm_lines):,} line features, "
             f"{len(cptm_stations):,} stations")
ax.set_axis_off()
cfg.save_fig("06_cptm_network", fig)
