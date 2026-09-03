# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_07_bus_gtfs — Bus GTFS (SPTrans + EMTU)
# -------------------------------------------------------------------------
"""
Load, validate and explore the bus GTFS feeds used for transit routing:
  - SPTrans (municipal bus) — manual download (developer portal login).
  - EMTU (metropolitan bus) — optional; load if present, else flagged to source.

Feeds are processed with gtfs-kit (per the methodology) and checked so they are
ready for r5py in the index step.

Dependencies: gtfs-kit, pandas, geopandas, matplotlib.
"""
# %%
import gtfs_kit as gk
import pandas as pd

import bcw_dis_00_config as cfg

# Feed name -> directory. EMTU is optional.
BUS_FEEDS = {
    "SPTrans": cfg.GTFS_BUS_SPTRANS,
    "EMTU": cfg.GTFS_BUS_EMTU,
}


def load_feed(path):
    """Read a GTFS feed (folder or .zip) with gtfs-kit, distances in km."""
    return gk.read_feed(str(path), dist_units="km")


# %%
feeds = {}
for name, path in BUS_FEEDS.items():
    if path.exists():
        print(f"Loading {name} GTFS from {path} ...")
        feeds[name] = load_feed(path)
    else:
        print(f"[skip] {name} GTFS not found at {path} "
              f"({'source EMTU feed to include metropolitan buses' if name == 'EMTU' else 'required'})")

if not feeds:
    raise FileNotFoundError("No bus GTFS feeds found — at least SPTrans is required.")

# %% [markdown]
## Data exploration
# %%
for name, feed in feeds.items():
    print("=" * 60)
    print(f"FEED: {name}")
    print("=" * 60)

    # Table sizes.
    for tbl in ["agency", "routes", "trips", "stops", "stop_times", "calendar",
                "frequencies", "shapes"]:
        df = getattr(feed, tbl, None)
        n = 0 if df is None else len(df)
        print(f"  {tbl:<12}: {n:,} rows" + ("" if df is not None else "  (absent)"))

    # Service date span (validity window).
    try:
        dates = feed.get_dates()
        if dates:
            print(f"  service dates: {dates[0]} → {dates[-1]} ({len(dates)} days)")
    except Exception as e:
        print(f"  service dates: n/a ({e})")

    # Route-type breakdown (3 = bus).
    if feed.routes is not None and "route_type" in feed.routes.columns:
        print("\n  route_type counts:")
        print(feed.routes["route_type"].value_counts().to_string())

    # Stops sanity (coords numeric, in/around SP).
    stops = feed.stops
    assert pd.api.types.is_numeric_dtype(stops["stop_lat"]), "stop_lat must be numeric"
    assert pd.api.types.is_numeric_dtype(stops["stop_lon"]), "stop_lon must be numeric"
    print(f"\n  stops lat range: {stops['stop_lat'].min():.4f} → {stops['stop_lat'].max():.4f}")
    print(f"  stops lon range: {stops['stop_lon'].min():.4f} → {stops['stop_lon'].max():.4f}")

    # gtfs-kit structural validation (errors only, to keep output short).
    try:
        report = feed.validate()
        errors = report[report["type"] == "error"]
        print(f"\n  gtfs-kit validation: {len(errors)} error(s), "
              f"{len(report) - len(errors)} warning(s)")
        if len(errors):
            print(errors[["message", "table"]].head(10).to_string(index=False))
    except Exception as e:
        print(f"  validation skipped: {e}")

# %%
# Map SPTrans stops over the municipality boundary.
import matplotlib.pyplot as plt

if "SPTrans" in feeds:
    stops = feeds["SPTrans"].stops
    fig, ax = plt.subplots(figsize=(10, 10))
    cfg.load_sp_boundary().boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.scatter(stops["stop_lon"], stops["stop_lat"], s=1, alpha=0.3, color="#d62728")
    ax.set_title(f"SPTrans bus stops (n={len(stops):,})")
    ax.set_axis_off()
    cfg.save_fig("07_sptrans_stops", fig)
