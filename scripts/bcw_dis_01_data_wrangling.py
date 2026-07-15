# %% [markdown]
## 0. Import Libraries

# Pandas for tabular data, geopandas for spatial data
# Pathlib for cross-platform file paths that don't break when you move folders

# %%
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %% [markdown]
## 1. Define Paths
# Define the project root so paths work no matter where you run the script from
# __file__ is the script's location; .parent goes up to scripts/, .parent again to 05-Code/

# %%
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'


# %% [markdown]
## 2. Convert Database from .dbf to .csv
# Read trip microdata file (.dbf is a dBase database format used by legacy GIS/SPSS workflows)

# %%
dbf_path = RAW_DATA_DIR / 'Banco2023_divulgacao_190225.dbf'
od_sp_2023_dbf = gpd.read_file(dbf_path)

csv_path = RAW_DATA_DIR / 'Banco2023_divulgacao_190225.csv'
od_sp_2023_dbf.to_csv(csv_path, index=False)


# %% [markdown]
## 3. Read csv

# %%
od_sp_2023_csv = pd.read_csv(csv_path, low_memory=False)

# Clean names: strip whitespace, lowercase, replace spaces with underscores
# This prevents silent errors from columns like 'MOTIVO_D ' vs 'MOTIVO_D'
od_sp_2023_csv.columns = (
    od_sp_2023_csv.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('/', '_')
)

# Check columns data type in csv
# OD microdata mixes floats (distances, coordinates), ints (codes), and objects (strings)
# Knowing dtypes helps us spot whether ID columns were misread as numbers
print("--- CSV Dtypes ---")
print(od_sp_2023_csv.dtypes)

# Summarise csv data
print(f'Shape: {od_sp_2023_csv.shape}  ({od_sp_2023_csv.shape[0]:,} trips, {od_sp_2023_csv.shape[1]} columns)')
print(f'Memory: {od_sp_2023_csv.memory_usage(deep=True).sum() / 1e6:.1f} MB')

# describe() gives count/mean/std/min/max for numeric columns
# nunique() tells us how many distinct zones / modes / purposes exist
print("\n--- Numeric Summary ---")
print(od_sp_2023_csv.describe())

print("\n--- Cardinality (unique values per column) ---")
print(od_sp_2023_csv.nunique().sort_values(ascending=False).head(15))


# %% 
# Visualise csv data

# First few rows
od_sp_2023_csv.head(10)

# %%
# All column names — compare against the data dictionary PDF
print(od_sp_2023_csv.columns.tolist())

# %% 
# Set a clean default style for all plots in this session
sns.set_theme(style='whitegrid')

# Select only numeric columns for plotting — the OD file has many categorical string codes
numeric_cols = od_sp_2023_csv.select_dtypes(include=['number']).columns.tolist()

# Drop coordinate/ID columns that aren't meaningful distributions
# Adjust this list after you run the dtype check above if your column names differ
cols_to_skip = ['coord_x_o', 'coord_y_o', 'coord_x_d', 'coord_y_d', 'zonao', 'zonad']
plot_cols = [c for c in numeric_cols if c not in cols_to_skip]

# Distribution of numeric variables
# Histograms let us spot skewed travel times, extreme distances, or zero-inflation
# We cap at 9 subplots (3×3) to keep the figure readable; add more later if needed
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

for ax, col in zip(axes, plot_cols[:9]):
    sns.histplot(od_sp_2023_csv[col], bins=50, ax=ax, color='steelblue')
    ax.set_title(col)
    ax.set_xlabel('')

# Remove empty subplots if we have fewer than 9 columns to plot
for ax in axes[len(plot_cols[:9]):]:
    ax.set_visible(False)

plt.suptitle('Distribution of Numeric Variables (Trip Microdata)', y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap (numeric columns only)
# This shows which numeric attributes move together — e.g., trip duration vs. distance
# We keep only a subset of interpretable columns so the heatmap isn't crowded
heatmap_cols = [c for c in plot_cols if c in plot_cols[:12]]
corr = od_sp_2023_csv[heatmap_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Correlation Heatmap (Numeric Trip Variables)')
plt.tight_layout()
plt.show()


# %% [markdown]
## 4. Wrangling csv

# %%
# Create the clean output folder if it doesn't exist 
CLEAN_DATA_DIR = BASE_DIR / 'data' / 'clean'
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Filter to trips where BOTH origin AND destination are in São Paulo municipality
# Municipality code 36 = São Paulo city proper. This excludes trips from/to the metro region.
df = od_sp_2023_csv.copy()
df = df[(df['muni_o'] == 36) & (df['muni_d'] == 36)].copy()
print(f"Rows after muni_o == 36 AND muni_d == 36: {len(df):,}")

# Keep only working-age passengers (18 to 65 inclusive)
df = df[(df['idade'] >= 18) & (df['idade'] <= 65)].copy()
print(f"Rows after age filter (18–65): {len(df):,}")

# %%
# Filter anchors and include chained trips 
# Anchors = long-permanence activities at the origin (residence, work, study).
# OD SP 2023 purpose codes: 1)Work in industry, 2)Work in commerce, 3)Work in services, 4)School/Education, 8)Residence
# Also keep intermediate trips that are linked between two anchors (include chaining trips)

ANCHOR_PURPOSES = {1, 2, 3, 4, 8}

# The OD survey uses a person ID and a trip sequence number to order each person's day.
person_id_col = 'id_pess'
trip_seq_col  = 'n_viag'

# Legend mapping for MOTIVO_O (and MOTIVO_D)
purpose_labels = {
    1:  "Work in industry",
    2:  "Work in commerce",
    3:  "Work in services",
    4:  "School/Education",
    5:  "Shopping",
    6:  "Doctor/Dentist/Health",
    7:  "Recreation/Visits/Leisure",
    8:  "Residence",
    9:  "Job search",
    10: "Personal Matters",
    11: "Dining"
}

# --- NEW: Anchor-only count (no trip chaining) for comparison ---
anchor_only = df[df['motivo_o'].isin(ANCHOR_PURPOSES)].copy()
print(f"\nRows if we keep ONLY anchor-origin trips (no chaining): {len(anchor_only):,}")
print(f"Difference vs. chained filter will be: {len(anchor_only) - len(df):,} trips recovered by chaining")


# %%
# --- Build filtered dataframe manually (avoids groupby.apply() KeyError) ---
pre_filter_count = len(df)
filtered_chunks = []

for person_id, person_trips in df.groupby(person_id_col, sort=False):
    # sort this person's trips by trip sequence number
    person_trips = person_trips.sort_values(trip_seq_col).reset_index(drop=True)
    n = len(person_trips)
    keep = np.zeros(n, dtype=bool)
    drop_reason = [''] * n  # NEW: label each row with why it was excluded
    
    # Rule A: keep all trips that originate at an anchor activity
    is_anchor_o = person_trips['motivo_o'].isin(ANCHOR_PURPOSES).values
    keep[is_anchor_o] = True
    
    # Label anchor-origin trips immediately so we know they were kept by Rule A
    for i in np.where(is_anchor_o)[0]:
        drop_reason[i] = 'kept_anchor_origin'
    
    # Rule B: walk through the day and mark completed chains
    in_chain = False
    buffer = []
    
    for i in range(n):
        if is_anchor_o[i]:
            # Leaving an anchor starts a potential chain
            in_chain = True
            buffer = [i]
        elif in_chain:
            buffer.append(i)
            if person_trips['motivo_d'].iloc[i] in ANCHOR_PURPOSES:
                # Chain closes at an anchor destination — keep every trip in between
                for idx in buffer:
                    keep[idx] = True
                    if drop_reason[idx] == '':
                        drop_reason[idx] = 'kept_chain_interior'
                in_chain = False
                buffer = []
        # else: trip is standalone and not an anchor origin → will be labelled later
    
    # Any trips still left in an open buffer at the end of the day mean the chain
    # started with an anchor origin but never reached an anchor destination.
    # Those trips are dropped (except the anchor origin trip itself, which Rule A already saved).
    if in_chain:
        for idx in buffer:
            if not keep[idx]:
                drop_reason[idx] = 'after_last_anchor_no_close'
    
# Everything else that wasn't kept and has no label fell before the first anchor
    # or is a completely standalone non-anchor trip.
    for i in range(n):
        if not keep[i] and drop_reason[i] == '':
            drop_reason[i] = 'before_first_anchor_or_standalone'
    
    person_trips['keep'] = keep
    person_trips['drop_reason'] = drop_reason  # NEW: attach diagnosis to this chunk
    filtered_chunks.append(person_trips)

# Reassemble the FULL dataset (kept + dropped) before splitting
df_all = pd.concat(filtered_chunks, ignore_index=True)

# Split into kept and dropped for downstream use
df_kept = df_all[df_all['keep']].copy()
df_kept = df_kept.drop(columns=['keep', 'drop_reason']).copy()

df_dropped = df_all[~df_all['keep']].copy()  # NEW: dataframe of excluded trips

# Use the kept dataframe as the main analysis dataframe going forward
df = df_kept.copy()

print(f"\nRows before anchor + chain filter: {pre_filter_count:,}")
print(f"Rows after anchor + chain filter:  {len(df):,}")
print(f"Rows dropped:                    {len(df_dropped):,}")

# --- NEW: Compare anchor-only vs. anchor+chain --------------------------
recovered_by_chain = len(df) - len(anchor_only)
print(f"\n--- Comparison: anchor-only vs. anchor + chain ---")
print(f"Anchor-only (motivo_o in anchors):     {len(anchor_only):,}")
print(f"Anchor + chain (including intermediates): {len(df):,}")
print(f"Trips recovered by chaining:            {recovered_by_chain:,}")
print(f"Trips lost to chaining filter:            {len(df_dropped):,}")

# --- NEW: Diagnostic block so you can inspect the excluded trips ----------
print("\n" + "="*60)
print("DIAGNOSTIC: Why were trips dropped?")
print("="*60)

print("\n--- Drop reason breakdown ---")
print(df_dropped['drop_reason'].value_counts())

print("\n--- Origin purpose (motivo_o) of dropped trips ---")
print(df_dropped['motivo_o'].value_counts().sort_index())

print("\n--- Destination purpose (motivo_d) of dropped trips ---")
print(df_dropped['motivo_d'].value_counts().sort_index())

print("\n--- Dropped trips: origin x destination matrix ---")
print(pd.crosstab(df_dropped['motivo_o'], df_dropped['motivo_d'], margins=True))

print("\n--- People affected ---")
print(f"Unique persons with dropped trips: {df_dropped['id_pess'].nunique():,}")
print(f"Total persons in sample:           {df_all['id_pess'].nunique():,}")
print(f"Share: {df_dropped['id_pess'].nunique()/df_all['id_pess'].nunique()*100:.2f}%")

# Check if dropped trips tend to be the first or last trip of the day
df_dropped['is_first_trip'] = df_dropped.groupby('id_pess')['n_viag'].transform('min') == df_dropped['n_viag']
df_dropped['is_last_trip']  = df_dropped.groupby('id_pess')['n_viag'].transform('max') == df_dropped['n_viag']
print("\n--- Dropped trips by first/last position ---")
print(pd.crosstab(df_dropped['is_first_trip'], df_dropped['is_last_trip'], margins=True)
        .rename(index={False:'not_first',True:'first'}, columns={False:'not_last',True:'last'}))

if 'modo1' in df_dropped.columns:
    print("\n--- Main mode (modo1) of dropped trips ---")
    print(df_dropped['modo1'].value_counts().sort_index())

print("\n--- Sample of 10 dropped trips ---")
print(df_dropped[['id_pess','n_viag','motivo_o','motivo_d','drop_reason']]
      .head(10).to_string(index=False))

# --- User and trip summary -------------------------------------------------
raw_users      = df['id_pess'].nunique()
expanded_users = df.groupby('id_pess')['fe_pess'].first().sum()
raw_trips      = len(df)
expanded_trips = df['fe_via'].sum()

print("\n=== User and trip summary ===")
print(f"Unique users (raw):             {raw_users:,}")
print(f"Unique users (expanded):        {expanded_users:,.1f}")
print(f"Total trips (raw):              {raw_trips:,}")
print(f"Total trips (expanded):         {expanded_trips:,.1f}")
print(f"Mean trips per user (raw):      {raw_trips / raw_users:.2f}")
print(f"Mean trips per user (expanded): {expanded_trips / expanded_users:.2f}")

# --- Purpose at origin (motivo_o) in filtered data -----------------------
print("\n=== Purpose at origin (motivo_o) in filtered data ===")
print(f"{'Code':<5} {'Label':<30} {'Raw count':>12} {'Expanded (fe_via)':>18}")
print("-" * 68)

counts   = df['motivo_o'].value_counts().sort_index()
expanded = df.groupby('motivo_o')['fe_via'].sum().sort_index()

for code in counts.index:
    label = purpose_labels.get(code, "Unknown")
    print(f"{code:<5} {label:<30} {counts[code]:>12,} {expanded[code]:>18,.1f}")

print("-" * 68)
print(f"{'Total':<5} {'':30} {counts.sum():>12,} {expanded.sum():>18,.1f}")


# %% [markdown]
## 4b. Socio-economic profile (Critério Brasil) — for equity analysis

# %%
# criteriobr codes the household economic class (Critério Brasil), used later to
# group the 15-Minute City index by socio-economic profile (Saraiva & Barros, 2022).
ECONOMIC_CLASS = {1: "A", 2: "B1", 3: "B2", 4: "C1", 5: "C2", 6: "D-E"}
df["classe_economica"] = df["criteriobr"].map(ECONOMIC_CLASS)

print("=== Economic class (criteriobr) distribution ===")
print(f"{'Class':<6}{'Raw trips':>12}{'Persons':>12}{'Expanded persons':>20}")
print("-" * 50)
for code, label in ECONOMIC_CLASS.items():
    sub = df[df["criteriobr"] == code]
    raw_trips = len(sub)
    persons = sub["id_pess"].nunique()
    exp_persons = sub.groupby("id_pess")["fe_pess"].first().sum()
    print(f"{label:<6}{raw_trips:>12,}{persons:>12,}{exp_persons:>20,.0f}")
print("-" * 50)
print(f"{'Total':<6}{len(df):>12,}{df['id_pess'].nunique():>12,}"
      f"{df.groupby('id_pess')['fe_pess'].first().sum():>20,.0f}")

n_missing_class = df["classe_economica"].isna().sum()
if n_missing_class:
    print(f"\nWARNING: {n_missing_class:,} trips have an unmapped criteriobr value "
          f"({sorted(df.loc[df['classe_economica'].isna(), 'criteriobr'].unique())}).")

# Exploration note: the index is computed per origin regardless of class; the
# class label travels with each origin so results can be grouped and
# expansion-weighted (fe_pess / fe_via) for the equity analysis.


# %% [markdown]
## 5. Save the analysis-ready subset

# %%
clean_path = CLEAN_DATA_DIR / 'od_2023_sp_clean.csv'
df.to_csv(clean_path, index=False)
print(f"\nClean filtered dataset saved to: {clean_path}")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# %% [markdown]
## 6. Read and prepare zone shapefile

# %%
# Read shapefile with OD zone polygons.
zones_shp = gpd.read_file(
    RAW_DATA_DIR / '002_Site Metro Mapas_190225' / 'Shape' / 'Zonas_2023.shp'
)

# Clean names to match the CSV style (lowercase, no spaces)
zones_shp.columns = (
    zones_shp.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('/', '_')
)

# Verify CRS
zones_shp.crs


# Create a WGS84 copy for downstream spatial joins with OSM / R5.
# OSM amenities, road network and GTFS all use EPSG:4326.
# R5/r5py also requires origin coordinates as lat/lon.
zones_shp_4326 = zones_shp.to_crs(epsg=4326)


# %%
# Check columns and geometry
print("\n--- SHP CRS ---")
print(zones_shp.crs)

print("\n--- SHP Dtypes ---")
print(zones_shp.dtypes)

# %%
# Area is safe to calculate because EPSG:22523 is metric (metres)
zones_shp['area_m2'] = zones_shp.geometry.area

print("\n--- Shapefile Numeric Summary ---")
print(zones_shp.drop(columns='geometry').describe())

print("\n--- Zone count and total area ---")
print(f"Number of zones: {len(zones_shp)}")
print(f"Total area (km²): {zones_shp['area_m2'].sum() / 1e6:.1f}")


# %%
# Visualise
fig, ax = plt.subplots(figsize=(10, 10))
zones_shp.plot(ax=ax, edgecolor='black', facecolor='lightblue', linewidth=0.5)
ax.set_title('São Paulo OD Zones 2023 (EPSG:22523)')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()


# %% [markdown]
## 7. Wrangling shp file

# %%
# Filter to SP municipality
# numeromuni == 36 is São Paulo municipality proper
zones_sp = zones_shp[zones_shp['numeromuni'] == 36].copy()
zones_sp_4326 = zones_shp_4326[zones_shp_4326['numeromuni'] == 36].copy()

print(f"Zones in São Paulo municipality: {len(zones_sp)} (from {len(zones_shp)} total)")


# %% [markdown]
## 8. Save the analysis-ready subset geopackage

# Metric version: for internal OD work (area, density, zone-level aggregations)
clean_zones_path = CLEAN_DATA_DIR / 'zonas_sp_municipio_2023.gpkg'
zones_sp.to_file(clean_zones_path, driver='GPKG')
print(f"Clean zones (EPSG:22523) saved to: {clean_zones_path}")

# WGS84 version: for OSM / R5 spatial joins later
clean_zones_4326_path = CLEAN_DATA_DIR / 'zonas_sp_municipio_2023_4326.gpkg'
zones_sp_4326.to_file(clean_zones_4326_path, driver='GPKG')
print(f"Clean zones (EPSG:4326) saved to: {clean_zones_4326_path}")

# %% [markdown]
## 9. Georeference trip origins from the cleaned CSV

# The CSV coordinates (co_o_x, co_o_y) are in the same metric CRS as the zones.
# We build a GeoDataFrame explicitly in EPSG:22523, then derive lat/lon for r5py.


# %%
# Create point objects from coords
gdf_origins = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['co_o_x'], df['co_o_y']),
    crs='EPSG:22523'
)

# Reproject the points to WGS EPSG:4326
gdf_origins_4326 = gdf_origins.to_crs(epsg=4326)
# Pull the lat/lon back into the table as simple columns
df['lat_o'] = gdf_origins_4326.geometry.y
df['lon_o'] = gdf_origins_4326.geometry.x

# %% [markdown]
## 10. Save a parquet file with all the geometry transformations

# %%
trips_path = CLEAN_DATA_DIR / 'trips_sp_enriched.parquet'

# Force all columns into plain numpy-compatible dtypes.
# PyArrow sometimes fails on pandas ExtensionDtypes (e.g. string[pyarrow], nullable Int64).
df_save = pd.DataFrame()
for col in df.columns:
    if pd.api.types.is_integer_dtype(df[col]):
        df_save[col] = df[col].astype('int64')
    elif pd.api.types.is_float_dtype(df[col]):
        df_save[col] = df[col].astype('float64')
    else:
        # All remaining columns (strings, categoricals, etc.) → plain strings
        df_save[col] = df[col].astype('str')

df_save.to_parquet(trips_path, index=False)
print(f"\nEnriched trips saved to: {trips_path}")
print(f"Rows: {len(df_save):,} | CRS of metric coords: EPSG:22523 | CRS of lat/lon: EPSG:4326")


# %%[markdown]
# Now the DataFrame df carries both versions:
# - co_o_x and co_o_y in metres (EPSG:22523). These stay aligned with the zones and are good for distance calculations.
# - lat_o and lon_o in degrees (EPSG:4326). These are ready for R5 and OSM.

# The routing script later does not need to think about projections at all. 
# It just opens the file and feeds lat_o / lon_o to R5.
