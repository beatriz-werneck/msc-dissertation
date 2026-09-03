# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_08_srtm_elevation — SRTM DEM for slope adjustment
# -------------------------------------------------------------------------
"""
Download / load and explore the SRTM 1-arcsec (~30 m) Digital Elevation Model
for São Paulo. Used by the index to penalise walk/bike travel times on steep
grades (Saraiva and Barros, 2022; Farr et al., 2007).

Dependencies: elevation, rasterio, numpy, matplotlib.
"""
# %%
import numpy as np
import rasterio
from rasterio.plot import show as rio_show

import bcw_dis_00_config as cfg

cfg.SRTM_DIR.mkdir(parents=True, exist_ok=True)

# SP bounding box (W, S, E, N) in WGS84 for tile selection.
SRTM_BOUNDS = (-46.90, -24.00, -46.37, -23.35)

# %%
if not cfg.SRTM_TIF.exists():
    import elevation
    print("Downloading SRTM 30 m tiles...")
    elevation.clip(bounds=SRTM_BOUNDS, output=str(cfg.SRTM_TIF), product="SRTM1")
    elevation.clean()
    print(f"Saved DEM to: {cfg.SRTM_TIF}")
else:
    print(f"Using existing DEM: {cfg.SRTM_TIF}")

# %% [markdown]
## Data exploration
# %%
with rasterio.open(cfg.SRTM_TIF) as src:
    band = src.read(1)
    nodata = src.nodata
    profile = src.profile
    bounds = src.bounds
    res = src.res
    crs = src.crs
    transform = src.transform

# Mask nodata before computing statistics.
valid = band.astype("float64")
if nodata is not None:
    valid = np.where(band == nodata, np.nan, valid)

print("=" * 60)
print(f"DEM CRS: {crs}")
print(f"Size: {profile['width']} x {profile['height']} px | dtype: {profile['dtype']}")
print(f"Resolution (deg): {res[0]:.6f} x {res[1]:.6f}  (~{res[0]*111320:.0f} m)")
print(f"Bounds (W,S,E,N): {bounds.left:.4f}, {bounds.bottom:.4f}, "
      f"{bounds.right:.4f}, {bounds.top:.4f}")
print(f"NoData value: {nodata}")
print("=" * 60)

print("\n--- Elevation statistics (m) ---")
print(f"min:  {np.nanmin(valid):.1f}")
print(f"max:  {np.nanmax(valid):.1f}")
print(f"mean: {np.nanmean(valid):.1f}")
print(f"std:  {np.nanstd(valid):.1f}")
print(f"NoData pixels: {np.isnan(valid).sum():,} / {valid.size:,} "
      f"({np.isnan(valid).mean()*100:.2f}%)")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 9))
img = ax.imshow(
    valid,
    extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    cmap="terrain",
)
cfg.load_sp_boundary().boundary.plot(ax=ax, color="black", linewidth=0.8)
fig.colorbar(img, ax=ax, shrink=0.7, label="Elevation (m)")
ax.set_title("São Paulo SRTM elevation (~30 m)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
cfg.save_fig("08_srtm_elevation", fig)
