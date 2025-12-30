import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as crs_transform
from rasterio.transform import xy as transform_xy
from pystac_client import Client
import planetary_computer as pc
from shapely.geometry import box
import torch  # for Phase II context; not used here

warnings.filterwarnings("ignore")

# --- GLOBAL CONFIGURATION (7-Channel Setup) ---
INPUT_CHANNELS = 7      # B02, B03, B04, B08, DEM, SLOPE, ELEV_CLASS
NUM_CLASSES = 5         # T1..T4 + T5 catch-all (index 4)
BANDS = ["B02", "B03", "B04", "B08"]
MAX_CLOUD_COVER = 25
TILE_SIZE = 256

OUTPUT_DIR = Path("DRDO_Terrain_Dataset_7Channel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CRITICAL: Path to the merged ASTER GDEM file
DEM_PATH = Path("local_gdem_30m.tif")

AOI_CONFIG = {
    "T1_Ladakh": {
        "box": [77.0, 33.5, 78.5, 34.5],
        "date_range": "2020-06-01/2024-10-30",
        "class_label": 0,
    },
    "T2_Thar": {
        "box": [71.5, 27.0, 72.5, 28.0],
        "date_range": "2020-01-01/2024-12-31",
        "class_label": 1,
    },
    "T3_NE_Forest": {
        "box": [94.5, 27.0, 95.5, 28.0],
        "date_range": "2020-11-01/2025-01-30",
        "class_label": 2,
    },
    "T4_Urban": {
        "box": [77.0, 28.5, 77.5, 29.0],
        "date_range": "2020-01-01/2024-12-31",
        "class_label": 3,
    },
}

global_labels_list = []


# ---------------------------------------------------------------------
# Helper: ensure DEM exists
# ---------------------------------------------------------------------
if not DEM_PATH.exists():
    print(f"❌ FATAL ERROR: DEM file not found at '{DEM_PATH}'. Please run the mosaic script first.")
    raise SystemExit


# ---------------------------------------------------------------------
# Core AOI processing function
# ---------------------------------------------------------------------
def process_aoi(aoi_name, config):
    print(f"\n--- Starting Processing for {aoi_name} (Class {config['class_label']}) ---")
    aoi_geometry = box(*config["box"])
    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    # 1. STAC search
    try:
        client = Client.open(STAC_URL)
    except Exception as e:
        print(f"❌ Failed to connect to STAC API: {e}")
        return

    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi_geometry,
        datetime=config["date_range"],
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
    )

    items = list(search.get_all_items())
    if not items:
        print(f"⚠️ WARNING: No S2 items found for {aoi_name} in the date range. Skipping.")
        return

    # Sign best (lowest cloud) item
    selected_item = pc.sign(items[0])
    cc = selected_item.properties.get("eo:cloud_cover", 0.0)
    print(f"S2 item selected (CC: {cc:.2f}%). Proceeding with direct load...")

    # Ensure required bands exist
    missing = [b for b in BANDS if b not in selected_item.assets]
    if missing:
        print(f"❌ {aoi_name}: Missing bands in item assets: {missing}. Skipping.")
        return

    band_urls = {band: selected_item.assets[band].href for band in BANDS}
    s2_bands_list = []

    # 2. Load S2 bands (full scene) via rasterio
    try:
        with rasterio.Env():
            with rasterio.open(band_urls[BANDS[0]]) as src0:
                source_profile = src0.profile
                s2_crs = source_profile["crs"]
                s2_transform = source_profile["transform"]
                s2_nodata = src0.nodata
                s2_bands_list.append(src0.read(1))

            for band in BANDS[1:]:
                with rasterio.open(band_urls[band]) as src:
                    s2_bands_list.append(src.read(1))

        s2_array = np.stack(s2_bands_list, axis=0).astype(np.float32)

        # Optional: reflectance normalization (S2 L2A is typically 0–10000)
        if s2_nodata is not None:
            s2_array[s2_array == s2_nodata] = np.nan
        s2_array /= 10000.0  # now roughly 0–1

    except Exception as e:
        print(f"❌ Fatal Error during S2 Rasterio load/stack for {aoi_name}: {e}. Skipping.")
        return

    # 3. Align DEM to S2 grid
    H, W = s2_array.shape[1], s2_array.shape[2]
    with rasterio.open(DEM_PATH) as dem_src:
        dem_array = np.empty((H, W), dtype=np.float32)

        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_array,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
            resampling=Resampling.bilinear,
        )

        dem_nodata = dem_src.nodata

    # Mask DEM nodata
    if dem_nodata is not None:
        dem_array[dem_array == dem_nodata] = np.nan

    # 4. Compute slope from DEM  ### NEW
    # pixel sizes in meters from affine transform
    pixel_size_x = s2_transform.a
    pixel_size_y = -s2_transform.e if s2_transform.e < 0 else s2_transform.e

    # Gradient w.r.t. y (rows), x (cols)
    grad_y, grad_x = np.gradient(dem_array, pixel_size_y, pixel_size_x)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # 5. Elevation class from DEM  ### NEW
    # You can tune these bins as you like
    elev_bins = [0, 200, 500, 1000, 2000, 4000]  # meters
    elev_class = np.zeros_like(dem_array, dtype=np.int8)
    valid_elev = ~np.isnan(dem_array)
    elev_class[valid_elev] = np.digitize(dem_array[valid_elev], elev_bins).astype(np.int8)
    elev_class = elev_class.astype(np.float32)  # CNN usually expects floats

    # 6. Final 7-channel stack: [B02,B03,B04,B08,DEM,SLOPE,ELEV_CLASS]
    X_stack = np.stack(
        [
            s2_array[0],
            s2_array[1],
            s2_array[2],
            s2_array[3],
            dem_array,
            slope_deg,
            elev_class,
        ],
        axis=0,
    )

    approx_mb = X_stack.nbytes / (1024**2)
    print(f"{aoi_name}: stacked array shape {X_stack.shape}, ~{approx_mb:.1f} MB in RAM.")

    # 7. Tiling + labels, with lat/lon per tile  ### NEW
    current_aoi_labels = []
    num_skipped_empty = 0

    for r in range(0, H - TILE_SIZE + 1, TILE_SIZE):
        for c in range(0, W - TILE_SIZE + 1, TILE_SIZE):
            tile_data = X_stack[:, r : r + TILE_SIZE, c : c + TILE_SIZE]

            # Optional: skip tiles that are mostly NaN / nodata
            if np.isnan(tile_data).mean() > 0.5:
                num_skipped_empty += 1
                continue

            # Compute tile center lat/lon (WGS84)  ### NEW
            center_row = r + TILE_SIZE // 2
            center_col = c + TILE_SIZE // 2

            x_center, y_center = transform_xy(s2_transform, center_row, center_col)
            lon_list, lat_list = crs_transform(s2_crs, "EPSG:4326", [x_center], [y_center])
            center_lon, center_lat = float(lon_list[0]), float(lat_list[0])

            tile_filename = OUTPUT_DIR / f"{aoi_name}_r{r}_c{c}.npy"
            np.save(tile_filename, tile_data)

            current_aoi_labels.append(
                {
                    "filepath": str(tile_filename),
                    "label": config["class_label"],
                    "aoi_name": aoi_name,
                    "row": r,
                    "col": c,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                }
            )

    print(
        f"Successfully generated {len(current_aoi_labels)} tiles for {aoi_name}. "
        f"(Skipped {num_skipped_empty} mostly-empty tiles.)"
    )
    return current_aoi_labels


# ---------------------------------------------------------------------
# EXECUTION LOOP
# ---------------------------------------------------------------------
print("--- Starting Phase I: Full Multi-AOI Data Pipeline (7 Channels: S2+DEM+Slope+ElevClass) ---")

for aoi_name, config in AOI_CONFIG.items():
    labels = process_aoi(aoi_name, config)
    if labels:
        global_labels_list.extend(labels)

# ---------------------------------------------------------------------
# Final Deliverable: CSV with geo-aware labels
# ---------------------------------------------------------------------
if global_labels_list:
    final_df = pd.DataFrame(global_labels_list)
    csv_path = OUTPUT_DIR / "labels_master.csv"
    final_df.to_csv(csv_path, index=False)

    print("\n----------------------------------------------------------------------")
    print(f"🎉 PHASE I COMPLETE: {len(final_df)} labeled 7-channel tiles generated!")
    print(f"Labels CSV written to: {csv_path}")
    print("Columns include: filepath, label, aoi_name, row, col, center_lat, center_lon")
    print("----------------------------------------------------------------------")
    print("➡️ NEXT: Use Phase II (CNN Training) with 7 input channels.")
else:
    print("\nERROR: No tiles were generated. Check network connection or data availability.")
