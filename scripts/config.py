"""
Forest Fire Susceptibility — Shared Configuration
Single source of truth for all constants, paths, and API keys.
Every other script imports from here — never define these values elsewhere.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS        = PROJECT_ROOT / "outputs"
LOGS_DIR       = PROJECT_ROOT / "logs"

# Raw-data sub-directories (one per source). 
DEM30_DIR        = DATA_RAW / "dem30"        
LST_DIR          = DATA_RAW / "lst"          
S2_DIR           = DATA_RAW / "sentinel2"    
S2_16_DIR        = DATA_RAW / "sentinel2-16" # 16 days granules
BURNED_AREA_DIR  = DATA_RAW / "burned_area"  # MODIS MCD64A1 burned-area footprints

# Bagmati Province, Nepal

STUDY_AREA = {
    "name"      : "Bagmati Province, Nepal",
    "lat_min"   : 27.00,
    "lat_max"   : 28.40,
    "lon_min"   : 84.00,
    "lon_max"   : 86.60,
    "center_lat": 27.70,
    "center_lon": 85.30,
    "crs"       : "EPSG:4326",
}

# Province AOI mask — restricts analysis to Bagmati Province (within the bbox).
GADM_PATH           = DATA_RAW / "gadm" / "nepal_admin_level3.geojson"
PROVINCE_NAME       = "Bagmati"        
GADM_DISTRICT_FIELD = "NAME_3"         
BAGMATI_DISTRICTS   = frozenset({
    "Bhaktapur", "Chitawan", "Dhading", "Dolakha", "Kathmandu", "Kavrepalanchok",
    "Lalitpur", "Makwanpur", "Nuwakot", "Ramechhap", "Rasuwa", "Sindhuli",
    "Sindhupalchok",
})

# FIRMS API coordinates
BBOX_STR    = (                                     
    f"{STUDY_AREA['lon_min']},"
    f"{STUDY_AREA['lat_min']},"
    f"{STUDY_AREA['lon_max']},"
    f"{STUDY_AREA['lat_max']}"
)
# GEE coordinates
BBOX_COORDS = [                                     
    STUDY_AREA["lon_min"],
    STUDY_AREA["lat_min"],
    STUDY_AREA["lon_max"],
    STUDY_AREA["lat_max"],
]
# Overpass API coordinates
BBOX_OVERPASS = (                                   
    f"{STUDY_AREA['lat_min']},{STUDY_AREA['lon_min']},"
    f"{STUDY_AREA['lat_max']},{STUDY_AREA['lon_max']}"
)
# Date range
START_YEAR = 2015
END_YEAR   = 2026

# Nepal pre-monsoon fire season (February–May)
# Fire season dates
FIRE_SEASON_START = "02-01"
FIRE_SEASON_END   = "05-31"
FIRE_MONTHS       = (2, 3, 4, 5)   

# ─────────────────────────────────────────────────────────────────────────────
# 100 m Analysis Grid
GRID_RES_M    = 100                      
GRID_RES_DEG  = 3.0 / 3600.0             
GRID_TAG      = "100m"                   
GRID_WIDTH    = int(round((STUDY_AREA["lon_max"] - STUDY_AREA["lon_min"]) / GRID_RES_DEG))
GRID_HEIGHT   = int(round((STUDY_AREA["lat_max"] - STUDY_AREA["lat_min"]) / GRID_RES_DEG))
FEATURE_STACK     = DATA_PROCESSED / f"feature_stack_{GRID_TAG}"   
DATASET_PARQUET   = DATA_PROCESSED / f"forest_fire_dataset_{GRID_TAG}.parquet"
TRAINING_PARQUET  = DATA_PROCESSED / f"forest_fire_training_{GRID_TAG}.parquet"
# Backwards-compatible alias (older code referenced FEATURE_STACK_30M).
FEATURE_STACK_30M = FEATURE_STACK


def canonical_grid():
    """
    Return (transform, width, height, crs) for the canonical 30 m grid.
    """
    from rasterio.transform import from_origin
    transform = from_origin(
        STUDY_AREA["lon_min"], STUDY_AREA["lat_max"],   # north-west origin
        GRID_RES_DEG, GRID_RES_DEG,
    )
    return transform, GRID_WIDTH, GRID_HEIGHT, STUDY_AREA["crs"]

# API KEYS
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")
GEE_PROJECT   = os.getenv("GEE_PROJECT_ID", "")

# LULC Class Definitions
LULC_CLASSES = {
    10 : "Tree cover",
    20 : "Shrubland",
    30 : "Grassland",
    40 : "Cropland",
    50 : "Built-up",
    60 : "Bare / Sparse vegetation",
    70 : "Snow and ice",
    80 : "Permanent water bodies",
    90 : "Herbaceous wetland",
    95 : "Mangroves",
    100: "Moss and lichen",
}
FLAMMABLE_CLASSES = frozenset({10, 20, 30})   

def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured (e.g. when imported multiple times)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOGS_DIR / log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
