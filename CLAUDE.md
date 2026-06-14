# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Forest fire susceptibility analysis for **Bagmati Province, Nepal** (2015–2026). The pipeline downloads raw geospatial data from public APIs, engineers features at 100m resolution, and trains ML models on the resulting dataset.

## Commands

**Package manager**: `uv` — use `uv run` to execute scripts within the project environment.

```bash
# Install dependencies
uv sync

# Run all data download modules
uv run python scripts/download_data.py

# Run a single download module
uv run python scripts/download_data.py --module <name>
# Valid modules: firms, weather, dem, lulc, osm, worldpop, hansen, gadm, ndvi
```

> **Note**: The `ndvi` module only prints Google Earth Engine (GEE) submission instructions — actual extraction runs via `gee_extractor.py`.

## Architecture & Data Flow

```
Public APIs
    │
    ▼
scripts/download_data.py  →  data/raw/
    │                          ├── firms/       (NASA fire hotspots CSV)
    │                          ├── dem/         (SRTM 30m GeoTIFF)
    │                          ├── lulc/        (ESA WorldCover 10m)
    │                          ├── osm/         (roads, settlements, water JSON)
    │                          ├── worldpop/    (100m population GeoTIFF)
    │                          ├── hansen/      (Global Forest Change GeoTIFF)
    │                          ├── gadm/        (admin boundaries GeoJSON)
    │                          ├── ndvi/        (MODIS 250m via GEE)
    │                          ├── climate/     (gridded climate via GEE)
    │                          ├── sentinel2/   (Sentinel-2 imagery via GEE)
    │                          └── burned_area/ (MCD64A1 monthly GeoTIFFs)
    ▼
Feature engineering (script not yet committed)
    │
    ▼
data/processed/
    ├── forest_fire_dataset_100m.parquet   (177 MB — full dataset)
    ├── forest_fire_training_100m.parquet  (91 MB — training subset)
    ├── ndvi_sequences_100m.npz            (55 MB — NDVI time series)
    └── feature_stack_100m/               (43 individual feature GeoTIFFs)
```

## Key Architecture Notes

- **`scripts/config.py`** — single source of truth for all paths, constants, and API keys. All scripts import from here. Loads `.env` via `python-dotenv`. Key exports: `DATA_RAW`, `DATA_PROCESSED`, `STUDY_AREA`, `BAGMATI_DISTRICTS` (the 13 districts that compose Bagmati Province), `FLAMMABLE_CLASSES` (ESA codes 10/20/30 — Tree cover, Shrubland, Grassland only), `canonical_grid()`, `get_logger()`
- **`gee_extractor.py`** — Google Earth Engine extractor for NDVI, climate, LST, Sentinel-2, and burned area rasters. Not yet committed to the repo.
- **GADM boundary** — `nepal_admin_level3.geojson` uses pre-2015 Nepal zones (no "Bagmati" field). `config.py` reconstructs the province by filtering `NAME_3` on `BAGMATI_DISTRICTS`.
- **Canonical grid** — 3 arc-seconds (GRID_RES_DEG = 3/3600 ≈ 92 m), ~2,880 × 1,680 cells over the Bagmati bbox. All rasters are reprojected to this grid before stacking.

## Environment Variables

Credentials live in `.env` at the project root (loaded by `config.py`):

| Variable | Purpose |
|---|---|
| `FIRMS_MAP_KEY` | NASA FIRMS Archive API key (free — register at firms.modaps.eosdis.nasa.gov) |
| `GEE_PROJECT_ID` | Google Earth Engine project ID |
| `NASA_EARTHDATA_USERNAME` | NASA Earthdata login |
| `NASA_EARTHDATA_PASSWORD` | NASA Earthdata password |

**Warning**: `.env` is not currently in `.gitignore`. Add it before making any commits to avoid exposing credentials.
