"""
Forest Fire Susceptibility — Google Earth Engine Data Extractor
===============================================================
Study Area : Kathmandu Valley, Nepal
Period     : 2015–2025 (11 years)

Extracts the following datasets via the GEE Python API:

  1. NDVI         — MODIS MOD13Q1 v6.1 (250m, individual 16-day composites)
                    Every composite across the full year — ~23 per year × 11 years ≈ 253 tifs
                    Naming: ndvi_16day_YYYY-MM-DD.tif  (date = composite start date)
                    Output: data/raw/ndvi/ndvi_16day_YYYY-MM-DD.tif

                    During feature engineering, for each fire/non-fire sample point,
                    use the composite whose start date is closest to (and before) the
                    sample date to get the pre-event vegetation state.

  2. FIRMS        — NASA MODIS fire hotspots via GEE ImageCollection 'FIRMS'
                    Alternative to the FIRMS MAP_KEY API (requires GCP project only)
                    Output: data/raw/firms/firms_gee_{year}.csv (via Drive export)

  3. Burned Area  — MODIS MCD64A1 v6.1 (500m, monthly burn date composites)
                    Complements FIRMS point detections with spatial burn footprints.
                    Band: BurnDate (DOY 1–366; 0 = unburned; 900+ = ocean/water)
                    Output: data/raw/burned_area/mcd64a1_{year}_{month:02d}.tif

Usage
-----
  # First-time authentication (opens browser OAuth flow):
  uv run python scripts/gee_extractor.py --authenticate --project YOUR_GCP_PROJECT

  # Or set GEE_PROJECT_ID in .env to skip the --project flag:
  uv run python scripts/gee_extractor.py --authenticate

  uv run python scripts/gee_extractor.py --module ndvi
  uv run python scripts/gee_extractor.py --module firms
  uv run python scripts/gee_extractor.py --module all

Prerequisites
-------------
  1. Sign up for GEE (free for research): https://earthengine.google.com/
  2. Create a Google Cloud Project: https://console.cloud.google.com/
  3. Enable the Earth Engine API in your GCP project
  4. Add to .env:  GEE_PROJECT_ID=your-gcp-project-id
  5. Run with --authenticate once to cache credentials

GEE Dataset Catalog IDs
-----------------------
  NDVI  : MODIS/061/MOD13Q1  (250m, 16-day, from 2000-02-18)
  FIRMS : FIRMS               (MODIS MCD14DL hotspots, ~1 km)
"""

import argparse
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    BAGMATI_DISTRICTS,
    BBOX_COORDS,
    DATA_RAW,
    DEM30_DIR,
    END_YEAR,
    FIRE_SEASON_END,
    FIRE_SEASON_START,
    GADM_DISTRICT_FIELD,
    GADM_PATH,
    GEE_PROJECT,
    GRID_RES_M,
    LST_DIR,
    PROVINCE_NAME,
    S2_16_DIR,
    S2_DIR,
    START_YEAR,
    get_logger,
)

log = get_logger(__name__, "download.log")

# Drive folder for oversized-export fallbacks (province-scoped).
DRIVE_FOLDER = "ForestFire_Bagmati"


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT REGION  (province polygon if available, else the bbox rectangle)
# ─────────────────────────────────────────────────────────────────────────────

def _aoi_region():
    """
    Return the ee.Geometry to export over.

    Prefers the configured province polygon (GADM NAME_1) so exports are clipped
    to Bagmati — smaller downloads and no neighbouring-province bleed.  Falls
    back to the bbox rectangle if GADM/geopandas is unavailable.  The result is
    cached so we read the polygon only once per run.
    """
    import ee
    global _AOI_REGION_CACHE
    try:
        return _AOI_REGION_CACHE
    except NameError:
        pass

    region = None
    try:
        if GADM_PATH.exists():
            import geopandas as gpd
            gdf = gpd.read_file(GADM_PATH)
            if GADM_DISTRICT_FIELD in gdf.columns:
                wanted = {d.lower() for d in BAGMATI_DISTRICTS}
                prov = gdf[gdf[GADM_DISTRICT_FIELD].astype(str).str.strip()
                          .str.lower().isin(wanted)]
                if not prov.empty:
                    prov = prov.to_crs("EPSG:4326")
                    geom = (prov.geometry.union_all()
                            if hasattr(prov.geometry, "union_all")
                            else prov.geometry.unary_union)
                    region = ee.Geometry(geom.__geo_interface__)
                    log.info(f"  export region: '{PROVINCE_NAME}' province "
                             f"({prov[GADM_DISTRICT_FIELD].nunique()} districts)")
    except Exception as exc:
        log.warning(f"  province geometry unavailable ({exc}) — using bbox rectangle.")

    if region is None:
        region = ee.Geometry.Rectangle(list(BBOX_COORDS))
        log.info("  export region: bbox rectangle")

    _AOI_REGION_CACHE = region
    return region


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def _authenticate(project: str, force: bool = False) -> None:
    """
    Authenticate with Google Earth Engine and initialise the API.

    On the first call (or when force=True) this opens a browser window for
    OAuth2 consent; credentials are cached in ~/.config/earthengine/.
    Subsequent calls only need ee.Initialize().
    """
    try:
        import ee
    except ImportError:
        log.error(
            "earthengine-api is not installed.\n"
            "  Run: uv add earthengine-api   OR   uv sync"
        )
        sys.exit(1)

    if force:
        log.info("Opening browser for GEE OAuth2 authentication ...")
        ee.Authenticate()

    try:
        ee.Initialize(project=project)
        log.info(f"GEE initialised with project: {project}")
    except Exception as exc:
        log.error(
            f"GEE initialisation failed: {exc}\n"
            "  Try running with --authenticate to refresh credentials."
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _gee_download(
    image,
    region,
    dest: Path,
    scale_m: int,
    desc: str = "",
    drive_folder: str = "ForestFire_Bagmati",
    drive_fallback: bool = True,
) -> bool:
    """
    Download a single GEE image as a GeoTIFF via getDownloadURL().

    Parameterised by `scale_m` so every extractor (NDVI 250 m, LST 1 km,
    DEM/Sentinel-2 30 m, …) shares one implementation instead of copy-pasting
    the request/zip-extract/retry logic.  On failure (e.g. the 32 MB direct-
    download cap) it optionally starts a Drive export task and returns False.

    Returns True on a successful local save.
    """
    import ee

    if dest.exists():
        log.info(f"    [cache] {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        url = image.getDownloadURL({
            "scale"      : scale_m,
            "region"     : region,
            "format"     : "GEO_TIFF",
            "crs"        : "EPSG:4326",
            "filePerBand": False,
        })
    except Exception as exc:
        log.error(f"    getDownloadURL failed for {desc or dest.name}: {exc}")
        if drive_fallback:
            _gee_drive_export(image, region, scale_m, desc or dest.stem, drive_folder)
        return False

    log.info(f"    Downloading {desc or dest.name} ...")
    MAX_SECONDS = 240          # hard per-attempt wall-clock cap — kills trickle-stalls
    RETRIES = 4
    for attempt in range(RETRIES):
        try:
            # (connect, read) timeouts: if no chunk arrives for 60 s the read
            # raises and we retry, instead of blocking forever on a half-open socket.
            r = requests.get(url, stream=True, timeout=(15, 60))
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            t0 = time.time()

            if "zip" in content_type or url.endswith(".zip"):
                buf = bytearray()
                for chunk in r.iter_content(chunk_size=65536):
                    buf.extend(chunk)
                    if time.time() - t0 > MAX_SECONDS:
                        raise TimeoutError(f"stalled > {MAX_SECONDS}s")
                with zipfile.ZipFile(BytesIO(bytes(buf))) as zf:
                    tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                    if not tif_names:
                        log.error("    No .tif inside downloaded zip")
                        return False
                    with zf.open(tif_names[0]) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
            else:
                total = int(r.headers.get("content-length", 0))
                with open(dest, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True,
                    desc=desc or dest.name, leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        bar.update(len(chunk))
                        if time.time() - t0 > MAX_SECONDS:
                            raise TimeoutError(f"stalled > {MAX_SECONDS}s")

            if dest.stat().st_size == 0:
                dest.unlink()
                raise ValueError("Empty response")
            log.info(f"    → saved {dest.name}  ({dest.stat().st_size / 1024:.1f} KB)")
            return True

        except Exception as exc:
            log.warning(f"    Attempt {attempt + 1}/{RETRIES} failed: {exc}")
            # Drop any partial file so the retry (or a later run's cache check) is clean.
            if dest.exists():
                dest.unlink()
            if attempt < RETRIES - 1:
                time.sleep(5)

    log.error(f"    Direct download failed for {desc or dest.name}.")
    if drive_fallback:
        _gee_drive_export(image, region, scale_m, desc or dest.stem, drive_folder)
    return False


def _gee_drive_export(image, region, scale_m: int, desc: str, drive_folder: str):
    """Start a Drive export task when direct download is unavailable (e.g. >32 MB)."""
    import ee
    safe = desc.replace(" ", "_")[:100]
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=safe,
        folder=drive_folder,
        fileNamePrefix=safe,
        scale=scale_m,
        region=region,
        crs="EPSG:4326",
        maxPixels=int(1e9),
    )
    task.start()
    log.warning(
        f"\n    Drive export started for '{desc}'.\n"
        f"    Task ID : {task.id}\n"
        f"    Monitor : https://code.earthengine.google.com/tasks\n"
        f"    Then download from Google Drive → {drive_folder}/ to the matching data/raw/ folder.\n"
    )


def _gee_download_tiled(
    image, dest: Path, scale_m: int, desc: str, bbox_coords, n: int = 2,
    require_all: bool = False,
) -> bool:
    """
    Download a large image as n×n direct-download tiles, then mosaic to one GeoTIFF.

    Splitting the bbox keeps every getDownloadURL request under GEE's ~50 MB cap,
    so the result lands straight in data/raw/ with NO Drive fallback / manual step.
    `bbox_coords` is [W, S, E, N]; tiles overlap slightly to avoid seams (the final
    raster is resampled onto the canonical grid by feature_engineering anyway).

    require_all=True: if ANY tile fails, do NOT save a partial mosaic — return False
    and leave the good tiles in the temp dir (reused as cache on the next attempt),
    so the caller's retry fetches only the missing tile and the file is only written
    once all n×n tiles are present. Off by default (seasonal callers keep the old
    best-effort behaviour); the 16-day extractor turns it on to avoid silent
    missing-quadrant composites.
    """
    import ee
    import rasterio
    from rasterio.merge import merge as rio_merge

    if dest.exists():
        log.info(f"    [cache] {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)

    w, s, e, nlat = bbox_coords
    dx, dy = (e - w) / n, (nlat - s) / n
    ov = 0.005                                   # ~0.5 km tile overlap
    tmpdir = dest.parent / f".tiles_{desc}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    tile_paths = []
    for i in range(n):
        for j in range(n):
            tw, ts = w + j * dx, s + i * dy
            te, tn = min(w + (j + 1) * dx + ov, e), min(s + (i + 1) * dy + ov, nlat)
            rect = ee.Geometry.Rectangle([tw, ts, te, tn])
            tpath = tmpdir / f"{desc}_r{i}c{j}.tif"
            if _gee_download(image, rect, tpath, scale_m,
                             desc=f"{desc} tile r{i}c{j}", drive_fallback=False):
                tile_paths.append(tpath)
            else:
                log.warning(f"    tile r{i}c{j} failed for {desc}")
            time.sleep(1)

    if not tile_paths:
        log.error(f"    all tiles failed for {desc}")
        return False
    if require_all and len(tile_paths) < n * n:
        log.warning(f"    {desc}: only {len(tile_paths)}/{n * n} tiles succeeded — "
                    f"NOT saving (require_all); good tiles kept in {tmpdir.name} for retry")
        return False

    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = rio_merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    for sobj in srcs:
        sobj.close()
    with rasterio.open(dest, "w", **meta) as dst:
        dst.write(mosaic)
    for p in tile_paths:
        p.unlink(missing_ok=True)
    try:
        tmpdir.rmdir()
    except OSError:
        pass
    log.info(f"    → merged {len(tile_paths)} tiles → {dest.name} "
             f"({dest.stat().st_size / 1024:.1f} KB)")
    return True


def _modis_16day_windows(year: int):
    """
    Generate the MODIS MOD13Q1 16-day composite windows for a year.

    MODIS uses fixed DOY-based periods starting Jan 1 every year, advancing by
    16 days each time (~23 windows per year). Returns a list of
    (start_date_str, end_date_str) tuples — fully deterministic, no GEE call.
    Module-level so both the NDVI and the 16-day Sentinel-2 extractors share one
    definition (and therefore land on identical, 1:1-alignable window dates).
    """
    from datetime import date as date_cls, timedelta
    windows = []
    start = date_cls(year, 1, 1)
    while start.year == year:
        end = min(start + timedelta(days=15), date_cls(year, 12, 31))
        windows.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        start += timedelta(days=16)
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — NDVI via MODIS MOD13Q1
# ─────────────────────────────────────────────────────────────────────────────

class NDVIExtractor:
    """
    Downloads every individual MODIS MOD13Q1 16-day NDVI composite for
    Kathmandu Valley across the full study period (2015–2024).

    GEE Collection : MODIS/061/MOD13Q1
    Band           : NDVI  (raw int16 × 0.0001 → real NDVI in [-1, 1])
    Resolution     : 250 m
    Temporal res   : 16-day (finest cloud-free optical NDVI available)

    Output
    ------
    One GeoTIFF per composite window:
      ndvi_16day_YYYY-MM-DD.tif   ← date is the composite START date
      ~23 composites/year × 10 years ≈ 230 tifs (~33 MB total)

    Usage in feature engineering
    ----------------------------
    For each sample point (fire or background) with a known date D,
    pick the composite whose start date is the closest date ≤ D.
    This gives the pre-event vegetation state at that location.

    Download strategy
    -----------------
    Kathmandu Valley at 250 m is ~132 × 176 pixels (~145 KB per tif) —
    well below GEE's 32 MB direct-download limit.
    Uses ee.Image.getDownloadURL() for a direct zip-GeoTIFF response
    (no Google Drive export needed).
    Falls back to a Drive export task if the direct download fails.
    """

    COLLECTION = "MODIS/061/MOD13Q1"
    NDVI_BAND  = "NDVI"
    SCALE_M    = 250      # metres per pixel
    SCALE_F    = 0.0001   # raw int16 → real NDVI

    def __init__(self):
        self.out_dir = DATA_RAW / "ndvi"

    def _download_image(self, image, bbox, dest: Path, desc: str = "") -> bool:
        """
        Download a single GEE image as a GeoTIFF via getDownloadURL().
        Returns True on success; falls back to a Drive export task on failure.
        """
        import ee

        if dest.exists():
            log.info(f"    [cache] {dest.name}")
            return True

        try:
            url = image.getDownloadURL({
                "scale"      : self.SCALE_M,
                "region"     : bbox,
                "format"     : "GEO_TIFF",
                "crs"        : "EPSG:4326",
                "filePerBand": False,
            })
        except Exception as exc:
            log.error(f"    getDownloadURL failed: {exc}")
            self._start_drive_export(image, bbox, desc)
            return False

        log.info(f"    Downloading {desc or dest.name} ...")
        for attempt in range(2):
            try:
                r            = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                content_type = r.headers.get("content-type", "")

                if "zip" in content_type or url.endswith(".zip"):
                    data = b"".join(r.iter_content(chunk_size=65536))
                    with zipfile.ZipFile(BytesIO(data)) as zf:
                        tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                        if not tif_names:
                            log.error("    No .tif inside downloaded zip")
                            return False
                        with zf.open(tif_names[0]) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                else:
                    total = int(r.headers.get("content-length", 0))
                    with open(dest, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=desc or dest.name, leave=False,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                            bar.update(len(chunk))

                size_kb = dest.stat().st_size / 1024
                if dest.stat().st_size == 0:
                    dest.unlink()
                    log.warning(f"    Empty file received for {desc} — will retry")
                    raise ValueError("Empty response")
                log.info(f"    → saved {dest.name}  ({size_kb:.1f} KB)")
                return True

            except Exception as exc:
                log.warning(f"    Attempt {attempt + 1} failed: {exc}")
                if dest.exists() and dest.stat().st_size == 0:
                    dest.unlink()
                if attempt == 0:
                    time.sleep(5)

        log.error("    Direct download failed — starting Drive export as fallback.")
        self._start_drive_export(image, bbox, desc)
        return False

    def _start_drive_export(self, image, bbox, desc: str):
        """Start a Drive export task when direct download is unavailable."""
        import ee
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=desc.replace(" ", "_")[:100],
            folder="ForestFire_Bagmati",
            fileNamePrefix=desc.replace(" ", "_")[:100],
            scale=self.SCALE_M,
            region=bbox,
            crs="EPSG:4326",
            maxPixels=int(1e9),
        )
        task.start()
        log.warning(
            f"\n    Drive export started for '{desc}'.\n"
            f"    Task ID : {task.id}\n"
            f"    Monitor : https://code.earthengine.google.com/tasks\n"
            f"    Download from Google Drive → ForestFire_Bagmati/ → data/raw/ndvi/\n"
        )

    def download_16day_composites(self):
        """
        Download every MODIS MOD13Q1 16-day composite for the study period.

        Key design: composite windows are pre-computed locally (no aggregate_array /
        size().getInfo() calls) — this eliminates hanging server-side GEE calls.
        For each window we call collection.filterDate().first() then
        getDownloadURL() — only one network round-trip per composite.
        """
        import ee

        bbox = _aoi_region()
        total_saved  = 0
        total_cached = 0
        total_failed = 0

        for year in range(START_YEAR, END_YEAR + 1):
            windows = _modis_16day_windows(year)
            log.info(f"  {year}: {len(windows)} composite windows to process")

            for start_str, end_str in windows:
                dest = self.out_dir / f"ndvi_16day_{start_str}.tif"

                if dest.exists():
                    log.info(f"    [cache] {dest.name}")
                    total_cached += 1
                    continue

                # Filter to this exact 16-day window and take the single image
                # end_str + 1 day so filterDate includes the end date
                end_exclusive = (
                    pd.Timestamp(end_str) + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d")

                image = (
                    ee.ImageCollection(self.COLLECTION)
                    .filterBounds(bbox)
                    .filterDate(start_str, end_exclusive)
                    .select(self.NDVI_BAND)
                    .first()                          # single image, no .getInfo()
                    .multiply(self.SCALE_F)
                    .rename("NDVI")
                )

                if self._download_image(image, bbox, dest, desc=f"NDVI_16day_{start_str}"):
                    total_saved += 1
                else:
                    total_failed += 1

                time.sleep(1)

        log.info(
            f"  16-day composites complete — "
            f"saved: {total_saved}, cached: {total_cached}, failed: {total_failed}"
        )

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 1 (GEE): MODIS NDVI — MOD13Q1 v6.1 (individual 16-day composites)")
        log.info(f"  Collection : {self.COLLECTION}")
        log.info(f"  Period     : {START_YEAR}–{END_YEAR}")
        log.info(f"  Resolution : {self.SCALE_M} m")
        log.info(f"  Expected   : ~{(END_YEAR - START_YEAR + 1) * 23} tifs  (~{(END_YEAR - START_YEAR + 1) * 23 * 145 // 1024} MB)")
        log.info(f"  Output dir : {self.out_dir}")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.download_16day_composites()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — FIRMS fire hotspots via GEE (alternative to FIRMS MAP_KEY API)
# ─────────────────────────────────────────────────────────────────────────────

class FIRMSGEEExtractor:
    """
    Extracts NASA FIRMS MODIS fire hotspot data from GEE.

    GEE Collection : FIRMS (MODIS MCD14DL, ~1 km, daily)
    Output         : data/raw/firms/firms_gee_{year}.csv  (via Drive export)

    This is an alternative to FIRMSDownloader in download_data.py.
    It requires only a GCP project — no FIRMS MAP_KEY needed.

    Export strategy
    ---------------
    The FIRMS ImageCollection stores one image per detection date.  To get
    point records (lat, lon, confidence, year) we build a max-confidence
    annual composite, sample fire pixels (confidence ≥ 50), and export to
    Google Drive as a CSV.  After downloading from Drive, move the files to
    data/raw/firms/.
    """

    COLLECTION   = "FIRMS"
    DRIVE_FOLDER = "ForestFire_Bagmati"

    def __init__(self):
        self.out_dir = DATA_RAW / "firms"

    def _export_year(self, year: int) -> str | None:
        import ee
        bbox  = _aoi_region()
        firms = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
        )

        count = firms.size().getInfo()
        if count == 0:
            log.info(f"  {year}: no FIRMS images in study area")
            return None

        log.info(f"  {year}: {count} FIRMS images — sampling fire pixels ...")

        # Max-confidence composite for the year
        confidence_mosaic = firms.select("confidence").max()
        fire_mask         = confidence_mosaic.gte(50)
        fire_pixels       = confidence_mosaic.updateMask(fire_mask)

        fire_features = fire_pixels.sample(
            region    = bbox,
            scale     = 1000,
            geometries= True,
            numPixels = 10000,
        )
        fire_features = fire_features.map(lambda f: f.set("year", year))

        desc = f"FIRMS_GEE_{year}"
        task = ee.batch.Export.table.toDrive(
            collection    = fire_features,
            description   = desc,
            folder        = self.DRIVE_FOLDER,
            fileNamePrefix= desc,
            fileFormat    = "CSV",
        )
        task.start()
        log.info(
            f"  {year}: Drive export started  task={task.id}\n"
            f"    Monitor: https://code.earthengine.google.com/tasks"
        )
        return task.id

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 2 (GEE): NASA FIRMS MODIS Fire Hotspots")
        log.info(f"  Collection   : {self.COLLECTION}")
        log.info(f"  Period       : {START_YEAR}–{END_YEAR}")
        log.info(f"  Drive folder : {self.DRIVE_FOLDER}")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        task_ids = {}
        for year in range(START_YEAR, END_YEAR + 1):
            tid = self._export_year(year)
            if tid:
                task_ids[year] = tid
            time.sleep(0.5)

        if task_ids:
            manifest = self.out_dir / "firms_gee_task_manifest.csv"
            pd.DataFrame(
                [{"year": y, "task_id": tid} for y, tid in task_ids.items()]
            ).to_csv(manifest, index=False)
            log.info(f"  Task manifest saved → {manifest.name}")
            log.info(
                "\n"
                "  Next steps:\n"
                "  1. Visit https://code.earthengine.google.com/tasks\n"
                "  2. Wait for all FIRMS_GEE_* tasks to complete\n"
                f" 3. Download CSVs from Google Drive → {self.DRIVE_FOLDER}/\n"
                "  4. Move files to: data/raw/firms/\n"
            )
        else:
            log.warning("  No FIRMS export tasks started (no fire data found).")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — MODIS Burned Area MCD64A1 (monthly, 500 m)
# ─────────────────────────────────────────────────────────────────────────────

class BurnedAreaExtractor:
    """
    Downloads MODIS MCD64A1 v6.1 monthly burned area composites for
    Kathmandu Valley across the full study period (2015–2025).

    GEE Collection : MODIS/061/MCD64A1
    Band           : BurnDate — day-of-year (1–366) of detected burn;
                     0 = unburned; 900+ = ocean/water/unclassified.
    Resolution     : 500 m
    Temporal res   : Monthly

    Why use this alongside FIRMS?
    FIRMS gives point detections (thermal anomaly centroids at ~375–1000 m).
    MCD64A1 gives *spatial burn footprints* — every 500-m pixel that was
    confirmed burned in that calendar month.  Using both improves target-
    variable quality: FIRMS catches small/brief fires; MCD64A1 captures
    the full spatial extent of larger burn events.

    Output
    ------
    One GeoTIFF per calendar month:
      mcd64a1_{year}_{month:02d}.tif   (BurnDate band, uint16, EPSG:4326)
      ~11 years × 12 months = 132 tifs

    Download strategy
    -----------------
    Kathmandu Valley at 500 m is ~67 × 89 pixels — well below GEE's 32 MB
    direct-download limit.  Uses ee.Image.getDownloadURL() for a direct
    GeoTIFF response (no Google Drive export needed).
    """

    COLLECTION = "MODIS/061/MCD64A1"
    BAND       = "BurnDate"
    SCALE_M    = 500    # metres per pixel

    def __init__(self):
        self.out_dir = DATA_RAW / "burned_area"

    def _download_image(self, image, bbox, dest: Path, desc: str = "") -> bool:
        """Download a single GEE image as a GeoTIFF. Returns True on success."""
        import ee

        if dest.exists():
            log.info(f"    [cache] {dest.name}")
            return True

        try:
            url = image.getDownloadURL({
                "scale"      : self.SCALE_M,
                "region"     : bbox,
                "format"     : "GEO_TIFF",
                "crs"        : "EPSG:4326",
                "filePerBand": False,
            })
        except Exception as exc:
            log.error(f"    getDownloadURL failed for {desc}: {exc}")
            return False

        log.info(f"    Downloading {desc or dest.name} ...")
        for attempt in range(2):
            try:
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                content_type = r.headers.get("content-type", "")

                if "zip" in content_type or url.endswith(".zip"):
                    data = b"".join(r.iter_content(chunk_size=65536))
                    with zipfile.ZipFile(BytesIO(data)) as zf:
                        tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                        if not tif_names:
                            log.error("    No .tif inside downloaded zip")
                            return False
                        with zf.open(tif_names[0]) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                else:
                    total = int(r.headers.get("content-length", 0))
                    with open(dest, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=desc or dest.name, leave=False,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                            bar.update(len(chunk))

                if dest.stat().st_size == 0:
                    dest.unlink()
                    raise ValueError("Empty response")
                size_kb = dest.stat().st_size / 1024
                log.info(f"    → saved {dest.name}  ({size_kb:.1f} KB)")
                return True

            except Exception as exc:
                log.warning(f"    Attempt {attempt + 1} failed: {exc}")
                if dest.exists() and dest.stat().st_size == 0:
                    dest.unlink()
                if attempt == 0:
                    time.sleep(5)

        log.error(f"    Failed to download {desc} after 2 attempts.")
        return False

    def run(self):
        import ee

        log.info("=" * 60)
        log.info("MODULE 3 (GEE): MODIS Burned Area — MCD64A1 v6.1")
        log.info(f"  Collection : {self.COLLECTION}")
        log.info(f"  Band       : {self.BAND} (DOY of burn; 0 = unburned)")
        log.info(f"  Period     : {START_YEAR}–{END_YEAR}")
        log.info(f"  Resolution : {self.SCALE_M} m")
        log.info(f"  Expected   : ~{(END_YEAR - START_YEAR + 1) * 12} monthly tifs")
        log.info(f"  Output dir : {self.out_dir}")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        bbox         = _aoi_region()
        total_saved  = 0
        total_cached = 0
        total_failed = 0

        for year in range(START_YEAR, END_YEAR + 1):
            log.info(f"  {year}: processing 12 monthly composites ...")
            for month in range(1, 13):
                dest = self.out_dir / f"mcd64a1_{year}_{month:02d}.tif"
                if dest.exists():
                    log.info(f"    [cache] {dest.name}")
                    total_cached += 1
                    continue

                # Calendar month boundaries — no server-side size() calls
                import calendar
                last_day   = calendar.monthrange(year, month)[1]
                start_str  = f"{year}-{month:02d}-01"
                end_str    = f"{year}-{month:02d}-{last_day}"
                # filterDate end is exclusive, so add one day
                end_excl   = (
                    pd.Timestamp(end_str) + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d")

                image = (
                    ee.ImageCollection(self.COLLECTION)
                    .filterBounds(bbox)
                    .filterDate(start_str, end_excl)
                    .select(self.BAND)
                    .first()
                    .rename(self.BAND)
                )

                desc = f"MCD64A1_{year}_{month:02d}"
                if self._download_image(image, bbox, dest, desc=desc):
                    total_saved += 1
                else:
                    total_failed += 1

                time.sleep(0.5)

        log.info(
            f"  Burned area complete — "
            f"saved: {total_saved}, cached: {total_cached}, failed: {total_failed}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Copernicus GLO-30 DEM (true 30 m terrain)
# ─────────────────────────────────────────────────────────────────────────────

class DEM30Extractor:
    """
    Downloads the Copernicus GLO-30 Digital Elevation Model (~30 m) for the
    study area — the canonical terrain source for the 30 m feature stack.

    GEE Collection : COPERNICUS/DEM/GLO30  (band 'DEM', metres, EGM2008)
    Resolution     : 1 arc-second (~30 m) — matches the project's canonical grid.

    Why not the on-disk SRTM?  The downloaded SRTM tile is 3 arc-second (~90 m);
    GLO-30 gives genuine 30 m relief so slope/aspect/TRI/TWI carry real detail.

    Output : data/raw/dem30/copdem30.tif  (single GeoTIFF, EPSG:4326)
    """

    COLLECTION = "COPERNICUS/DEM/GLO30"
    BAND       = "DEM"
    SCALE_M    = GRID_RES_M

    def __init__(self):
        self.out_dir = DEM30_DIR

    def run(self):
        import ee
        log.info("=" * 60)
        log.info("MODULE 4 (GEE): Copernicus GLO-30 DEM (true 30 m terrain)")
        log.info(f"  Collection : {self.COLLECTION}  band={self.BAND}")
        log.info(f"  Resolution : {self.SCALE_M} m   Output dir: {self.out_dir}")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        bbox  = _aoi_region()
        # GLO30 is an ImageCollection of 1°×1° tiles — mosaic + clip to bbox.
        image = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .select(self.BAND)
            .mosaic()
            .rename("elevation")
            .clip(bbox)
        )
        dest = self.out_dir / "copdem30.tif"
        ok = _gee_download(image, bbox, dest, self.SCALE_M, desc="CopDEM30")
        log.info(f"  DEM30 {'saved' if ok else 'FAILED (see Drive tasks)'}: {dest.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — Land Surface Temperature (MODIS MOD11A2)
# ─────────────────────────────────────────────────────────────────────────────

class LSTExtractor:
    """
    Downloads fire-season (Feb–May) mean daytime Land Surface Temperature
    per year from MODIS MOD11A2 (8-day, 1 km).

    GEE Collection : MODIS/061/MOD11A2  (band 'LST_Day_1km')
    Scale factor   : 0.02 → Kelvin;  −273.15 → °C
    Output         : data/raw/lst/lst_fire_season_{year}.tif  (°C, EPSG:4326)

    LST is a strong fire-susceptibility predictor (surface dryness / heat load)
    and is one of the methodology's novel-for-Nepal variables.
    """

    COLLECTION = "MODIS/061/MOD11A2"
    BAND       = "LST_Day_1km"
    SCALE_M    = 1000

    def __init__(self):
        self.out_dir = LST_DIR

    def run(self):
        import ee
        log.info("=" * 60)
        log.info("MODULE 5 (GEE): MODIS Land Surface Temperature — MOD11A2")
        log.info(f"  Period: {START_YEAR}–{END_YEAR}  fire season {FIRE_SEASON_START}–{FIRE_SEASON_END}")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        bbox = _aoi_region()
        saved = cached = failed = 0
        for year in range(START_YEAR, END_YEAR + 1):
            dest = self.out_dir / f"lst_fire_season_{year}.tif"
            if dest.exists():
                log.info(f"    [cache] {dest.name}")
                cached += 1
                continue

            start = f"{year}-{FIRE_SEASON_START}"
            end   = f"{year}-{FIRE_SEASON_END}"
            end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            coll = (
                ee.ImageCollection(self.COLLECTION)
                .filterBounds(bbox)
                .filterDate(start, end_excl)
                .select(self.BAND)
            )
            image = (
                coll.mean()
                .multiply(0.02).subtract(273.15)
                .rename("lst_day_c")
            )
            if _gee_download(image, bbox, dest, self.SCALE_M, desc=f"LST_{year}"):
                saved += 1
            else:
                failed += 1
            time.sleep(1)

        log.info(f"  LST complete — saved: {saved}, cached: {cached}, failed: {failed}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — Sentinel-2 spectral indices (NBR / NDWI / EVI)
# ─────────────────────────────────────────────────────────────────────────────

def _compact_years(years):
    """[2017,2018,2019,2023] → '2017–2019, 2023' for compact manifest logging."""
    ys = sorted(years)
    if not ys:
        return "—"
    runs, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
            continue
        runs.append((start, prev))
        start = prev = y
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs)


class Sentinel2Extractor:
    """
    Builds a cloud-masked fire-season Sentinel-2 composite per year and exports
    three spectral indices as a multi-band GeoTIFF.

    GEE Collection : COPERNICUS/S2_SR_HARMONIZED  (surface reflectance, 10–20 m)
    Indices
      NBR  = (B8 − B12) / (B8 + B12)     burn / vegetation stress
      NDWI = (B8 − B11) / (B8 + B11)     canopy moisture (Gao / NDMI form)
      EVI  = 2.5·(B8 − B4) / (B8 + 6·B4 − 7.5·B2 + 1)

    Exported at the 100 m grid scale (aggregated from the native 10–20 m bands).
    The province-wide 3-band image is ~90 MB, over GEE's ~50 MB direct-download
    cap, so it is fetched as a 2×2 grid of sub-cap tiles and mosaicked locally
    (_gee_download_tiled) — no Drive fallback / manual step required.

    Output : data/raw/sentinel2/s2_{season}_{year}.tif  (bands NBR, NDWI, EVI),
             one composite per season per year — mirrors the multi-temporal NDVI features.
    """

    COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
    SCALE_M    = GRID_RES_M       # 100 m; tiled so each request fits direct download
    N_TILES    = 2                # n×n tiles per export to stay under the 50 MB cap
    # Fire-relevant seasons (month-day start/end, inclusive). premonsoon & fire_season
    # mirror the NDVI windows; postmonsoon uses the Oct–Nov dry-season onset (S2 is too
    # cloud-blocked in the Jun–Aug monsoon that NDVI's postmonsoon label uses).
    SEASONS = {
        "premonsoon" : ("01-01", "02-28"),
        "fire_season": (FIRE_SEASON_START, FIRE_SEASON_END),
        "postmonsoon": ("10-01", "11-30"),
    }
    # COPERNICUS/S2_SR_HARMONIZED surface-reflectance archive begins 2017-03-28.
    # Any window that *ends* before this is expectedly empty (e.g. premonsoon 2017,
    # Jan 1–Feb 28) — flagged KNOWN-EMPTY in the manifest, not counted as a gap.
    S2_SR_START = pd.Timestamp("2017-03-28")

    def __init__(self):
        self.out_dir = S2_DIR

    @staticmethod
    def _mask_clouds(img):
        """Mask clouds/cirrus using the QA60 bitmask (bits 10 & 11)."""
        import ee
        qa = img.select("QA60")
        cloud_bit  = 1 << 10
        cirrus_bit = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit).eq(0))
        )
        return img.updateMask(mask).divide(10000)   # scale reflectance to 0–1

    def _indices(self, composite):
        nbr  = composite.normalizedDifference(["B8", "B12"]).rename("NBR")
        ndwi = composite.normalizedDifference(["B8", "B11"]).rename("NDWI")
        evi  = composite.expression(
            "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
            {
                "NIR" : composite.select("B8"),
                "RED" : composite.select("B4"),
                "BLUE": composite.select("B2"),
            },
        ).rename("EVI")
        # Cast to a uniform Float32: normalizedDifference yields Float32 but the
        # EVI expression yields Float64, and a mixed-dtype image makes
        # Export.toDrive fail ("bands must have compatible data types").
        return nbr.addBands(ndwi).addBands(evi).toFloat()

    def run(self):
        import ee
        log.info("=" * 60)
        log.info("MODULE 6 (GEE): Sentinel-2 spectral indices — NBR / NDWI / EVI")
        log.info(f"  Period: {START_YEAR}–{END_YEAR}  export scale {self.SCALE_M} m")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        today = pd.Timestamp.today().normalize()
        years = list(range(max(START_YEAR, 2017), END_YEAR + 1))

        # ── 1. Manifest: classify every (season, year) BEFORE downloading ──────
        # CACHED       file already on disk
        # FUTURE-SKIP  window end-date is still in the future (data not yet collected)
        # KNOWN-EMPTY  window ends before the S2_SR archive begins (2017-03-28)
        # TO-DOWNLOAD  a real, fetchable gap we will attempt this pass
        manifest = {s: {"CACHED": [], "TO-DOWNLOAD": [],
                        "FUTURE-SKIP": [], "KNOWN-EMPTY": []} for s in self.SEASONS}
        todo = []   # (season, year, start, end_excl, dest) for each TO-DOWNLOAD
        for season, (s_md, e_md) in self.SEASONS.items():
            for year in years:
                dest    = self.out_dir / f"s2_{season}_{year}.tif"
                win_end = pd.Timestamp(f"{year}-{e_md}")
                if dest.exists():
                    manifest[season]["CACHED"].append(year)
                elif win_end > today:
                    manifest[season]["FUTURE-SKIP"].append(year)
                elif win_end < self.S2_SR_START:
                    manifest[season]["KNOWN-EMPTY"].append(year)
                else:
                    end_excl = (win_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    todo.append((season, year, f"{year}-{s_md}", end_excl, dest))
                    manifest[season]["TO-DOWNLOAD"].append(year)

        log.info("  ┌─ Sentinel-2 manifest  (season · status → years) ───────────")
        for season in self.SEASONS:
            for status in ("CACHED", "TO-DOWNLOAD", "FUTURE-SKIP", "KNOWN-EMPTY"):
                yrs = manifest[season][status]
                if yrs:
                    log.info(f"  │  {season:<11} {status:<12} {_compact_years(yrs)}")
        log.info(f"  └─ {len(todo)} mosaic(s) to download this pass")

        # ── 2. Download loop — each mosaic isolated so one failure ≠ run death ──
        bbox = _aoi_region()
        saved, empty, failed = 0, [], []
        for i, (season, year, start, end_excl, dest) in enumerate(todo, 1):
            name = f"s2_{season}_{year}"
            log.info(f"  [{i}/{len(todo)}] {name}  ({start} → {end_excl}) ...")
            try:
                coll = (
                    ee.ImageCollection(self.COLLECTION)
                    .filterBounds(bbox)
                    .filterDate(start, end_excl)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
                    .map(self._mask_clouds)
                )
                n_scenes = coll.size().getInfo()
                if n_scenes == 0:
                    log.warning(f"      no S2 scenes for {name} — skipped (empty window)")
                    empty.append(name)
                    continue
                log.info(f"      {n_scenes} scenes → median composite, tiling {self.N_TILES}×{self.N_TILES}")
                composite = coll.median()
                image     = self._indices(composite).clip(bbox)
                if _gee_download_tiled(image, dest, self.SCALE_M,
                                       name, BBOX_COORDS, n=self.N_TILES):
                    saved += 1
                    log.info(f"      ✓ saved {dest.name}")
                else:
                    failed.append(name)
                    log.error(f"      ✗ tiled download failed for {name}")
            except Exception as exc:   # transient GEE/network death → log & keep going
                failed.append(name)
                log.error(f"      ✗ {name} crashed: {exc!r} — continuing to next mosaic")
            time.sleep(1)

        # ── 3. Gap report — explicit, names the real gaps that remain ──────────
        real = [(s, y) for s in self.SEASONS for y in
                manifest[s]["CACHED"] + manifest[s]["TO-DOWNLOAD"]]
        still_missing = [f"s2_{s}_{y}" for s, y in real
                         if not (self.out_dir / f"s2_{s}_{y}.tif").exists()]
        log.info("  ┌─ Sentinel-2 run summary ───────────────────────────────────")
        log.info(f"  │  saved this pass : {saved}")
        log.info(f"  │  empty windows   : {len(empty)}  {empty or ''}")
        log.info(f"  │  failed this pass: {len(failed)}  {failed or ''}")
        log.info(f"  │  present / real  : {len(real) - len(still_missing)}/{len(real)} mosaics")
        if still_missing:
            log.warning(f"  │  REAL GAPS REMAIN ({len(still_missing)}): {still_missing}")
            log.warning("  │  → re-run `gee_extractor.py --module sentinel2` to mop up")
        else:
            log.info("  │  no real gaps remain ✓  (future / known-empty windows excluded)")
        log.info("  └────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6b — Sentinel-2 spectral indices as 16-day composites (NDVI-aligned)
# ─────────────────────────────────────────────────────────────────────────────

class Sentinel2SixteenDayExtractor(Sentinel2Extractor):
    """
    Sentinel-2 NBR/NDWI/EVI at the SAME 16-day cadence as the MODIS NDVI series
    (`_modis_16day_windows`), one composite per window → data/raw/sentinel2-16/.

    Reuses every building block of the seasonal `Sentinel2Extractor` (cloud mask,
    index math, 2×2 tiled download); only the output dir and the per-window loop
    differ. Output: s2_16day_<window-start>.tif (bands NBR, NDWI, EVI) — the S2
    analogue of ndvi_16day_<start>.tif, 1:1-alignable with it by start date.

    NOTE: a 16-day window has far fewer cloud-free S2 scenes than a seasonal
    composite, so many windows (esp. the Jun–Sep monsoon) return zero scenes and
    are logged + skipped — expected sparsity, not a failure.
    """

    # Legit windows finish in well under ~90 s (4 tiles). Anything past this cap is
    # a hung EE RPC, not heavy compute — abandon it and let the driver retry.
    WINDOW_CAP_S = 240

    def __init__(self):
        self.out_dir = S2_16_DIR

    def _fetch_window(self, ee, start, end_excl, dest, name, bbox):
        """Fetch ONE 16-day window. Returns 'saved' | 'empty' | 'failed'.
        Runs inside a daemon watchdog thread (see run) so a hung getInfo /
        getDownloadURL can never freeze the whole extraction."""
        coll = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(bbox)
            .filterDate(start, end_excl)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(self._mask_clouds)
        )
        if coll.size().getInfo() == 0:
            return "empty"
        image = self._indices(coll.median()).clip(bbox)
        ok = _gee_download_tiled(image, dest, self.SCALE_M, name, BBOX_COORDS,
                                 n=self.N_TILES, require_all=True)
        return "saved" if ok else "failed"

    def run(self):
        import ee
        import socket
        import threading
        # EE getInfo()/getDownloadURL intermittently hang with no client read
        # timeout. No single deadline fits every case (a generous one makes genuine
        # hangs grind via EE's internal retries; a tight one cuts off slow windows),
        # so the real defence is a per-window WALL-CLOCK CAP: each window runs in a
        # daemon thread we join with a timeout; if it overruns WINDOW_CAP_S it is
        # abandoned (the daemon thread dies at process exit, so it can't block the
        # run) and the multi-pass driver retries it. socket/deadline are secondary.
        socket.setdefaulttimeout(150)              # client-side read timeout (s)
        try:
            ee.data.setDeadline(120_000)           # EE RPC deadline (ms)
        except Exception as exc:
            log.warning(f"  could not set EE request deadline ({exc!r})")
        log.info("=" * 60)
        log.info("MODULE 6b (GEE): Sentinel-2 16-day composites — NBR / NDWI / EVI")
        log.info(f"  Period: {max(START_YEAR, 2017)}–{END_YEAR}  "
                 f"export scale {self.SCALE_M} m  (windows aligned to MODIS NDVI)")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        today = pd.Timestamp.today().normalize()

        # ── 1. Manifest: classify every 16-day window before downloading ──────
        #   CACHED file exists · FUTURE-SKIP window not begun · KNOWN-EMPTY ends
        #   before the S2_SR archive (2017-03-28) · TO-DOWNLOAD a real fetch.
        todo = []                       # (start, end_excl, dest) per TO-DOWNLOAD window
        n_cached = n_known = n_future = 0
        for year in range(max(START_YEAR, 2017), END_YEAR + 1):
            windows = _modis_16day_windows(year)
            y_cached = y_todo = y_known = y_future = 0
            for start, end in windows:
                dest      = self.out_dir / f"s2_16day_{start}.tif"
                win_start = pd.Timestamp(start)
                win_end   = pd.Timestamp(end)
                if dest.exists():
                    y_cached += 1
                elif win_start > today:
                    y_future += 1
                elif win_end < self.S2_SR_START:
                    y_known += 1
                else:
                    end_excl = (win_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    todo.append((start, end_excl, dest))
                    y_todo += 1
            n_cached += y_cached; n_known += y_known; n_future += y_future
            log.info(f"  {year}: {len(windows):2d} win — cached {y_cached}, "
                     f"to-dl {y_todo}, known-empty {y_known}, future {y_future}")
        log.info(f"  └─ {len(todo)} window(s) to download this pass "
                 f"(cached {n_cached}, known-empty {n_known}, future {n_future})")

        # ── 2. Download loop — each window under a hard wall-clock cap ─────────
        bbox = _aoi_region()
        saved, empty, failed = 0, [], []
        for i, (start, end_excl, dest) in enumerate(todo, 1):
            name = f"s2_16day_{start}"
            log.info(f"  [{i}/{len(todo)}] {name}  (→ {end_excl}) ...")
            box = {"result": "failed"}

            def _work(box=box, start=start, end_excl=end_excl, dest=dest, name=name):
                try:
                    box["result"] = self._fetch_window(ee, start, end_excl,
                                                        dest, name, bbox)
                except Exception as exc:               # any EE/network/IO death
                    box["result"], box["err"] = "failed", repr(exc)

            t = threading.Thread(target=_work, daemon=True)
            t.start()
            t.join(self.WINDOW_CAP_S)
            if t.is_alive():                           # overran the cap → hung RPC
                failed.append(name)
                log.error(f"      ✗ {name} exceeded {self.WINDOW_CAP_S}s wall-clock — "
                          f"abandoned (hung RPC); driver will retry next pass")
            else:
                r = box["result"]
                if r == "saved":
                    saved += 1
                    log.info(f"      ✓ saved {dest.name}")
                elif r == "empty":
                    empty.append(name)
                    log.warning(f"      no S2 scenes for {name} — skipped (empty window)")
                else:
                    failed.append(name)
                    log.error(f"      ✗ {name} failed"
                              + (f": {box['err']}" if "err" in box else ""))
            time.sleep(1)

        # ── 3. Gap report — empties are expected, NOT counted as real gaps ─────
        empty_set = set(empty)
        still_missing = [f"s2_16day_{s}" for (s, _e, d) in todo
                         if not d.exists() and f"s2_16day_{s}" not in empty_set]
        log.info("  ┌─ Sentinel-2 16-day run summary ────────────────────────────")
        log.info(f"  │  saved this pass : {saved}")
        log.info(f"  │  empty windows   : {len(empty)} (no cloud-free scenes — expected)")
        log.info(f"  │  failed this pass: {len(failed)}  {failed or ''}")
        if still_missing:
            log.warning(f"  │  REAL GAPS REMAIN ({len(still_missing)}): {still_missing}")
            log.warning("  │  → re-run `gee_extractor.py --module sentinel2_16day` to mop up")
        else:
            log.info("  │  no real gaps remain ✓  (empty / future / known-empty excluded)")
        log.info("  └────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — Gridded fire-season climate (ERA5-Land + CHIRPS)
# ─────────────────────────────────────────────────────────────────────────────

class ClimateExtractor:
    """
    Gridded fire-season (Feb–May) climate per year as a 6-band raster.

    This replaces the old single-point Open-Meteo weather: at province scale a
    single point cannot represent Bagmati's Terai-to-Himal climate gradient, so
    climate must be spatial.

    Sources (Google Earth Engine)
      ERA5-Land daily aggregates  ECMWF/ERA5_LAND/DAILY_AGGR   (temp, wind ~11 km)
      CHIRPS daily precipitation  UCSB-CHG/CHIRPS/DAILY        (precip ~5.5 km)

    Bands — order MUST match feature_engineering.CLIMATE_BANDS:
      1 temp_max_mean_c        mean daily Tmax over the season (°C)
      2 wind_max_mean_kmh      mean daily 10 m wind speed (km/h)
      3 precip_fire_season_mm  total season precipitation (mm)
      4 drought_factor         1/(1+P/100) season dryness (0–1)
      5 fwi_proxy              T·W·drought/(1+P/30) fire-weather proxy
      6 consec_dry_days_max    longest run of days with < 1 mm precip

    The derived indices (drought_factor, fwi_proxy) are season-scale proxies of
    the daily formulas used in the valley pipeline — adequate for static
    susceptibility; the prediction phase will compute a true daily FWI.

    Output : data/raw/climate/climate_fire_season_{year}.tif  (EPSG:4326)
    Native climate is coarse, so it is exported at SCALE_M and upsampled onto the
    100 m grid by feature_engineering.align_raster.
    """

    ERA5    = "ECMWF/ERA5_LAND/DAILY_AGGR"
    CHIRPS  = "UCSB-CHG/CHIRPS/DAILY"
    SCALE_M = 5000   # climate fields are smooth; upsampled to the grid locally

    def __init__(self):
        self.out_dir = DATA_RAW / "climate"

    def _year_image(self, year: int):
        import ee
        region   = _aoi_region()
        start    = f"{year}-{FIRE_SEASON_START}"
        end      = f"{year}-{FIRE_SEASON_END}"
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        era = (ee.ImageCollection(self.ERA5)
               .filterDate(start, end_excl).filterBounds(region))
        tmax_c = era.select("temperature_2m_max").mean().subtract(273.15).rename("temp_max_mean_c")
        u = era.select("u_component_of_wind_10m").mean()
        v = era.select("v_component_of_wind_10m").mean()
        wind_kmh = u.hypot(v).multiply(3.6).rename("wind_max_mean_kmh")

        chirps = (ee.ImageCollection(self.CHIRPS)
                  .filterDate(start, end_excl).filterBounds(region)
                  .select("precipitation"))
        precip_mm = chirps.sum().rename("precip_fire_season_mm")

        drought = precip_mm.expression(
            "1.0/(1.0 + P/100.0)", {"P": precip_mm}).rename("drought_factor")
        fwi = tmax_c.expression(
            "T * W * D / (1.0 + P/30.0)",
            {"T": tmax_c, "W": wind_kmh, "D": drought, "P": precip_mm},
        ).rename("fwi_proxy")

        # Longest consecutive dry-day run (daily precip < 1 mm) via iterate:
        #   run = (prev_run + dry) * dry   → +1 while dry, reset to 0 when wet
        dry = chirps.map(lambda img: ee.Image(img).lt(1.0).rename("dry").toFloat())
        init = ee.Image(0).rename("run").addBands(ee.Image(0).rename("max")).toFloat()

        def accum(img, prev):
            prev = ee.Image(prev)
            d   = ee.Image(img).select("dry")
            run = prev.select("run").add(d).multiply(d).rename("run")
            mx  = prev.select("max").max(run).rename("max")
            return run.addBands(mx)

        consec = ee.Image(dry.iterate(accum, init)).select("max").rename("consec_dry_days_max")

        return (tmax_c.addBands(wind_kmh).addBands(precip_mm)
                .addBands(drought).addBands(fwi).addBands(consec)
                .toFloat().clip(region))

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 7 (GEE): Gridded fire-season climate (ERA5-Land + CHIRPS)")
        log.info(f"  Period : {START_YEAR}–{END_YEAR}  fire season {FIRE_SEASON_START}–{FIRE_SEASON_END}")
        log.info(f"  Bands  : temp/wind/precip/drought/fwi/consec_dry  scale {self.SCALE_M} m")
        log.info(f"  Output : {self.out_dir}")
        log.info("=" * 60)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        region = _aoi_region()
        saved = cached = failed = 0
        for year in range(START_YEAR, END_YEAR + 1):
            dest = self.out_dir / f"climate_fire_season_{year}.tif"
            if dest.exists():
                log.info(f"    [cache] {dest.name}")
                cached += 1
                continue
            try:
                image = self._year_image(year)
            except Exception as exc:
                log.error(f"  {year}: building climate image failed: {exc}")
                failed += 1
                continue
            if _gee_download(image, region, dest, self.SCALE_M,
                             desc=f"Climate_{year}", drive_folder=DRIVE_FOLDER):
                saved += 1
            else:
                failed += 1
            time.sleep(1)
        log.info(f"  Climate complete — saved: {saved}, cached: {cached}, failed: {failed}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GEE data extractor for forest fire susceptibility project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/gee_extractor.py --authenticate
  uv run python scripts/gee_extractor.py --module dem30
  uv run python scripts/gee_extractor.py --module ndvi
  uv run python scripts/gee_extractor.py --module lst
  uv run python scripts/gee_extractor.py --module sentinel2
  uv run python scripts/gee_extractor.py --module sentinel2_16day   # 16-day S2, NOT in --module all
  uv run python scripts/gee_extractor.py --module firms
  uv run python scripts/gee_extractor.py --module burned_area
  uv run python scripts/gee_extractor.py --module all
        """,
    )
    parser.add_argument(
        "--project",
        default=GEE_PROJECT or None,
        help="GCP project ID (default: GEE_PROJECT_ID from .env)",
    )
    parser.add_argument(
        "--module",
        choices=["dem30", "ndvi", "lst", "sentinel2", "sentinel2_16day", "firms",
                 "burned_area", "climate", "all"],
        default="all",
        help="Which dataset to extract (default: all). Note: sentinel2_16day is "
             "heavy/opt-in and is NOT included in 'all'.",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Force browser OAuth2 authentication (first-time setup)",
    )
    args = parser.parse_args()

    if not args.project:
        parser.error(
            "No GCP project specified.\n"
            "  Use --project YOUR_PROJECT  or  set GEE_PROJECT_ID in .env"
        )

    _authenticate(project=args.project, force=args.authenticate)

    if args.module in ("dem30", "all"):
        DEM30Extractor().run()

    if args.module in ("ndvi", "all"):
        NDVIExtractor().run()

    if args.module in ("lst", "all"):
        LSTExtractor().run()

    if args.module in ("sentinel2", "all"):
        Sentinel2Extractor().run()

    # 16-day Sentinel-2 is heavy (~230 composites) and opt-in — NOT part of 'all'.
    if args.module == "sentinel2_16day":
        Sentinel2SixteenDayExtractor().run()

    if args.module in ("firms", "all"):
        FIRMSGEEExtractor().run()

    if args.module in ("burned_area", "all"):
        BurnedAreaExtractor().run()

    if args.module in ("climate", "all"):
        ClimateExtractor().run()

    log.info("Done.")


if __name__ == "__main__":
    main()
