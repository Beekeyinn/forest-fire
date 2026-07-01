import argparse
import json
import time
import zipfile
from datetime import timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from config import (
    BBOX_OVERPASS,
    BBOX_STR,
    DATA_RAW,
    END_YEAR,
    FIRMS_MAP_KEY,
    START_YEAR,
    STUDY_AREA,
    get_logger,
)
from tqdm import tqdm

log = get_logger(__name__, "download.log")


def _ensure_raw_dirs():
    for sub in ["firms", "weather", "dem", "lulc", "osm", "worldpop", "ndvi", "hansen", "gadm"]:
        (DATA_RAW / sub).mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path, desc: str = "", timeout: int = 120) -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        log.info(f"  → saved {dest.name}  ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    except requests.exceptions.HTTPError as e:
        log.error(f"  HTTP error {e.response.status_code}: {url}")
    except requests.exceptions.Timeout:
        log.error(f"  Timeout after {timeout}s: {url}")
    except Exception as e:
        log.error(f"  Download failed: {e}")
    if dest.exists():
        dest.unlink()
    return False


class FIRMSDownloader:
    """
    Products:
      MODIS_SP        — MODIS Standard Processing archive, from 2000
      VIIRS_SNPP_SP   — VIIRS Suomi-NPP Standard Processing, from 2012
      VIIRS_NOAA20_SP — VIIRS NOAA-20 Standard Processing, from 2020
    """

    API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    MAX_DAYS = 5   

    PRODUCTS = {
        "MODIS_SP"       : 2000,
        "VIIRS_SNPP_SP"  : 2012,
        "VIIRS_NOAA20_SP": 2020,
    }

    def __init__(self):
        self.out_dir = DATA_RAW / "firms"
        self.key     = FIRMS_MAP_KEY

    def _clip(self, df: pd.DataFrame) -> pd.DataFrame:
        for lat_col in ("latitude", "lat"):
            if lat_col in df.columns:
                df = df.rename(columns={lat_col: "lat"})
                break
        for lon_col in ("longitude", "lon"):
            if lon_col in df.columns:
                df = df.rename(columns={lon_col: "lon"})
                break
        if "lat" in df.columns and "lon" in df.columns:
            df = df[
                df["lat"].between(STUDY_AREA["lat_min"], STUDY_AREA["lat_max"]) &
                df["lon"].between(STUDY_AREA["lon_min"], STUDY_AREA["lon_max"])
            ]
        return df

    def _check_key(self) -> bool:
        url = (
            "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"
            f"?MAP_KEY={self.key}"
        )
        try:
            r    = requests.get(url, timeout=15)
            data = r.json()
            log.info(f"  FIRMS key status: {data}")
            limit = data.get("transaction_limit", 5000)
            used  = data.get("current_transactions", 0)
            if used >= limit:
                log.warning("  FIRMS daily transaction limit reached — try again tomorrow.")
                return False
            log.info(f"  FIRMS transactions: {used}/{limit} used")
            return True
        except Exception as e:
            log.warning(f"  Could not verify FIRMS key: {e}")
            return True   # proceed optimistically

    def fetch_year(self, product: str, year: int) -> pd.DataFrame:
        from datetime import date as date_cls

        out_csv    = self.out_dir / f"firms_{product}_{year}.csv"
        if out_csv.exists():
            log.info(f"  [cache] {out_csv.name}")
            return pd.read_csv(out_csv)

        year_start = date_cls(year, 1, 1)
        year_end   = date_cls(year, 12, 31)
        frames, cursor = [], year_start

        while cursor <= year_end:
            chunk_end = min(cursor + timedelta(days=self.MAX_DAYS - 1), year_end)
            days      = (chunk_end - cursor).days + 1
            start_str = cursor.strftime("%Y-%m-%d")
            url       = f"{self.API_BASE}/{self.key}/{product}/{BBOX_STR}/{days}/{start_str}"
            log.info(f"  GET {product} {start_str} (+{days}d) ...")
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                text = r.text.strip()
                if text and "latitude" in text.lower():
                    df = pd.read_csv(StringIO(text))
                    df = self._clip(df)
                    if not df.empty:
                        df["year"]    = year
                        df["product"] = product
                        frames.append(df)
                        log.info(f"    → {len(df)} hotspots in study area")
                    else:
                        log.info(f"    → 0 hotspots in study bbox")
                else:
                    log.info(f"    → 0 detections (empty response)")
            except requests.exceptions.HTTPError as e:
                log.error(f"    HTTP {e.response.status_code}: {e.response.text[:120]}")
            except Exception as e:
                log.error(f"    Failed: {e}")
            cursor += timedelta(days=self.MAX_DAYS)
            time.sleep(0.5)

        if frames:
            combined = pd.concat(frames, ignore_index=True).drop_duplicates()
            combined.to_csv(out_csv, index=False)
            log.info(f"  Saved → {out_csv.name}  ({len(combined):,} rows)")
            return combined

        log.info(f"  No fire detections for {product} {year} in study area.")
        return pd.DataFrame()

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 1: NASA FIRMS Historical Fire Hotspots")
        log.info(f"  Period   : {START_YEAR}–{END_YEAR}")
        log.info(f"  Products : {list(self.PRODUCTS.keys())}")
        log.info("=" * 60)

        if not self.key:
            log.warning(
                "\n"
                "  FIRMS_MAP_KEY not set — skipping fire data download.\n"
                "  Get a free key (~1 min) at:\n"
                "    https://firms.modaps.eosdis.nasa.gov/usfs/api/\n"
                "  Then add to .env:  FIRMS_MAP_KEY=your_key\n"
                "  And re-run:  uv run python scripts/download_data.py --module firms\n"
            )
            return

        if not self._check_key():
            return

        all_frames = []
        for product, avail_from in self.PRODUCTS.items():
            product_frames = []
            for year in range(max(START_YEAR, avail_from), END_YEAR + 1):
                df = self.fetch_year(product, year)
                if not df.empty:
                    product_frames.append(df)

            if product_frames:
                combined = pd.concat(product_frames, ignore_index=True).drop_duplicates()
                out      = self.out_dir / f"firms_{product}_{START_YEAR}_{END_YEAR}_bagmati.csv"
                combined.to_csv(out, index=False)
                log.info(f"  {product}: {len(combined):,} total hotspots → {out.name}")
                all_frames.append(combined)
            else:
                log.info(f"  {product}: no detections in study area for {START_YEAR}–{END_YEAR}")

        if all_frames:
            merged = pd.concat(all_frames, ignore_index=True).drop_duplicates()
            out    = self.out_dir / f"firms_all_{START_YEAR}_{END_YEAR}_bagmati.csv"
            merged.to_csv(out, index=False)
            log.info(f"  All products combined: {len(merged):,} records → {out.name}")


# Open-Meteo: Historical daily weather (free, no key)
class WeatherDownloader:
    """
    Fetches historical daily weather from the Open-Meteo Archive API.
    """
    API_URL = "https://archive-api.open-meteo.com/v1/archive"

    DAILY_VARS = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "et0_fao_evapotranspiration",
        "sunshine_duration",
        "precipitation_hours",
        "shortwave_radiation_sum",
        "vapour_pressure_deficit_max",
    ]

    def __init__(self):
        self.out_dir = DATA_RAW / "weather"
        self.lat     = STUDY_AREA["center_lat"]
        self.lon     = STUDY_AREA["center_lon"]

    def fetch_year(self, year: int) -> pd.DataFrame:
        out_csv = self.out_dir / f"weather_daily_{year}.csv"
        if out_csv.exists():
            log.info(f"  [cache] weather_daily_{year}.csv")
            return pd.read_csv(out_csv)

        from datetime import date as _date
        from datetime import timedelta as _td
        year_end    = _date(year, 12, 31)
        archive_max = _date.today() - _td(days=5)
        end_date    = min(year_end, archive_max).isoformat()

        params = {
            "latitude"  : self.lat,
            "longitude" : self.lon,
            "start_date": f"{year}-01-01",
            "end_date"  : end_date,
            "daily"     : ",".join(self.DAILY_VARS),
            "timezone"  : "Asia/Kathmandu",
        }
        log.info(f"  Fetching weather {year} from Open-Meteo ...")
        r = requests.get(self.API_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        if "error" in data:
            raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")

        df = pd.DataFrame(data["daily"])
        df.rename(columns={"time": "date"}, inplace=True)
        df["year"]   = year
        df["lat"]    = self.lat
        df["lon"]    = self.lon
        df["source"] = "Open-Meteo"

        T      = df["temperature_2m_max"].fillna(20)
        W      = df["wind_speed_10m_max"].fillna(5)
        P      = df["precipitation_sum"].fillna(0)
        P_roll = df["precipitation_sum"].rolling(30, min_periods=1).sum().fillna(0)
        df["drought_factor"] = 1 / (1 + P_roll / 10)
        df["fwi_proxy"]      = (T * W * df["drought_factor"]) / (1 + P)
        df["dry_day"]        = (P < 1.0).astype(int)

        df.to_csv(out_csv, index=False)
        log.info(f"  Saved → {out_csv.name}  ({len(df)} rows)")
        time.sleep(0.3)
        return df

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 2: Open-Meteo Historical Daily Weather")
        log.info(f"  Period   : {START_YEAR}–{END_YEAR}")
        log.info(f"  Location : {self.lat}°N, {self.lon}°E (Kathmandu Valley centroid)")
        log.info("=" * 60)
        frames = []
        for year in range(START_YEAR, END_YEAR + 1):
            try:
                frames.append(self.fetch_year(year))
            except Exception as e:
                log.error(f"  Failed year {year}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            out      = self.out_dir / f"weather_daily_{START_YEAR}_{END_YEAR}.csv"
            combined.to_csv(out, index=False)
            log.info(f"  Combined → {out.name}  ({len(combined):,} rows)")
            return combined
        return pd.DataFrame()

class DEMDownloader:
    """
    Downloads SRTM 30m elevation GeoTIFF from the CGIAR-CSI public archive.
    """
    CGIAR_BASE = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"

    def __init__(self):
        self.out_dir = DATA_RAW / "dem"

    def _tile_ids(self):
        tiles = set()
        for lon in [STUDY_AREA["lon_min"], STUDY_AREA["lon_max"]]:
            for lat in [STUDY_AREA["lat_min"], STUDY_AREA["lat_max"]]:
                x = int((lon + 180) / 5) + 1
                y = int((60 - lat)  / 5) + 1
                tiles.add((x, y))
        return sorted(tiles)

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 3: SRTM 30m DEM (CGIAR-CSI)")
        log.info("=" * 60)
        tiles = self._tile_ids()
        log.info(f"  Tiles covering study area: {tiles}")

        for (x, y) in tiles:
            name     = f"srtm_{x:02d}_{y:02d}"
            zip_path = self.out_dir / f"{name}.zip"
            tif_path = self.out_dir / f"{name}.tif"

            if tif_path.exists():
                log.info(f"  [cache] {tif_path.name}")
                continue

            url = self.CGIAR_BASE + f"{name}.zip"
            log.info(f"  Downloading {name}.zip (~35 MB) ...")
            if not _download_file(url, zip_path, desc=f"{name}.zip", timeout=180):
                log.warning(
                    "  CGIAR tile unavailable. Manual download options:\n"
                    "    NASA EarthExplorer : https://earthexplorer.usgs.gov/ (SRTM 1 Arc-Second)\n"
                    "    NASA Earthdata CMR : https://search.earthdata.nasa.gov/search?q=SRTMGL1\n"
                    "    JAXA ALOS 30m      : https://www.eorc.jaxa.jp/ALOS/en/aw3d30/\n"
                    f"  Place GeoTIFF for bbox {STUDY_AREA} in data/raw/dem/"
                )
                continue

            try:
                with zipfile.ZipFile(zip_path) as zf:
                    tifs = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                    if tifs:
                        with open(tif_path, "wb") as f:
                            f.write(zf.read(tifs[0]))
                        log.info(f"  Extracted → {tif_path.name}  ({tif_path.stat().st_size/1024/1024:.1f} MB)")
                    else:
                        log.warning(f"  No .tif found inside {name}.zip")
                zip_path.unlink(missing_ok=True)
            except Exception as e:
                log.error(f"  Failed to extract {name}.zip: {e}")

class LULCDownloader:
    """
    Downloads ESA WorldCover 2021 v200 10m GeoTIFF from the public AWS S3 bucket.
    """

    S3_BASE = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"

    def __init__(self):
        self.out_dir = DATA_RAW / "lulc"

    def _tile_names(self):
        tiles = set()
        for lat in [STUDY_AREA["lat_min"], STUDY_AREA["lat_max"]]:
            for lon in [STUDY_AREA["lon_min"], STUDY_AREA["lon_max"]]:
                lat_t = int(lat // 3) * 3
                lon_t = int(lon // 3) * 3
                lat_s = f"N{lat_t:02d}" if lat_t >= 0 else f"S{abs(lat_t):02d}"
                lon_s = f"E{lon_t:03d}" if lon_t >= 0 else f"W{abs(lon_t):03d}"
                tiles.add(f"{lat_s}{lon_s}")
        return sorted(tiles)

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 4: ESA WorldCover 2021 v200 (10m LULC)")
        log.info("=" * 60)
        tiles = self._tile_names()
        log.info(f"  Tiles: {tiles}")

        for tile in tiles:
            filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            dest     = self.out_dir / filename
            if dest.exists():
                log.info(f"  [cache] {filename}")
                continue
            url = self.S3_BASE + filename
            log.info(f"  Downloading {filename} (~200–400 MB) ...")
            if not _download_file(url, dest, desc=filename, timeout=300):
                log.warning(
                    "  ESA WorldCover S3 download failed.\n"
                    "  Alternatives:\n"
                    "    Viewer   : https://viewer.esa-worldcover.org/worldcover/\n"
                    "    Zenodo   : https://zenodo.org/records/7254221\n"
                    "  Place the downloaded .tif in data/raw/lulc/"
                )


class OSMDownloader:
    """
    Extracts roads, settlements and water features from a Geofabrik Nepal extract.
    """

    PBF_URL  = "https://download.geofabrik.de/asia/nepal-latest.osm.pbf"
    PBF_NAME = "nepal-latest.osm.pbf"

    def __init__(self):
        self.out_dir = DATA_RAW / "osm"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pbf_path = self.out_dir / self.PBF_NAME
        self.bbox = [STUDY_AREA["lon_min"], STUDY_AREA["lat_min"],
                     STUDY_AREA["lon_max"], STUDY_AREA["lat_max"]]

    def _download_pbf(self):
        if self.pbf_path.exists() and self.pbf_path.stat().st_size > 0:
            log.info(f"  [cache] {self.PBF_NAME} ({self.pbf_path.stat().st_size/1e6:.0f} MB)")
            return
        log.info(f"  Downloading {self.PBF_URL} ...")
        with requests.get(self.PBF_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(self.pbf_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        log.info(f"    {done/1e6:.0f}/{total/1e6:.0f} MB")
        log.info(f"  Saved → {self.PBF_NAME} ({self.pbf_path.stat().st_size/1e6:.0f} MB)")

    def _write_coords(self, coords, out_path: Path):
        """Write a list of (lat, lon) tuples as an elements JSON, clipped to bbox."""
        s, w = STUDY_AREA["lat_min"], STUDY_AREA["lon_min"]
        n, e = STUDY_AREA["lat_max"], STUDY_AREA["lon_max"]
        elements = [{"lat": round(lat, 6), "lon": round(lon, 6)}
                    for lat, lon in coords if s <= lat <= n and w <= lon <= e]
        out_path.write_text(json.dumps({"elements": elements}), encoding="utf-8")
        log.info(f"  Saved → {out_path.name}  ({len(elements):,} vertices, "
                 f"{out_path.stat().st_size/1024:.1f} KB)")

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 5: OpenStreetMap — Roads, Settlements & Water (Geofabrik)")
        log.info("=" * 60)
        self._download_pbf()
        try:
            import osmium
        except ImportError:
            log.error("  pyosmium not installed. Run:  uv add osmium")
            return

        s, w = STUDY_AREA["lat_min"], STUDY_AREA["lon_min"]
        n, e = STUDY_AREA["lat_max"], STUDY_AREA["lon_max"]
        PLACES = {"city", "town", "village", "hamlet"}

        class _Collector(osmium.SimpleHandler):
            """One streaming pass: bins road / settlement / water vertices, bbox-clipped."""
            def __init__(self):
                super().__init__()
                self.roads, self.settle, self.water = [], [], []

            @staticmethod
            def _inside(lat, lon):
                return s <= lat <= n and w <= lon <= e

            def node(self, node):
                loc = node.location
                if not loc.valid():
                    return
                if node.tags.get("place") in PLACES and self._inside(loc.lat, loc.lon):
                    self.settle.append((loc.lat, loc.lon))   # village/town points

            def way(self, way):
                tags = way.tags
                is_road  = "highway" in tags
                is_water = ("waterway" in tags) or (tags.get("natural") == "water")
                if not (is_road or is_water):
                    return
                for nd in way.nodes:
                    if not nd.location.valid():
                        continue
                    lat, lon = nd.location.lat, nd.location.lon
                    if not self._inside(lat, lon):
                        continue
                    if is_road:
                        self.roads.append((lat, lon))
                    if is_water:
                        self.water.append((lat, lon))

        log.info("  Parsing Nepal PBF (one streaming pass, with node locations) ...")
        handler = _Collector()
        handler.apply_file(str(self.pbf_path), locations=True)
        log.info(f"  Extracted: {len(handler.roads):,} road / "
                 f"{len(handler.settle):,} settlement / {len(handler.water):,} water vertices")

        self._write_coords(handler.roads,  self.out_dir / "osm_roads.json")
        self._write_coords(handler.settle, self.out_dir / "osm_settlements.json")
        self._write_coords(handler.water,  self.out_dir / "osm_water.json")


# WorldPop: Nepal 100m population density (free)

class WorldPopDownloader:
    """
    Downloads WorldPop 100m constrained UN-adjusted population density for Nepal.
    """

    BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/NPL/"
    FILENAME = "npl_ppp_2020_constrained.tif"

    def __init__(self):
        self.out_dir = DATA_RAW / "worldpop"

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 6: WorldPop Nepal 100m Population Density (2020)")
        log.info("=" * 60)
        dest = self.out_dir / self.FILENAME
        if dest.exists():
            log.info(f"  [cache] {self.FILENAME}")
            return
        url = self.BASE_URL + self.FILENAME
        log.info(f"  Downloading {self.FILENAME} (~25 MB) ...")
        _download_file(url, dest, desc=self.FILENAME, timeout=120)


# Hansen GFC: Global Forest Change v1.11 (free, Google Cloud)

class HansenDownloader:
    """
    Downloads Hansen Global Forest Change v1.11 tiles from Google Cloud Storage.
    """
    GCS_BASE   = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/"
    TILE       = "30N_080E"
    FILES      = [
        ("treecover2000", "~1.5 GB"),
        ("lossyear",      "~300 MB"),
    ]
    MANUAL_URL = "https://glad.umd.edu/dataset/global-2010-tree-cover-70"

    def __init__(self):
        self.out_dir = DATA_RAW / "hansen"

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 7: Hansen Global Forest Change v1.11")
        log.info(f"  Tile : {self.TILE}")
        log.info("=" * 60)
        log.warning(
            "  NOTE: Hansen treecover2000 tile is ~1.5 GB — ensure sufficient disk space."
        )

        for layer, size in self.FILES:
            filename = f"Hansen_GFC-2023-v1.11_{layer}_{self.TILE}.tif"
            dest     = self.out_dir / filename
            if dest.exists():
                log.info(f"  [cache] {filename}")
                continue
            url = self.GCS_BASE + filename
            log.info(f"  Downloading {filename} ({size}) ...")
            if not _download_file(url, dest, desc=filename, timeout=600):
                log.warning(
                    f"  Hansen download failed for {filename}.\n"
                    f"  Manual download: {self.MANUAL_URL}\n"
                    f"  Place the downloaded .tif in data/raw/hansen/"
                )


# GADM: Nepal admin boundaries Level 3 (free, UC Davis)

class GADMDownloader:
    """
    Downloads GADM 4.1 Nepal Level-3 administrative boundaries as GeoJSON.
    """
    URL      = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NPL_3.json"
    FILENAME = "nepal_admin_level3.geojson"

    def __init__(self):
        self.out_dir = DATA_RAW / "gadm"

    def run(self):
        log.info("=" * 60)
        log.info("MODULE 8: GADM Nepal Admin Boundaries (Level 3)")
        log.info("=" * 60)
        dest = self.out_dir / self.FILENAME
        if dest.exists():
            log.info(f"  [cache] {self.FILENAME}")
            return
        log.info(f"  Downloading {self.FILENAME} ...")
        if not _download_file(self.URL, dest, desc=self.FILENAME, timeout=120):
            log.warning(
                "  GADM download failed.\n"
                "  Manual download:\n"
                f"    curl -L '{self.URL}' -o data/raw/gadm/{self.FILENAME}\n"
                "  Or visit: https://gadm.org/download_country.html  (select Nepal, Level 3, GeoJSON)"
            )



MODULES = {
    "firms"   : FIRMSDownloader,
    "weather" : WeatherDownloader,
    "dem"     : DEMDownloader,
    "lulc"    : LULCDownloader,
    "osm"     : OSMDownloader,
    "worldpop": WorldPopDownloader,
    "hansen"  : HansenDownloader,
    "gadm"    : GADMDownloader,
}


def main():
    parser = argparse.ArgumentParser(
        description="Forest Fire Susceptibility — Raw Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--module",
        choices=list(MODULES.keys()) + ["all"],
        default="all",
        help="Which data module to run (default: all)",
    )
    args = parser.parse_args()

    _ensure_raw_dirs()

    if args.module == "all":
        targets = [m for m in MODULES if m != "weather"]
    else:
        targets = [args.module]

    for name in targets:
        try:
            MODULES[name]().run()
        except Exception as e:
            log.error(f"Module '{name}' failed: {e}", exc_info=True)

    log.info("Downloads complete. Check data/raw/ for results.")


if __name__ == "__main__":
    main()
