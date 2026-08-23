"""Download all open datasets required by the pipeline into data/raw.

Every endpoint here was verified live on 2026-08-23; see specifications/data_sources.md.
Run: uv run python -m england_pbv.data.download [--only NAME]
"""

import argparse
import time
from pathlib import Path

import requests

from england_pbv import paths

OS_TERRAIN50_URL: str = (
    "https://api.os.uk/downloads/v1/products/Terrain50/downloads"
    "?area=GB&format=ASCII+Grid+and+GML+%28Grid%29&redirect"
)

WORLDCOVER_URL_TEMPLATE: str = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
# 3x3 degree tiles covering Great Britain up to 57N (N54E000 is open sea: skipped).
WORLDCOVER_TILES: list[str] = [
    "N48W006",
    "N48W003",
    "N48E000",
    "N51W006",
    "N51W003",
    "N51E000",
    "N54W006",
    "N54W003",
]

# hill-bagging.co.uk is the canonical host but rejects some networks; legacy mirror as fallback.
DOBIH_URLS: list[str] = [
    "https://www.hill-bagging.co.uk/dobih-downloads/hillcsv.zip",
    "https://www.hills-database.co.uk/hillcsv.zip",
]
USER_AGENT: str = "england-pbv/0.1 (open-data research; vassiliphilippov@gmail.com)"
SCENICORNOT_URL: str = "http://scenicornot.datasciencelab.co.uk/votes.tsv"

ONS_COUNTRIES_URL: str = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Countries_December_2024_Boundaries_UK_{variant}/FeatureServer/0/query"
    "?where=CTRY24NM%3D%27England%27&outFields=CTRY24NM&f=geojson&outSR=4326"
)

OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
ENGLAND_AREA_ID: int = 3600058447  # OSM relation 58447 (England) + 3600000000

OVERPASS_QUERIES: dict[str, str] = {
    "viewpoints": (
        f"[out:json][timeout:180];area({ENGLAND_AREA_ID})->.a;"
        'nwr["tourism"="viewpoint"](area.a);out center;'
    ),
    "peaks": (
        f'[out:json][timeout:180];area({ENGLAND_AREA_ID})->.a;node["natural"="peak"](area.a);out;'
    ),
    "places": (
        f"[out:json][timeout:300];area({ENGLAND_AREA_ID})->.a;"
        'node["place"~"^(city|town|village)$"](area.a);out;'
    ),
}

DOWNLOAD_CHUNK_BYTES: int = 1 << 20
REQUEST_TIMEOUT_S: float = 600.0


def _stream_download(url: str, destination: Path, description: str) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {description}: already at {destination}")
        return
    print(f"[get ] {description}: {url}")
    started = time.time()
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}
    with requests.get(
        url, stream=True, timeout=REQUEST_TIMEOUT_S, allow_redirects=True, headers=headers
    ) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    tmp_path.rename(destination)
    size_mb = destination.stat().st_size / 1.0e6
    print(f"[done] {description}: {size_mb:.1f} MB in {time.time() - started:.0f}s")


def download_terrain50() -> None:
    _stream_download(OS_TERRAIN50_URL, destination=paths.TERRAIN50_ZIP, description="OS Terrain 50")


def download_worldcover() -> None:
    for tile in WORLDCOVER_TILES:
        _stream_download(
            WORLDCOVER_URL_TEMPLATE.format(tile=tile),
            destination=paths.WORLDCOVER_DIR / f"{tile}.tif",
            description=f"WorldCover {tile}",
        )


def download_dobih() -> None:
    destination = paths.DOBIH_CSV.with_suffix(".zip")
    last_error: Exception | None = None
    for url in DOBIH_URLS:
        try:
            _stream_download(url, destination=destination, description="DoBIH")
            return
        except requests.RequestException as error:
            last_error = error
            print(f"[warn] DoBIH mirror failed ({url}): {error}")
    raise RuntimeError(f"all DoBIH mirrors failed: {last_error}")


def download_scenicornot() -> None:
    _stream_download(
        SCENICORNOT_URL,
        destination=paths.RAW_DIR / "scenicornot_votes.tsv",
        description="ScenicOrNot votes archive",
    )


def download_england_boundary() -> None:
    if paths.ENGLAND_BOUNDARY_GEOJSON.exists():
        print("[skip] England boundary: already present")
        return
    # BGC (20 m generalised) preferred; BUC (500 m) fallback.
    for variant in ("BGC", "BUC"):
        url = ONS_COUNTRIES_URL.format(variant=variant)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        if resp.status_code == 200 and b'"FeatureCollection"' in resp.content[:200]:
            paths.ENGLAND_BOUNDARY_GEOJSON.write_bytes(resp.content)
            print(f"[done] England boundary ({variant}): {len(resp.content) / 1e6:.1f} MB")
            return
    raise RuntimeError("England boundary download failed for both BGC and BUC variants")


def download_osm() -> None:
    targets: dict[str, Path] = {
        "viewpoints": paths.OSM_VIEWPOINTS_JSON,
        "peaks": paths.OSM_PEAKS_JSON,
        "places": paths.OSM_PLACES_JSON,
    }
    for name, destination in targets.items():
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[skip] OSM {name}: already present")
            continue
        print(f"[get ] OSM {name} via Overpass")
        resp = requests.post(
            OVERPASS_URL,
            data={"data": OVERPASS_QUERIES[name]},
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        destination.write_bytes(resp.content)
        print(f"[done] OSM {name}: {len(resp.content) / 1e6:.1f} MB")
        time.sleep(5.0)  # polite gap between Overpass queries


DOWNLOADERS: dict[str, object] = {
    "terrain50": download_terrain50,
    "worldcover": download_worldcover,
    "dobih": download_dobih,
    "scenicornot": download_scenicornot,
    "boundary": download_england_boundary,
    "osm": download_osm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets")
    parser.add_argument("--only", choices=sorted(DOWNLOADERS), default=None)
    args = parser.parse_args()

    paths.ensure_dirs()
    names = [args.only] if args.only is not None else list(DOWNLOADERS)
    for name in names:
        downloader = DOWNLOADERS[name]
        assert callable(downloader), "downloader is callable"
        downloader()


if __name__ == "__main__":
    main()
