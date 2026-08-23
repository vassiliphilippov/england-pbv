"""Fetch Priority Habitats Inventory polygons around calibration sites and flag the 10 m grid.

Altitude is a poor proxy for vegetation: Mam Tor's 517 m crest is grazed green sward while
Valley of Rocks' bracken heath sits near sea level. Natural England's Priority Habitats
Inventory (open data, EPSG:27700 polygons) records where moorland vegetation actually is.
Cells inside moor/heath/bog habitat polygons get a +2 flag on the 10 m land-cover code
(``code % 10 == 2``; the +1 path flag wins where both apply), and the renderer colours
moor grass, bracken and heather from the flag instead of altitude.

Run: uv run python -m england_pbv.data.habitats
"""

import json
import time
from typing import cast

import numpy as np
import requests
from PIL import Image, ImageDraw

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng

PHI_QUERY_URL: str = (
    "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
    "Priority_Habitats_Inventory_England/FeatureServer/0/query"
)
PHOTO_SET_FILES: tuple[str, str] = (
    "render_calibration_photos.json",
    "render_calibration_photos_set2.json",
)
SEARCH_RADIUS_M: float = 3000.0
PAGE_SIZE: int = 2000
GRID_X0_M: float = 80000.0
GRID_Y0_M: float = 4000.0
GRID_CELL_M: float = 10.0
USER_AGENT: str = "england-pbv-habitats (github.com/vassiliphilippov/england-pbv)"

# MainHabs substrings that read as open moor/heath/bracken country in a photo.
MOOR_HABITAT_SUBSTRINGS: tuple[str, ...] = (
    "heath",
    "blanket bog",
    "raised bog",
    "purple moor grass",
    "maritime cliff",
    "flushes",
)

# One site's habitat features: [main_habs, [ring, hole, ...]] with rings as [e, n] pairs.
SiteFeatures = list[list[object]]


def is_moor_habitat(main_habs: str) -> bool:
    lowered = main_habs.lower()
    return any(marker in lowered for marker in MOOR_HABITAT_SUBSTRINGS)


def fetch_site_features(easting: float, northing: float) -> SiteFeatures:
    envelope = (
        f"{easting - SEARCH_RADIUS_M},{northing - SEARCH_RADIUS_M},"
        f"{easting + SEARCH_RADIUS_M},{northing + SEARCH_RADIUS_M}"
    )
    features: SiteFeatures = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "geometry": envelope,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "27700",
            "outSR": "27700",
            "outFields": "MainHabs",
            "f": "geojson",
            "resultOffset": str(offset),
            "resultRecordCount": str(PAGE_SIZE),
        }
        payload = None
        for attempt in range(4):
            try:
                resp = requests.get(
                    PHI_QUERY_URL, params=params, timeout=180, headers={"User-Agent": USER_AGENT}
                )
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(8.0 * (attempt + 1))
        assert payload is not None, "PHI query succeeds after retries"
        page = payload.get("features", [])
        for feature in page:
            main_habs = str(feature["properties"].get("MainHabs") or "")
            geometry = feature.get("geometry") or {}
            rings: list[list[list[float]]] = []
            if geometry.get("type") == "Polygon":
                rings = [[[pt[0], pt[1]] for pt in ring] for ring in geometry["coordinates"]]
            elif geometry.get("type") == "MultiPolygon":
                for polygon in geometry["coordinates"]:
                    rings.extend([[pt[0], pt[1]] for pt in ring] for ring in polygon)
            if rings:
                features.append([main_habs, rings])
        if len(page) < PAGE_SIZE:
            return features
        offset += PAGE_SIZE


def fetch_all_habitats() -> dict[str, SiteFeatures]:
    """Fetch PHI features for every calibration site, keeping already-fetched sites."""
    collected: dict[str, SiteFeatures] = {}
    if paths.CALIBRATION_HABITATS_JSON.exists():
        collected = json.loads(paths.CALIBRATION_HABITATS_JSON.read_text(encoding="utf-8"))
    for set_file in PHOTO_SET_FILES:
        entries = json.loads((paths.VERIFICATION_DIR / set_file).read_text(encoding="utf-8"))
        for entry in entries:
            key = str(entry["key"])
            if key in collected:
                continue
            bng = latlon_to_bng(float(entry["lat"]), float(entry["lon"]))
            collected[key] = fetch_site_features(bng.easting, bng.northing)
            paths.CALIBRATION_HABITATS_JSON.write_text(json.dumps(collected), encoding="utf-8")
            moor = sum(1 for feature in collected[key] if is_moor_habitat(str(feature[0])))
            print(f"{key}: {len(collected[key])} features ({moor} moor-class)")
    return collected


def burn_habitats(site_features: dict[str, SiteFeatures]) -> int:
    """Flag moor-habitat cells (+2) on the 10 m land-cover grid; returns new cell count."""
    grid = np.load(paths.LANDCOVER10_GRID_NPY, mmap_mode="r+")
    grid_height, grid_width = grid.shape
    burned = 0
    for features in site_features.values():
        moor_rings = [
            cast(list[list[list[float]]], feature[1])
            for feature in features
            if is_moor_habitat(str(feature[0]))
        ]
        if not moor_rings:
            continue
        all_points = [pt for rings in moor_rings for ring in rings for pt in ring]
        cols = [int((float(pt[0]) - GRID_X0_M) / GRID_CELL_M) for pt in all_points]
        rows = [int((float(pt[1]) - GRID_Y0_M) / GRID_CELL_M) for pt in all_points]
        col0 = max(0, min(cols) - 1)
        row0 = max(0, min(rows) - 1)
        col1 = min(grid_width, max(cols) + 2)
        row1 = min(grid_height, max(rows) + 2)
        if col1 <= col0 or row1 <= row0:
            continue
        mask_img = Image.new("1", (col1 - col0, row1 - row0), 0)
        draw = ImageDraw.Draw(mask_img)
        for rings in moor_rings:
            # GeoJSON ring order: exterior first, then holes — paint, then punch holes.
            for ring_index, ring in enumerate(rings):
                pixel_ring = [
                    (
                        (float(pt[0]) - GRID_X0_M) / GRID_CELL_M - col0,
                        (float(pt[1]) - GRID_Y0_M) / GRID_CELL_M - row0,
                    )
                    for pt in ring
                ]
                if len(pixel_ring) >= 3:
                    draw.polygon(pixel_ring, fill=0 if ring_index > 0 else 1)
        mask = np.array(mask_img, dtype=bool)
        window = grid[row0:row1, col0:col1]
        # Only flag real land classes still ending in 0: never nodata (0) or water (80),
        # and never a cell already carrying the path flag (paths win visually).
        eligible = mask & (window != 0) & (window != 80) & (window % 10 == 0)
        window[eligible] += 2
        burned += int(eligible.sum())
    grid.flush()
    return burned


def main() -> None:
    site_features = fetch_all_habitats()
    burned = burn_habitats(site_features)
    print(f"flagged {burned} new moor-habitat cells across {len(site_features)} sites")


if __name__ == "__main__":
    main()
