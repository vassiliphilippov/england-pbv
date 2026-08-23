"""Fetch OSM footpaths around the calibration photo sites and burn them into the 10 m grid.

The renderer draws a worn trodden line wherever the 10 m land-cover code carries a +1 path
flag (``code % 10 == 1``). This module is the reproducible source of those flags: Overpass
ways tagged highway=path/footway/track/bridleway/steps within 2.5 km of each calibration
photo location, rasterized onto the England 10 m land-cover grid at 5 m steps along every
segment. Burning is idempotent — an already-flagged cell no longer ends in 0 and is skipped.

Run: uv run python -m england_pbv.data.calibration_paths
"""

import json
import time

import numpy as np
import requests

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng

PHOTO_SET_FILES: tuple[str, str, str] = (
    "render_calibration_photos.json",
    "render_calibration_photos_set2.json",
    "render_calibration_photos_set3.json",
)
OVERPASS_MIRRORS: tuple[str, str] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
SEARCH_RADIUS_M: int = 2500
SAMPLE_STEP_M: float = 5.0
GRID_X0_M: float = 80000.0
GRID_Y0_M: float = 4000.0
GRID_CELL_M: float = 10.0
USER_AGENT: str = "england-pbv-calibration-paths (github.com/vassiliphilippov/england-pbv)"

# One way is its node chain as [lat, lon] pairs; a site maps to its list of ways.
SiteWays = list[list[list[float]]]


def fetch_site_ways(lat: float, lon: float) -> SiteWays:
    query = (
        f"[out:json][timeout:180];"
        f"way(around:{SEARCH_RADIUS_M},{lat},{lon})"
        f'["highway"~"^(path|footway|track|bridleway|steps)$"];'
        f"out geom;"
    )
    last_error: Exception | None = None
    for attempt in range(4):
        endpoint = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=240,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            elements = resp.json()["elements"]
            return [
                [[point["lat"], point["lon"]] for point in element["geometry"]]
                for element in elements
                if element.get("geometry")
            ]
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = error
            time.sleep(10.0 * (attempt + 1))
    raise RuntimeError(f"Overpass failed after retries for {lat},{lon}") from last_error


def fetch_all_paths() -> dict[str, SiteWays]:
    """Fetch ways for every calibration site, keeping already-fetched sites (incremental)."""
    collected: dict[str, SiteWays] = {}
    if paths.CALIBRATION_PATHS_JSON.exists():
        collected = json.loads(paths.CALIBRATION_PATHS_JSON.read_text(encoding="utf-8"))
    for set_file in PHOTO_SET_FILES:
        entries = json.loads((paths.VERIFICATION_DIR / set_file).read_text(encoding="utf-8"))
        for entry in entries:
            key = str(entry["key"])
            if key in collected:
                continue
            collected[key] = fetch_site_ways(float(entry["lat"]), float(entry["lon"]))
            paths.CALIBRATION_PATHS_JSON.write_text(json.dumps(collected), encoding="utf-8")
            print(f"{key}: {len(collected[key])} ways")
    return collected


def burn_paths(site_ways: dict[str, SiteWays]) -> int:
    """Flag path cells (+1) on the 10 m land-cover grid; returns newly burned cell count."""
    grid = np.load(paths.LANDCOVER10_GRID_NPY, mmap_mode="r+")
    height, width = grid.shape
    burned = 0
    for ways in site_ways.values():
        for way in ways:
            points = [latlon_to_bng(float(lat), float(lon)) for lat, lon in way]
            for a, b in zip(points, points[1:], strict=False):
                dx = b.easting - a.easting
                dy = b.northing - a.northing
                length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                steps = int(length / SAMPLE_STEP_M) + 1
                for i in range(steps + 1):
                    t = i / steps
                    col = int((a.easting + dx * t - GRID_X0_M) / GRID_CELL_M)
                    row = int((a.northing + dy * t - GRID_Y0_M) / GRID_CELL_M)
                    if 0 <= col < width and 0 <= row < height:
                        code = int(grid[row, col])
                        # Only flag real land classes: never nodata (0) or water (80),
                        # and only codes still ending in 0 (idempotent re-runs).
                        if code != 0 and code != 80 and code % 10 == 0:
                            grid[row, col] = code + 1
                            burned += 1
    grid.flush()
    return burned


def main() -> None:
    site_ways = fetch_all_paths()
    burned = burn_paths(site_ways)
    print(f"burned {burned} new path cells across {len(site_ways)} sites")


if __name__ == "__main__":
    main()
