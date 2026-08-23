"""Fetch OSM rock features around calibration sites and flag the 10 m grid.

Tors, crags and limestone pavements define photos (Castle Rock at Valley of Rocks, the
Simonside gritstone tors, the Whin Sill cliffs) but no land-cover product maps them at
10 m. OSM does: ``natural=bare_rock/scree/rock/stone`` areas and ``natural=cliff`` lines.
Cells get a +3 flag on the 10 m land-cover code (``code % 10 == 3``); the renderer paints
them as grey crag instead of vegetation. The +1 path flag is preserved; the +2 moor flag
is overwritten (rock wins where OSM maps actual outcrop). Only ways are fetched —
multipolygon rock relations are rare at these sites and skipped for simplicity.

Run: uv run python -m england_pbv.data.rocks
"""

import json
import time
from typing import cast

import numpy as np
import requests
from PIL import Image, ImageDraw

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng

PHOTO_SET_FILES: tuple[str, str] = (
    "render_calibration_photos.json",
    "render_calibration_photos_set2.json",
)
OVERPASS_MIRRORS: tuple[str, str] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
SEARCH_RADIUS_M: int = 3000
LINE_STEP_M: float = 5.0
GRID_X0_M: float = 80000.0
GRID_Y0_M: float = 4000.0
GRID_CELL_M: float = 10.0
USER_AGENT: str = "england-pbv-rocks (github.com/vassiliphilippov/england-pbv)"

# One way: [natural_tag, [[lat, lon], ...]]. Closed area tags fill; cliff lines rasterize.
SiteWays = list[list[object]]
AREA_TAGS: tuple[str, ...] = ("bare_rock", "scree", "rock", "stone", "shingle")


def fetch_site_rocks(lat: float, lon: float) -> SiteWays:
    query = (
        f"[out:json][timeout:180];"
        f"way(around:{SEARCH_RADIUS_M},{lat},{lon})"
        f'["natural"~"^(bare_rock|scree|cliff|rock|stone|shingle)$"];'
        f"out tags geom;"
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
                [
                    str(element["tags"].get("natural", "")),
                    [[point["lat"], point["lon"]] for point in element["geometry"]],
                ]
                for element in elements
                if element.get("geometry")
            ]
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = error
            time.sleep(10.0 * (attempt + 1))
    raise RuntimeError(f"Overpass failed after retries for {lat},{lon}") from last_error


def fetch_all_rocks() -> dict[str, SiteWays]:
    collected: dict[str, SiteWays] = {}
    if paths.CALIBRATION_ROCKS_JSON.exists():
        collected = json.loads(paths.CALIBRATION_ROCKS_JSON.read_text(encoding="utf-8"))
    for set_file in PHOTO_SET_FILES:
        entries = json.loads((paths.VERIFICATION_DIR / set_file).read_text(encoding="utf-8"))
        for entry in entries:
            key = str(entry["key"])
            if key in collected:
                continue
            collected[key] = fetch_site_rocks(float(entry["lat"]), float(entry["lon"]))
            paths.CALIBRATION_ROCKS_JSON.write_text(json.dumps(collected), encoding="utf-8")
            print(f"{key}: {len(collected[key])} rock ways")
    return collected


def _flag_cell(grid: "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]", row: int, col: int) -> int:
    """Set the +3 rock flag when the cell is a real land class; returns 1 when changed."""
    code = int(grid[row, col])
    last = code % 10
    # Never nodata (0) or water (80); keep the path flag; overwrite the moor flag.
    if code == 0 or code - last == 80 or last == 1 or last == 3:
        return 0
    grid[row, col] = code - last + 3
    return 1


def burn_rocks(site_ways: dict[str, SiteWays]) -> int:
    grid = np.load(paths.LANDCOVER10_GRID_NPY, mmap_mode="r+")
    grid_height, grid_width = grid.shape
    burned = 0
    for ways in site_ways.values():
        for way in ways:
            tag = str(way[0])
            latlon = cast(list[list[float]], way[1])
            points = [latlon_to_bng(float(pt[0]), float(pt[1])) for pt in latlon]
            closed_area = (
                tag in AREA_TAGS
                and len(points) >= 4
                and abs(points[0].easting - points[-1].easting) < 1.0
                and abs(points[0].northing - points[-1].northing) < 1.0
            )
            if closed_area:
                cols = [int((p.easting - GRID_X0_M) / GRID_CELL_M) for p in points]
                rows = [int((p.northing - GRID_Y0_M) / GRID_CELL_M) for p in points]
                col0 = max(0, min(cols) - 1)
                row0 = max(0, min(rows) - 1)
                col1 = min(grid_width, max(cols) + 2)
                row1 = min(grid_height, max(rows) + 2)
                if col1 <= col0 or row1 <= row0:
                    continue
                mask_img = Image.new("1", (col1 - col0, row1 - row0), 0)
                ImageDraw.Draw(mask_img).polygon(
                    [
                        (
                            (p.easting - GRID_X0_M) / GRID_CELL_M - col0,
                            (p.northing - GRID_Y0_M) / GRID_CELL_M - row0,
                        )
                        for p in points
                    ],
                    fill=1,
                )
                mask = np.array(mask_img, dtype=bool)
                for local_row, local_col in zip(*np.nonzero(mask), strict=False):
                    burned += _flag_cell(grid, row0 + int(local_row), col0 + int(local_col))
            else:
                # Cliff (or unclosed) ways rasterize as lines, like footpaths.
                for a, b in zip(points, points[1:], strict=False):
                    dx = b.easting - a.easting
                    dy = b.northing - a.northing
                    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                    steps = int(length / LINE_STEP_M) + 1
                    for i in range(steps + 1):
                        t = i / steps
                        col = int((a.easting + dx * t - GRID_X0_M) / GRID_CELL_M)
                        row = int((a.northing + dy * t - GRID_Y0_M) / GRID_CELL_M)
                        if 0 <= col < grid_width and 0 <= row < grid_height:
                            burned += _flag_cell(grid, row, col)
    grid.flush()
    return burned


def main() -> None:
    site_ways = fetch_all_rocks()
    burned = burn_rocks(site_ways)
    print(f"flagged {burned} rock cells across {len(site_ways)} sites")


if __name__ == "__main__":
    main()
