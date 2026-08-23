"""Build a 10 m true-colour satellite mosaic of England on the British National Grid.

Sentinel-2 L2A scenes from the AWS Open Data programme (anonymous, not requester-pays)
carry a ready-made 8-bit true-colour COG (``TCI.tif``). For every MGRS tile touching
England this module picks the least-cloudy summer scene (May–September, lowest
``eo:cloud_cover``), reprojects it to EPSG:27700 at 10 m, and writes it into
``england_satellite_10m.npy`` — shape (H, W, 3) uint8, the exact frame of the 10 m
land-cover grid (x0=80000, y0=4000, row 0 = south). Only England land cells (nonzero
land cover) are filled; scenes are applied lowest-cloud-first and never overwrite.

The renderer blends this real imagery in beyond ~2 km, where procedural field patchwork
otherwise invents the landscape. Licence: CC BY-SA 3.0 IGO — the site must credit
"Contains modified Copernicus Sentinel data [year]".

Run: uv run python -m england_pbv.data.satellite
"""

import json

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from england_pbv import paths

STAC_SEARCH_URL: str = "https://earth-search.aws.element84.com/v1/search"
COLLECTION: str = "sentinel-2-l2a"
# WGS84 bbox generously covering England (neighbouring-country cells are masked out).
ENGLAND_BBOX: tuple[float, float, float, float] = (-6.7, 49.8, 2.0, 55.9)
DATE_RANGE: str = "2023-05-01T00:00:00Z/2026-09-30T23:59:59Z"
SUMMER_MONTHS: tuple[int, ...] = (5, 6, 7, 8, 9)
MAX_CLOUD_PERCENT: float = 12.0
GRID_X0_M: float = 80000.0
GRID_Y0_M: float = 4000.0
GRID_CELL_M: float = 10.0
USER_AGENT: str = "england-pbv-satellite (github.com/vassiliphilippov/england-pbv)"


def best_summer_scenes() -> list[dict[str, object]]:
    """One least-cloudy summer scene per MGRS tile, sorted lowest cloud first."""
    best: dict[str, dict[str, object]] = {}
    body: dict[str, object] = {
        "collections": [COLLECTION],
        "bbox": list(ENGLAND_BBOX),
        "datetime": DATE_RANGE,
        "query": {"eo:cloud_cover": {"lt": MAX_CLOUD_PERCENT}},
        "limit": 200,
    }
    url = STAC_SEARCH_URL
    for _page in range(60):
        resp = requests.post(url, json=body, timeout=120, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("features", []):
            properties = item["properties"]
            month = int(str(properties["datetime"])[5:7])
            if month not in SUMMER_MONTHS:
                continue
            tile = str(properties.get("grid:code", ""))
            cloud = float(properties.get("eo:cloud_cover", 100.0))
            visual = item.get("assets", {}).get("visual", {}).get("href")
            if not tile or visual is None:
                continue
            current = best.get(tile)
            if current is None or cloud < float(current["cloud"]):  # type: ignore[arg-type]
                best[tile] = {"tile": tile, "cloud": cloud, "url": str(visual), "id": item["id"]}
        next_link = next(
            (link for link in payload.get("links", []) if link.get("rel") == "next"), None
        )
        if next_link is None:
            break
        # earth-search pagination: POST the merged body from the next link.
        url = str(next_link["href"])
        merged = dict(body)
        merged.update(next_link.get("body", {}))
        body = merged
    return sorted(best.values(), key=lambda scene: float(scene["cloud"]))  # type: ignore[arg-type]


def burn_scene(
    scene_url: str,
    mosaic: np.memmap,
    landcover: np.ndarray,
) -> int:
    """Reproject one TCI scene into the BNG mosaic; returns cells newly filled."""
    grid_height, grid_width = landcover.shape
    with rasterio.open(scene_url) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:27700", *src.bounds)
        col0 = max(0, int((west - GRID_X0_M) / GRID_CELL_M))
        row0 = max(0, int((south - GRID_Y0_M) / GRID_CELL_M))
        col1 = min(grid_width, int((east - GRID_X0_M) / GRID_CELL_M) + 1)
        row1 = min(grid_height, int((north - GRID_Y0_M) / GRID_CELL_M) + 1)
        if col1 <= col0 or row1 <= row0:
            return 0
        width = col1 - col0
        height = row1 - row0
        # North-up destination for warp; our grid stores row 0 = south, so flip on write.
        dst_transform = from_origin(
            GRID_X0_M + col0 * GRID_CELL_M,
            GRID_Y0_M + row1 * GRID_CELL_M,
            GRID_CELL_M,
            GRID_CELL_M,
        )
        warped = np.zeros((3, height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, [1, 2, 3]),
            destination=warped,
            dst_transform=dst_transform,
            dst_crs="EPSG:27700",
            resampling=Resampling.bilinear,
        )
    rgb_south_up = np.moveaxis(warped, 0, -1)[::-1]
    window = mosaic[row0:row1, col0:col1]
    england = landcover[row0:row1, col0:col1] != 0
    has_data = rgb_south_up.max(axis=2) > 0
    empty = window.max(axis=2) == 0
    fill = england & has_data & empty
    window[fill] = rgb_south_up[fill]
    return int(fill.sum())


def main() -> None:
    landcover = np.load(paths.LANDCOVER10_GRID_NPY, mmap_mode="r")
    grid_height, grid_width = landcover.shape
    if paths.SATELLITE10_GRID_NPY.exists():
        mosaic = np.lib.format.open_memmap(paths.SATELLITE10_GRID_NPY, mode="r+")
    else:
        mosaic = np.lib.format.open_memmap(
            paths.SATELLITE10_GRID_NPY,
            mode="w+",
            dtype=np.uint8,
            shape=(grid_height, grid_width, 3),
        )
    scenes = best_summer_scenes()
    print(f"{len(scenes)} tiles with a summer scene under {MAX_CLOUD_PERCENT}% cloud")
    done_path = paths.RAW_DIR / "satellite_scenes_done.json"
    done: dict[str, int] = (
        json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    )
    for index, scene in enumerate(scenes):
        tile = str(scene["tile"])
        if tile in done:
            continue
        filled = burn_scene(str(scene["url"]), mosaic, landcover)
        done[tile] = filled
        done_path.write_text(json.dumps(done), encoding="utf-8")
        cloud = float(str(scene["cloud"]))
        print(
            f"[{index + 1}/{len(scenes)}] {tile} cloud={cloud:.1f}% "
            f"filled {filled} cells ({scene['id']})"
        )
    mosaic.flush()
    total = sum(done.values())
    print(f"mosaic complete: {total} England cells filled")


if __name__ == "__main__":
    main()
