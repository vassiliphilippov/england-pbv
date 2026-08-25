"""Fetch 1 m LiDAR windows (DTM + first-return DSM) and building footprints per site.

The next-generation renderer marches REAL geometry in the near field: the EA 1 m
first-return DSM carries actual tree crowns, hedgerows and buildings; DSM − DTM
separates canopy from ground; OSM footprints tag which raised cells are buildings.
Each site gets a (2*HALF_M)^2 window saved to outputs/nextgen/<key>_1m.npz with
float32 dtm/dsm (NaN where EA has no data), a uint8 building mask, and the window
origin in BNG metres.

Run: uv run python -m england_pbv.data.lidar_window outputs/nextgen_sites.json
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng

WCS_BASE: str = "https://environment.data.gov.uk/spatialdata/{slug}/wcs"
COVERAGES: dict[str, tuple[str, str]] = {
    "dtm": (
        "lidar-composite-digital-terrain-model-dtm-1m",
        "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m",
    ),
    "dsm": (
        "lidar-composite-digital-surface-model-first-return-dsm-1m",
        "df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m",
    ),
}
# One WCS request per layer: stitching quads put a seam through the camera, and a
# 0.1-0.2 m step between composite tiles throws the shading badly at grazing angles.
# 3600x3600 px is within the EA server's practical limit (~15 s, 67 MB).
HALF_M: int = 1800
OVERPASS_MIRRORS: tuple[str, str] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT: str = "england-pbv-nextgen (github.com/vassiliphilippov/england-pbv)"


def wcs_clip(kind: str, e_min: int, n_min: int, e_max: int, n_max: int) -> np.ndarray:
    slug, coverage = COVERAGES[kind]
    url = WCS_BASE.format(slug=slug)
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage,
        "SUBSET": [f"E({e_min},{e_max})", f"N({n_min},{n_max})"],
        "FORMAT": "image/tiff",
    }
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = requests.get(url, params=params, timeout=600, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            with rasterio.open(io.BytesIO(resp.content)) as src:
                band = src.read(1).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            band[band < -100.0] = np.nan
            # GeoTIFF row 0 = north; our convention row 0 = south.
            return np.asarray(band[::-1], dtype=np.float32)
        except Exception as error:  # noqa: BLE001 - retried; transient EA 500s expected
            last = error
            time.sleep(12.0 * (attempt + 1))
    raise RuntimeError(f"WCS failed for {kind} {e_min},{n_min}") from last


def fetch_grid(kind: str, x0: int, y0: int) -> np.ndarray:
    """(2*HALF_M)^2 seamless window from a single WCS request (row 0 = south)."""
    size = 2 * HALF_M
    out = np.full((size, size), np.nan, dtype=np.float32)
    band = wcs_clip(kind, x0, y0, x0 + size, y0 + size)
    h = min(band.shape[0], size)
    w = min(band.shape[1], size)
    out[:h, :w] = band[:h, :w]
    print(f"    {kind}: {h}x{w}", flush=True)
    return out


def fetch_buildings(lat: float, lon: float, x0: int, y0: int) -> np.ndarray:
    query = (
        f'[out:json][timeout:300];(way["building"](around:{HALF_M + 400},{lat},{lon}););out geom;'
    )
    payload = None
    for attempt in range(5):
        endpoint = OVERPASS_MIRRORS[attempt % 2]
        try:
            resp = requests.post(
                endpoint, data={"data": query}, timeout=600, headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception:  # noqa: BLE001
            time.sleep(15.0 * (attempt + 1))
    assert payload is not None, "Overpass buildings fetch succeeds"
    mask_img = Image.new("1", (2 * HALF_M, 2 * HALF_M), 0)
    draw = ImageDraw.Draw(mask_img)
    n = 0
    for element in payload.get("elements", []):
        geometry = element.get("geometry")
        if not geometry or len(geometry) < 3:
            continue
        ring = []
        for point in geometry:
            b = latlon_to_bng(point["lat"], point["lon"])
            ring.append((b.easting - x0, b.northing - y0))
        draw.polygon(ring, fill=1)
        n += 1
    print(f"    buildings: {n} footprints", flush=True)
    return np.array(mask_img, dtype=np.uint8)  # row 0 = south already (y up = row up)


def main() -> None:
    sites = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_dir = paths.OUTPUTS_DIR / "nextgen"
    out_dir.mkdir(parents=True, exist_ok=True)
    for site in sites:
        key = site["key"]
        out = out_dir / f"{key}_1m.npz"
        if out.exists():
            print(f"{key}: cached", flush=True)
            continue
        bng = latlon_to_bng(float(site["lat"]), float(site["lon"]))
        x0 = int(bng.easting) - HALF_M
        y0 = int(bng.northing) - HALF_M
        print(f"{key}: window origin {x0},{y0}", flush=True)
        dtm = fetch_grid("dtm", x0, y0)
        dsm = fetch_grid("dsm", x0, y0)
        buildings = fetch_buildings(float(site["lat"]), float(site["lon"]), x0, y0)
        np.savez_compressed(out, dtm=dtm, dsm=dsm, buildings=buildings, x0=x0, y0=y0)
        valid = float(np.isfinite(dtm).mean())
        print(f"{key}: saved ({valid:.0%} DTM coverage)", flush=True)


if __name__ == "__main__":
    main()
