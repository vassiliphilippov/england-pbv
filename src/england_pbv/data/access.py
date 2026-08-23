"""Compute a public-access flag for every scored viewpoint.

Two open datasets answer "may I legally stand here?":

- Natural England's CRoW Act 2000 Access Layer (OGL): legally mapped open-access
  land — mountain, moor, heath, down, registered common and s16 dedicated land.
  Downloaded once as bulk GeoJSON and rasterized onto the national 50 m grid.
- Public rights of way from OSM (ODbL): ways tagged with a legal ``designation``
  (public_footpath, public_bridleway, restricted_byway, byway_open_to_all_traffic),
  fetched tile-by-tile over England, sampled every 25 m into a KD-tree for exact
  nearest-distance queries. OSM is not the definitive map — the site labels access
  as computed and tells visitors to verify locally.

Classes (outputs/access_flags.json, keyed by candidate id):
- ``open``  — inside CRoW access land: freedom to roam
- ``path``  — a public right of way passes within 60 m
- ``near``  — a public right of way within 300 m
- ``none``  — no recorded public access at the point

Run: uv run python -m england_pbv.data.access            (all stages, resumable)
"""

import json
import time

import numpy as np
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.spatial import cKDTree

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng
from england_pbv.verification.evaluate import load_scored

CROW_BULK_URL: str = (
    "https://hub.arcgis.com/api/v3/datasets/6ce15f2cd06c4536983d315694dad16b_0/"
    "downloads/data?format=geojson&spatialRefId=4326"
)
OVERPASS_MIRRORS: tuple[str, str] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
PROW_DESIGNATIONS: str = (
    "public_footpath|public_bridleway|restricted_byway|byway_open_to_all_traffic"
)
# England bbox split into 2x2-degree Overpass tiles (lat 49.8..56, lon -6.5..2).
PROW_LAT_EDGES: list[float] = [49.8, 51.0, 52.0, 53.0, 54.0, 56.0]
PROW_LON_EDGES: list[float] = [-6.5, -4.5, -3.0, -1.5, 0.0, 2.0]
PATH_SAMPLE_M: float = 25.0
ON_PATH_M: float = 60.0
NEAR_PATH_M: float = 300.0
GRID_CELL_M: float = 50.0
# Mask frame: eastings/northings 0..660 km at 50 m (row 0 = south). England fits
# entirely (Berwick ~657 km N); this is NOT the frame of the full GB grids.
GRID_SHAPE: tuple[int, int] = (13200, 13200)
USER_AGENT: str = "england-pbv-access (github.com/vassiliphilippov/england-pbv)"


def download_crow() -> None:
    if paths.CROW_ACCESS_GEOJSON.exists() and paths.CROW_ACCESS_GEOJSON.stat().st_size > 10_000_000:
        return
    print("downloading CRoW access layer (bulk GeoJSON)…", flush=True)
    with requests.get(
        CROW_BULK_URL, timeout=1800, stream=True, headers={"User-Agent": USER_AGENT}
    ) as resp:
        resp.raise_for_status()
        with open(paths.CROW_ACCESS_GEOJSON, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    print(f"CRoW: {paths.CROW_ACCESS_GEOJSON.stat().st_size / 1e6:.0f} MB", flush=True)


def build_crow_mask() -> None:
    """Rasterize CRoW polygons (WGS84) onto a 50 m national mask (row 0 = south).

    Each polygon (exterior + its OWN holes) is rasterized onto a bbox-local
    scratch image and OR-ed into the national mask — painting all rings onto one
    shared canvas would let a later polygon's hole erase earlier parcels nested
    inside it (~33 km² of real access land, e.g. Beamsley Beacon's moor).
    """
    if paths.CROW_MASK_NPY.exists():
        return
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    data = json.loads(paths.CROW_ACCESS_GEOJSON.read_text(encoding="utf-8"))
    mask = np.zeros(GRID_SHAPE, dtype=np.uint8)
    n_polygons = 0
    for feature in data["features"]:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry.get("type") == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            rings_px: list[list[tuple[float, float]]] = []
            for ring in polygon:
                lons = [pt[0] for pt in ring]
                lats = [pt[1] for pt in ring]
                xs, ys = transformer.transform(lons, lats)
                rings_px.append(
                    [(x / GRID_CELL_M, y / GRID_CELL_M) for x, y in zip(xs, ys, strict=True)]
                )
            if not rings_px or len(rings_px[0]) < 3:
                continue
            all_px = [pt for ring in rings_px for pt in ring]
            col0 = max(0, int(min(p[0] for p in all_px)) - 1)
            row0 = max(0, int(min(p[1] for p in all_px)) - 1)
            col1 = min(GRID_SHAPE[1], int(max(p[0] for p in all_px)) + 2)
            row1 = min(GRID_SHAPE[0], int(max(p[1] for p in all_px)) + 2)
            if col1 <= col0 or row1 <= row0:
                continue
            scratch = Image.new("1", (col1 - col0, row1 - row0), 0)
            draw = ImageDraw.Draw(scratch)
            for ring_index, ring_px in enumerate(rings_px):
                local = [(x - col0, y - row0) for x, y in ring_px]
                if len(local) >= 3:
                    draw.polygon(local, fill=0 if ring_index > 0 else 1)
            window = mask[row0:row1, col0:col1]
            np.logical_or(window, np.array(scratch, dtype=np.uint8), out=window, casting="unsafe")
            n_polygons += 1
    # Pixel y = northing/50 and np.array(img) keeps that row order — row 0 is the
    # southernmost strip already; no flip (verified at Kinder Scout).
    np.save(paths.CROW_MASK_NPY, mask)
    print(f"CRoW mask: {n_polygons} polygons, {int(mask.sum())} open-access cells", flush=True)


def fetch_prow_tiles() -> None:
    paths.PROW_WAYS_DIR.mkdir(parents=True, exist_ok=True)
    tiles = [
        (PROW_LAT_EDGES[i], PROW_LON_EDGES[j], PROW_LAT_EDGES[i + 1], PROW_LON_EDGES[j + 1])
        for i in range(len(PROW_LAT_EDGES) - 1)
        for j in range(len(PROW_LON_EDGES) - 1)
    ]
    for index, (south, west, north, east) in enumerate(tiles):
        out = paths.PROW_WAYS_DIR / f"prow_{index:02d}.json"
        # A legitimately empty tile (North Sea) is "[]" — 2 bytes is a valid result.
        if out.exists() and out.stat().st_size >= 2:
            continue
        query = (
            f"[out:json][timeout:600];"
            f'way["designation"~"^({PROW_DESIGNATIONS})$"]({south},{west},{north},{east});'
            f"out geom;"
        )
        last_error: Exception | None = None
        for attempt in range(5):
            endpoint = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
            try:
                resp = requests.post(
                    endpoint, data={"data": query}, timeout=900, headers={"User-Agent": USER_AGENT}
                )
                resp.raise_for_status()
                payload = resp.json()
                slim = [
                    [[point["lat"], point["lon"]] for point in element["geometry"]]
                    for element in payload.get("elements", [])
                    if element.get("geometry")
                ]
                out.write_text(json.dumps(slim), encoding="utf-8")
                print(f"PRoW tile {index + 1}/{len(tiles)}: {len(slim)} ways", flush=True)
                last_error = None
                break
            except (requests.RequestException, KeyError, ValueError) as error:
                last_error = error
                time.sleep(20.0 * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(f"Overpass failed for PRoW tile {index}") from last_error


def path_sample_tree() -> cKDTree:
    """KD-tree of BNG points sampled every ~25 m along every fetched PRoW way."""
    samples: list[tuple[float, float]] = []
    for tile_file in sorted(paths.PROW_WAYS_DIR.glob("prow_*.json")):
        for way in json.loads(tile_file.read_text(encoding="utf-8")):
            points = [latlon_to_bng(float(lat), float(lon)) for lat, lon in way]
            for a, b in zip(points, points[1:], strict=False):
                dx = b.easting - a.easting
                dy = b.northing - a.northing
                length = max(1.0, float(np.hypot(dx, dy)))
                steps = int(length / PATH_SAMPLE_M) + 1
                for i in range(steps + 1):
                    t = i / steps
                    samples.append((a.easting + dx * t, a.northing + dy * t))
    print(f"PRoW samples: {len(samples)}", flush=True)
    assert len(samples) > 1_000_000, "national PRoW network expected"
    return cKDTree(np.array(samples, dtype=np.float64))


def main() -> None:
    download_crow()
    build_crow_mask()
    fetch_prow_tiles()

    mask = np.load(paths.CROW_MASK_NPY)
    tree = path_sample_tree()
    scored = load_scored()
    coords = np.array([[s.candidate.easting, s.candidate.northing] for s in scored])
    distances, _ = tree.query(coords, k=1)

    flags: dict[str, dict[str, object]] = {}
    counts = {"open": 0, "path": 0, "near": 0, "none": 0}
    for row, distance in zip(scored, distances, strict=True):
        col = int(row.candidate.easting / GRID_CELL_M)
        grid_row = int(row.candidate.northing / GRID_CELL_M)
        in_crow = (
            0 <= grid_row < GRID_SHAPE[0] and 0 <= col < GRID_SHAPE[1] and mask[grid_row, col] == 1
        )
        path_m = int(round(float(distance)))
        if in_crow:
            cls = "open"
        elif path_m <= ON_PATH_M:
            cls = "path"
        elif path_m <= NEAR_PATH_M:
            cls = "near"
        else:
            cls = "none"
        counts[cls] += 1
        flags[row.candidate.candidate_id] = {
            "cls": cls,
            "path_m": path_m if path_m <= 2000 else None,
        }
    paths.ACCESS_FLAGS_JSON.write_text(json.dumps(flags), encoding="utf-8")
    print(f"access flags: {counts} -> {paths.ACCESS_FLAGS_JSON}")


if __name__ == "__main__":
    main()
