"""Rasterize the England boundary (ONS Countries, EPSG:4326) onto the 50 m BNG grid."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation

from england_pbv.constants import (
    BNG_EPSG,
    GRID_CELL_M,
    GRID_HEIGHT_CELLS,
    GRID_WIDTH_CELLS,
    WGS84_EPSG,
)

COASTAL_DILATION_CELLS: int = 2  # keep clifftop cells the generalised boundary clips off

type _Ring = list[list[float]]


def _reproject_multipolygon(
    coordinates: list[list[_Ring]],
    transformer: Transformer,
) -> list[list[_Ring]]:
    reprojected: list[list[_Ring]] = []
    for polygon in coordinates:
        new_polygon: list[_Ring] = []
        for ring in polygon:
            arr = np.asarray(ring, dtype=np.float64)
            xs, ys = transformer.transform(arr[:, 0], arr[:, 1])
            new_polygon.append(np.column_stack([xs, ys]).tolist())
        reprojected.append(new_polygon)
    return reprojected


def build_england_mask(boundary_geojson: Path) -> NDArray[np.uint8]:
    payload = json.loads(boundary_geojson.read_text(encoding="utf-8"))
    features = payload["features"]
    assert len(features) >= 1, "boundary file contains England"
    transformer = Transformer.from_crs(WGS84_EPSG, BNG_EPSG, always_xy=True)

    shapes: list[dict[str, object]] = []
    for feature in features:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            coords = [geometry["coordinates"]]
        else:
            assert geometry["type"] == "MultiPolygon", "boundary is polygonal"
            coords = geometry["coordinates"]
        shapes.append(
            {
                "type": "MultiPolygon",
                "coordinates": _reproject_multipolygon(coords, transformer=transformer),
            }
        )

    north_up = from_origin(0.0, GRID_HEIGHT_CELLS * GRID_CELL_M, GRID_CELL_M, GRID_CELL_M)
    mask_north_up = rasterize(
        [(shape, 1) for shape in shapes],
        out_shape=(GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS),
        transform=north_up,
        fill=0,
        dtype="uint8",
    )
    mask = np.flipud(mask_north_up).copy()
    dilated = binary_dilation(mask.astype(bool), iterations=COASTAL_DILATION_CELLS)
    result: NDArray[np.uint8] = dilated.astype(np.uint8)
    assert 40_000_000 < int(result.sum()) < 70_000_000, "England mask has a plausible cell count"
    return result
