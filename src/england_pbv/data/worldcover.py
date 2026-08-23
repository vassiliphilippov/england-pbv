"""Sample ESA WorldCover 10 m land-cover classes onto the national 50 m BNG grid."""

from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from pyproj import Transformer

from england_pbv.constants import (
    BNG_EPSG,
    GRID_CELL_M,
    GRID_HEIGHT_CELLS,
    GRID_WIDTH_CELLS,
    WGS84_EPSG,
    WORLDCOVER_NODATA,
)

TRANSFORM_CHUNK_ROWS: int = 500


def _grid_latlon() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Lat/lon of every national grid cell centre, computed in row chunks."""
    transformer = Transformer.from_crs(BNG_EPSG, WGS84_EPSG, always_xy=True)
    lats = np.empty((GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), dtype=np.float32)
    lons = np.empty((GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), dtype=np.float32)
    eastings = (np.arange(GRID_WIDTH_CELLS, dtype=np.float64) + 0.5) * GRID_CELL_M
    for row_start in range(0, GRID_HEIGHT_CELLS, TRANSFORM_CHUNK_ROWS):
        row_stop = min(row_start + TRANSFORM_CHUNK_ROWS, GRID_HEIGHT_CELLS)
        n_rows = row_stop - row_start
        northings = (np.arange(row_start, row_stop, dtype=np.float64) + 0.5) * GRID_CELL_M
        ee = np.broadcast_to(eastings, (n_rows, GRID_WIDTH_CELLS)).ravel()
        nn = np.repeat(northings, GRID_WIDTH_CELLS)
        lon, lat = transformer.transform(ee, nn)
        lats[row_start:row_stop] = np.asarray(lat, dtype=np.float32).reshape(n_rows, -1)
        lons[row_start:row_stop] = np.asarray(lon, dtype=np.float32).reshape(n_rows, -1)
    return lats, lons


def build_landcover_grid(worldcover_dir: Path) -> NDArray[np.uint8]:
    print("computing grid cell lat/lon...")
    lats, lons = _grid_latlon()
    landcover = np.full(
        (GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), WORLDCOVER_NODATA, dtype=np.uint8
    )
    tile_paths = sorted(worldcover_dir.glob("*.tif"))
    assert len(tile_paths) > 0, "WorldCover tiles are present"
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as dataset:
            a, _, c, _, e, f = dataset.transform[:6]
            bounds = dataset.bounds
            mask = (
                (lons >= bounds.left)
                & (lons < bounds.right)
                & (lats > bounds.bottom)
                & (lats <= bounds.top)
            )
            n_cells = int(np.sum(mask))
            if n_cells == 0:
                print(f"{tile_path.name}: no grid cells, skipped")
                continue
            data = dataset.read(1)
            cols = ((lons[mask] - c) / a).astype(np.int64)
            rows = ((lats[mask] - f) / e).astype(np.int64)
            np.clip(cols, 0, data.shape[1] - 1, out=cols)
            np.clip(rows, 0, data.shape[0] - 1, out=rows)
            landcover[mask] = data[rows, cols]
            print(f"{tile_path.name}: filled {n_cells} cells")
    return landcover
