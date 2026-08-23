"""National 50 m grid handling: array layout, coordinate conversion, load/save.

Layout convention: row index grows with northing, column index with easting.
Cell (row, col) has its centre at easting = (col + 0.5) * cell, northing = (row + 0.5) * cell.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer

from england_pbv.constants import (
    BNG_EPSG,
    GRID_CELL_M,
    GRID_HEIGHT_CELLS,
    GRID_WIDTH_CELLS,
    WGS84_EPSG,
)

_TO_WGS84: Transformer = Transformer.from_crs(BNG_EPSG, WGS84_EPSG, always_xy=True)
_FROM_WGS84: Transformer = Transformer.from_crs(WGS84_EPSG, BNG_EPSG, always_xy=True)


@dataclass(frozen=True, slots=True)
class GridPosition:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class BngPoint:
    easting: float
    northing: float


@dataclass(frozen=True, slots=True)
class LatLon:
    lat: float
    lon: float


def bng_to_cell(easting: float, northing: float) -> GridPosition:
    return GridPosition(row=int(northing // GRID_CELL_M), col=int(easting // GRID_CELL_M))


def cell_to_bng(row: int, col: int) -> BngPoint:
    return BngPoint(
        easting=(col + 0.5) * GRID_CELL_M,
        northing=(row + 0.5) * GRID_CELL_M,
    )


def bng_to_latlon(easting: float, northing: float) -> LatLon:
    lon, lat = _TO_WGS84.transform(easting, northing)
    return LatLon(lat=float(lat), lon=float(lon))


def latlon_to_bng(lat: float, lon: float) -> BngPoint:
    easting, northing = _FROM_WGS84.transform(lon, lat)
    return BngPoint(easting=float(easting), northing=float(northing))


def in_grid(row: int, col: int) -> bool:
    return 0 <= row < GRID_HEIGHT_CELLS and 0 <= col < GRID_WIDTH_CELLS


def new_dem_grid() -> NDArray[np.float32]:
    return np.zeros((GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), dtype=np.float32)


def save_grid(path: Path, grid: NDArray[np.float32] | NDArray[np.uint8]) -> None:
    assert grid.shape == (GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), "grid has national shape"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid)


def load_dem_grid(path: Path) -> NDArray[np.float32]:
    grid: NDArray[np.float32] = np.load(path)
    assert grid.dtype == np.float32, "DEM grid is float32"
    assert grid.shape == (GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), "grid has national shape"
    return grid


def load_uint8_grid(path: Path) -> NDArray[np.uint8]:
    grid: NDArray[np.uint8] = np.load(path)
    assert grid.dtype == np.uint8, "grid is uint8"
    assert grid.shape == (GRID_HEIGHT_CELLS, GRID_WIDTH_CELLS), "grid has national shape"
    return grid
