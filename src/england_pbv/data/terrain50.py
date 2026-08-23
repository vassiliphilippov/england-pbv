"""Mosaic the OS Terrain 50 nested-zip archive into the national 50 m DEM grid.

Archive layout (verified 2026-08-23): outer zip holds data/<sq>/<tile>_OST50GRID_<date>.zip,
each containing <TILE>.asc — an ESRI ASCII grid, 200x200 cells, cellsize 50, EPSG:27700,
top row northernmost. Sea cells hold small negative values; offshore squares are absent.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from england_pbv.constants import GRID_CELL_M, SEA_LEVEL_M
from england_pbv.terrain.grid import new_dem_grid

TILE_CELLS: int = 200
ASC_HEADER_LINES: int = 5


@dataclass(frozen=True, slots=True)
class AscTile:
    xllcorner: float
    yllcorner: float
    values: NDArray[np.float32]  # (200, 200), top row northernmost


def parse_asc(text: str) -> AscTile:
    lines = text.splitlines()
    header: dict[str, float] = {}
    for line in lines[:ASC_HEADER_LINES]:
        key, raw_value = line.split()
        header[key.lower()] = float(raw_value)
    assert int(header["ncols"]) == TILE_CELLS, "OS Terrain 50 tiles are 200 columns"
    assert int(header["nrows"]) == TILE_CELLS, "OS Terrain 50 tiles are 200 rows"
    assert header["cellsize"] == GRID_CELL_M, "OS Terrain 50 cell size is 50 m"

    body = "\n".join(lines[ASC_HEADER_LINES:])
    values = np.fromstring(body, dtype=np.float32, sep=" ")  # noqa: NPY201
    assert values.size == TILE_CELLS * TILE_CELLS, "tile has 200x200 values"
    return AscTile(
        xllcorner=header["xllcorner"],
        yllcorner=header["yllcorner"],
        values=values.reshape(TILE_CELLS, TILE_CELLS),
    )


def build_dem_grid(terrain50_zip: Path) -> NDArray[np.float32]:
    grid = new_dem_grid()
    grid[:] = SEA_LEVEL_M
    tile_count = 0
    with zipfile.ZipFile(terrain50_zip) as outer:
        for entry in outer.namelist():
            if not entry.lower().endswith(".zip"):
                continue
            inner_bytes = outer.read(entry)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                asc_names = [n for n in inner.namelist() if n.lower().endswith(".asc")]
                for asc_name in asc_names:
                    tile = parse_asc(inner.read(asc_name).decode("ascii"))
                    col0 = int(tile.xllcorner // GRID_CELL_M)
                    row0 = int(tile.yllcorner // GRID_CELL_M)
                    # Grid rows grow northwards; .asc rows go north->south, so flip.
                    grid[row0 : row0 + TILE_CELLS, col0 : col0 + TILE_CELLS] = np.flipud(
                        tile.values
                    )
                    tile_count += 1
    assert tile_count > 2500, "national mosaic contains the expected tile count"
    print(f"mosaicked {tile_count} OS Terrain 50 tiles")
    return grid
