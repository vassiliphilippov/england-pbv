"""Candidate generation: multi-scale TPI screening + named-point seeding.

Run: uv run python -m england_pbv.pipeline.screening
"""

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter

from england_pbv import paths
from england_pbv.constants import (
    GRID_CELL_M,
    NMS_RADIUS_M,
    SCREENING_KEEP_FRACTION,
    TPI_RADII_M,
)
from england_pbv.data.pois import load_dobih_hills, load_osm_points
from england_pbv.enums import CandidateSource
from england_pbv.models import CandidatePoint, VerificationViewpoint
from england_pbv.terrain.derivatives import tpi
from england_pbv.terrain.grid import (
    bng_to_latlon,
    cell_to_bng,
    in_grid,
    latlon_to_bng,
    load_dem_grid,
    load_uint8_grid,
)

DOBIH_MIN_DROP_M: float = 30.0


@dataclass(frozen=True, slots=True)
class TpiSample:
    tpi_500m: float
    tpi_2km: float
    tpi_10km: float


def _screening_score(
    dem: NDArray[np.float32],
    england: NDArray[np.uint8],
) -> tuple[NDArray[np.float32], dict[float, NDArray[np.float32]]]:
    """Max of per-scale standardized positive TPI over England cells."""
    england_bool = england.astype(bool)
    score = np.full(dem.shape, -1.0, dtype=np.float32)
    tpi_grids: dict[float, NDArray[np.float32]] = {}
    for radius in TPI_RADII_M:
        tpi_grid = tpi(dem, radius_m=radius)
        tpi_grids[radius] = tpi_grid
        std = float(np.std(tpi_grid[england_bool]))
        assert std > 0.0, "TPI has spread over England"
        np.maximum(score, tpi_grid / std, out=score)
        print(f"TPI radius {radius:.0f} m: std {std:.2f} m")
    score[~england_bool] = -1.0
    return score, tpi_grids


def _select_candidates(score: NDArray[np.float32]) -> NDArray[np.bool_]:
    valid = score > 0.0
    threshold = float(np.quantile(score[valid], 1.0 - SCREENING_KEEP_FRACTION))
    kept = score >= threshold
    print(f"screening threshold {threshold:.2f}; kept cells before NMS: {int(kept.sum())}")

    nms_cells = 2 * int(round(NMS_RADIUS_M / GRID_CELL_M)) + 1
    masked = np.where(kept, score, np.float32(-1.0e9))
    local_max = maximum_filter(masked, size=nms_cells, mode="nearest")
    winners: NDArray[np.bool_] = kept & (masked >= local_max)
    print(f"candidates after {NMS_RADIUS_M:.0f} m NMS: {int(winners.sum())}")
    return winners


def main() -> None:
    dem = load_dem_grid(paths.DEM_GRID_NPY)
    england = load_uint8_grid(paths.ENGLAND_MASK_NPY)
    score, tpi_grids = _screening_score(dem, england=england)
    winners = _select_candidates(score)

    candidates: dict[tuple[int, int], CandidatePoint] = {}

    def sample_tpi(row: int, col: int) -> TpiSample:
        return TpiSample(
            tpi_500m=float(tpi_grids[500.0][row, col]),
            tpi_2km=float(tpi_grids[2000.0][row, col]),
            tpi_10km=float(tpi_grids[10000.0][row, col]),
        )

    rows, cols = np.nonzero(winners)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        point = cell_to_bng(row, col)
        latlon = bng_to_latlon(point.easting, point.northing)
        tpi_sample = sample_tpi(row, col)
        candidates[(row, col)] = CandidatePoint(
            candidate_id=f"c{row}_{col}",
            easting=point.easting,
            northing=point.northing,
            lat=latlon.lat,
            lon=latlon.lon,
            elevation_m=float(dem[row, col]),
            source=CandidateSource.SCREENING,
            tpi_500m=tpi_sample.tpi_500m,
            tpi_2km=tpi_sample.tpi_2km,
            tpi_10km=tpi_sample.tpi_10km,
        )

    def add_named(
        source: CandidateSource,
        source_id: str,
        name: str | None,
        lat: float,
        lon: float,
        require_england: bool,
    ) -> None:
        bng = latlon_to_bng(lat, lon)
        pos_row = int(bng.northing // GRID_CELL_M)
        pos_col = int(bng.easting // GRID_CELL_M)
        if not in_grid(pos_row, pos_col):
            return
        if require_england and england[pos_row, pos_col] == 0:
            return
        key = (pos_row, pos_col)
        existing = candidates.get(key)
        if existing is not None:
            if existing.name is None and name is not None:
                candidates[key] = existing.model_copy(update={"name": name, "source": source})
            return
        tpi_sample = sample_tpi(pos_row, pos_col)
        candidates[key] = CandidatePoint(
            candidate_id=source_id,
            easting=bng.easting,
            northing=bng.northing,
            lat=lat,
            lon=lon,
            elevation_m=float(dem[pos_row, pos_col]),
            source=source,
            name=name,
            tpi_500m=tpi_sample.tpi_500m,
            tpi_2km=tpi_sample.tpi_2km,
            tpi_10km=tpi_sample.tpi_10km,
        )

    for osm_point in load_osm_points(paths.OSM_VIEWPOINTS_JSON, source_prefix="osmv"):
        add_named(
            CandidateSource.OSM_VIEWPOINT,
            source_id=osm_point.source_id,
            name=osm_point.name,
            lat=osm_point.lat,
            lon=osm_point.lon,
            require_england=True,
        )
    for osm_point in load_osm_points(paths.OSM_PEAKS_JSON, source_prefix="osmp"):
        add_named(
            CandidateSource.OSM_PEAK,
            source_id=osm_point.source_id,
            name=osm_point.name,
            lat=osm_point.lat,
            lon=osm_point.lon,
            require_england=True,
        )
    for hill in load_dobih_hills(min_drop_m=DOBIH_MIN_DROP_M):
        add_named(
            CandidateSource.DOBIH_HILL,
            source_id=hill.source_id,
            name=hill.name,
            lat=hill.lat,
            lon=hill.lon,
            require_england=True,
        )

    verification = json.loads(paths.VERIFICATION_VIEWPOINTS_JSON.read_text(encoding="utf-8"))
    for index, raw in enumerate(verification["viewpoints"]):
        viewpoint = VerificationViewpoint.model_validate(raw)
        add_named(
            CandidateSource.VERIFICATION,
            source_id=f"ver{index}",
            name=viewpoint.name,
            lat=viewpoint.lat,
            lon=viewpoint.lon,
            require_england=False,
        )

    paths.OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(paths.CANDIDATES_JSONL, "w", encoding="utf-8") as handle:
        for candidate in candidates.values():
            handle.write(candidate.model_dump_json(exclude_none=True) + "\n")
    by_source: dict[str, int] = {}
    for candidate in candidates.values():
        by_source[candidate.source.value] = by_source.get(candidate.source.value, 0) + 1
    print(f"wrote {len(candidates)} candidates to {paths.CANDIDATES_JSONL}: {by_source}")


if __name__ == "__main__":
    main()
