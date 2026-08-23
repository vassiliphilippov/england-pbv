"""Numba horizon-sweep viewshed engine.

For each observer, rays are cast at N_AZIMUTHS bearings; along each ray the terrain
elevation angle (with Earth-curvature/refraction correction) is tracked. A sample is
visible when its angle sets a new record. Record increments are exact angular
(visual-magnitude) areas of the panorama, which makes distance-band and land-cover
composition metrics cheap by-products of the sweep.

Azimuth convention: 0 = north (+northing), 90 deg = east (+easting), clockwise.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from england_pbv.constants import (
    DISTANCE_BAND_EDGES_M,
    EARTH_RADIUS_M,
    EYE_HEIGHT_M,
    GRID_CELL_M,
    MIN_SAMPLE_DISTANCE_M,
    N_AZIMUTHS,
    RAY_SEGMENTS,
    REFRACTION_K,
    TREE_OBSTRUCTION_NEAR_M,
)

N_LANDCOVER_BINS: int = 11  # WorldCover code // 10 -> 0..10
N_SECTORS: int = 12  # 30-degree sectors for directional drop
SECTOR_RING_MIN_M: float = 500.0
SECTOR_RING_MAX_M: float = 3000.0
TREE_BIN: int = 1  # WorldCover TREE_COVER (10) // 10

CURVATURE_PER_M: float = (1.0 - REFRACTION_K) / (2.0 * EARTH_RADIUS_M)


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    distances_m: NDArray[np.float32]
    steps_m: NDArray[np.float32]
    band_index: NDArray[np.uint8]
    near_sample_count: int  # samples with distance <= TREE_OBSTRUCTION_NEAR_M
    ring_start: int  # first sample index inside the sector-drop ring
    ring_stop: int  # one past the last sample index inside the sector-drop ring


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Raw per-ray and per-candidate accumulators for one batch of observers.

    Angular areas are in radian units per ray; multiply by the azimuth step (radians)
    to obtain steradians.
    """

    horizon_rad: NDArray[np.float32]  # (n, n_az) final horizon angle
    alpha0_rad: NDArray[np.float32]  # (n, n_az) angle to nearest sampled ground
    d_far_m: NDArray[np.float32]  # (n, n_az) farthest visible terrain distance
    tree_blocked: NDArray[np.uint8]  # (n, n_az) near ring dominated by tree cover
    plan_area: NDArray[np.float32]  # (n, n_bands) sum of r*dr for visible samples
    ang_area: NDArray[np.float32]  # (n, n_bands) sum of angle increments (rad)
    landcover_ang: NDArray[np.float32]  # (n, N_LANDCOVER_BINS) angle increments by class
    vis_z_min: NDArray[np.float32]  # (n,)
    vis_z_max: NDArray[np.float32]  # (n,)
    sector_mean_z: NDArray[np.float32]  # (n, N_SECTORS) mean terrain elev in 0.5-3 km ring


def build_sampling_plan() -> SamplingPlan:
    distances: list[float] = []
    steps: list[float] = []
    previous_limit = 0.0
    for limit, step in RAY_SEGMENTS:
        d = previous_limit
        while d + step <= limit + 1e-6:
            d += step
            if d >= MIN_SAMPLE_DISTANCE_M:
                distances.append(d)
                steps.append(step)
        previous_limit = limit

    distances_arr = np.array(distances, dtype=np.float32)
    steps_arr = np.array(steps, dtype=np.float32)
    edges = np.array(DISTANCE_BAND_EDGES_M, dtype=np.float64)
    band_index = (np.searchsorted(edges, distances_arr, side="right") - 1).astype(np.uint8)
    n_bands = len(DISTANCE_BAND_EDGES_M) - 1
    band_index = np.minimum(band_index, np.uint8(n_bands - 1))

    near_sample_count = int(np.sum(distances_arr <= TREE_OBSTRUCTION_NEAR_M))
    ring_start = int(np.searchsorted(distances_arr, SECTOR_RING_MIN_M))
    ring_stop = int(np.searchsorted(distances_arr, SECTOR_RING_MAX_M, side="right"))

    assert len(distances) > 0, "sampling plan is non-empty"
    assert near_sample_count > 0, "near ring contains samples"
    assert ring_stop > ring_start, "sector ring contains samples"
    return SamplingPlan(
        distances_m=distances_arr,
        steps_m=steps_arr,
        band_index=band_index,
        near_sample_count=near_sample_count,
        ring_start=ring_start,
        ring_stop=ring_stop,
    )


def azimuth_unit_vectors() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    azimuths = np.deg2rad(np.arange(N_AZIMUTHS, dtype=np.float64) * (360.0 / N_AZIMUTHS))
    return np.sin(azimuths), np.cos(azimuths)


@njit(inline="always", cache=True)
def _bilinear(dem: NDArray[np.float32], x_m: float, y_m: float) -> float:
    """Bilinear DEM sample at BNG metres; outside the grid returns sea level 0."""
    fcol = x_m / GRID_CELL_M - 0.5
    frow = y_m / GRID_CELL_M - 0.5
    height, width = dem.shape
    if fcol < 0.0 or frow < 0.0 or fcol >= width - 1.0 or frow >= height - 1.0:
        return 0.0
    c0 = int(fcol)
    r0 = int(frow)
    fx = fcol - c0
    fy = frow - r0
    z00 = dem[r0, c0]
    z01 = dem[r0, c0 + 1]
    z10 = dem[r0 + 1, c0]
    z11 = dem[r0 + 1, c0 + 1]
    return float(
        z00 * (1.0 - fx) * (1.0 - fy)
        + z01 * fx * (1.0 - fy)
        + z10 * (1.0 - fx) * fy
        + z11 * fx * fy
    )


@njit(inline="always", cache=True)
def _landcover_at(landcover: NDArray[np.uint8], x_m: float, y_m: float) -> int:
    col = int(x_m / GRID_CELL_M)
    row = int(y_m / GRID_CELL_M)
    height, width = landcover.shape
    if col < 0 or row < 0 or col >= width or row >= height:
        return 0
    bin_index = int(landcover[row, col]) // 10
    if bin_index >= N_LANDCOVER_BINS:
        bin_index = N_LANDCOVER_BINS - 1
    return bin_index


@njit(parallel=True, fastmath=True, cache=True)
def _sweep_kernel(  # noqa: PLR0913
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    obs_e: NDArray[np.float64],
    obs_n: NDArray[np.float64],
    obs_eye_z: NDArray[np.float64],
    sin_az: NDArray[np.float64],
    cos_az: NDArray[np.float64],
    distances: NDArray[np.float32],
    steps: NDArray[np.float32],
    band_index: NDArray[np.uint8],
    near_sample_count: int,
    ring_start: int,
    ring_stop: int,
    n_bands: int,
    horizon_rad: NDArray[np.float32],
    alpha0_rad: NDArray[np.float32],
    d_far_m: NDArray[np.float32],
    tree_blocked: NDArray[np.uint8],
    plan_area: NDArray[np.float32],
    ang_area: NDArray[np.float32],
    landcover_ang: NDArray[np.float32],
    vis_z_min: NDArray[np.float32],
    vis_z_max: NDArray[np.float32],
    sector_mean_z: NDArray[np.float32],
) -> None:
    n_obs = obs_e.shape[0]
    n_az = sin_az.shape[0]
    n_samples = distances.shape[0]

    # numba's prange has no typing; scoped ignore is unavoidable here
    for i in prange(n_obs):  # type: ignore[no-untyped-call, attr-defined]
        eye_z = obs_eye_z[i]
        e0 = obs_e[i]
        n0 = obs_n[i]
        z_min = 1.0e9
        z_max = -1.0e9
        sec_sum = np.zeros(N_SECTORS, dtype=np.float64)
        sec_cnt = np.zeros(N_SECTORS, dtype=np.int64)

        for j in range(n_az):
            dx = sin_az[j]
            dy = cos_az[j]
            sector = (j * N_SECTORS) // n_az
            max_ang = -10.0
            alpha0 = 0.0
            far = 0.0
            tree_near = 0

            for s in range(n_samples):
                r = float(distances[s])
                x = e0 + dx * r
                y = n0 + dy * r
                z = _bilinear(dem, x, y)
                if s < near_sample_count and _landcover_at(landcover, x, y) == TREE_BIN:
                    tree_near += 1
                if ring_start <= s < ring_stop:
                    sec_sum[sector] += z
                    sec_cnt[sector] += 1

                z_eff = z - CURVATURE_PER_M * r * r
                ang = math.atan((z_eff - eye_z) / r)
                if s == 0:
                    alpha0 = ang

                delta = ang - max_ang
                if delta > 0.0:
                    band = band_index[s]
                    plan_area[i, band] += r * steps[s]
                    if s > 0:
                        ang_area[i, band] += delta
                        landcover_ang[i, _landcover_at(landcover, x, y)] += delta
                    if z < z_min:
                        z_min = z
                    if z > z_max:
                        z_max = z
                    far = r
                    max_ang = ang

            horizon_rad[i, j] = max_ang
            alpha0_rad[i, j] = alpha0
            d_far_m[i, j] = far
            tree_blocked[i, j] = 1 if tree_near * 2 >= near_sample_count else 0

        vis_z_min[i] = z_min
        vis_z_max[i] = z_max
        for k in range(N_SECTORS):
            if sec_cnt[k] > 0:
                sector_mean_z[i, k] = sec_sum[k] / sec_cnt[k]
            else:
                sector_mean_z[i, k] = 0.0


def sample_dem_at(
    dem: NDArray[np.float32],
    eastings: NDArray[np.float64],
    northings: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = np.empty(len(eastings), dtype=np.float64)
    for index in range(len(eastings)):
        result[index] = _bilinear(dem, eastings[index], northings[index])
    return result


def sweep_batch(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    eastings: NDArray[np.float64],
    northings: NDArray[np.float64],
    plan: SamplingPlan,
) -> SweepResult:
    n_obs = len(eastings)
    assert n_obs == len(northings), "coordinate arrays have equal length"
    n_bands = len(DISTANCE_BAND_EDGES_M) - 1
    sin_az, cos_az = azimuth_unit_vectors()

    obs_ground = sample_dem_at(dem, eastings=eastings, northings=northings)
    obs_eye_z = obs_ground + EYE_HEIGHT_M

    result = SweepResult(
        horizon_rad=np.zeros((n_obs, N_AZIMUTHS), dtype=np.float32),
        alpha0_rad=np.zeros((n_obs, N_AZIMUTHS), dtype=np.float32),
        d_far_m=np.zeros((n_obs, N_AZIMUTHS), dtype=np.float32),
        tree_blocked=np.zeros((n_obs, N_AZIMUTHS), dtype=np.uint8),
        plan_area=np.zeros((n_obs, n_bands), dtype=np.float32),
        ang_area=np.zeros((n_obs, n_bands), dtype=np.float32),
        landcover_ang=np.zeros((n_obs, N_LANDCOVER_BINS), dtype=np.float32),
        vis_z_min=np.zeros(n_obs, dtype=np.float32),
        vis_z_max=np.zeros(n_obs, dtype=np.float32),
        sector_mean_z=np.zeros((n_obs, N_SECTORS), dtype=np.float32),
    )
    _sweep_kernel(
        dem,
        landcover,
        eastings,
        northings,
        obs_eye_z,
        sin_az,
        cos_az,
        plan.distances_m,
        plan.steps_m,
        plan.band_index,
        plan.near_sample_count,
        plan.ring_start,
        plan.ring_stop,
        n_bands,
        result.horizon_rad,
        result.alpha0_rad,
        result.d_far_m,
        result.tree_blocked,
        result.plan_area,
        result.ang_area,
        result.landcover_ang,
        result.vis_z_min,
        result.vis_z_max,
        result.sector_mean_z,
    )
    return result
