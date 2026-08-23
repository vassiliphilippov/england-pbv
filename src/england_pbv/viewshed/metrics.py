"""Reduce raw sweep accumulators to the per-candidate ViewMetrics artifact."""

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from england_pbv.constants import (
    DISTANCE_BAND_EDGES_M,
    FAR_VIEW_DISTANCE_M,
    N_AZIMUTHS,
    OPEN_HORIZON_DEG,
)
from england_pbv.enums import LAND_COVER_LABELS, LandCoverClass
from england_pbv.models import CandidateId, ViewMetrics
from england_pbv.viewshed.horizon import N_LANDCOVER_BINS, SweepResult

AZIMUTH_STEP_RAD: float = 2.0 * math.pi / N_AZIMUTHS
STERADIAN_TO_DEG2: float = (180.0 / math.pi) ** 2
SKYLINE_SMOOTH_WINDOW: int = 5  # 2.5 degrees at 720 azimuths
SHANNON_CLASS_COUNT: int = 10  # fixed normalisation base across all candidates

_BIN_TO_CLASS: list[LandCoverClass] = [
    LandCoverClass.NODATA,
    LandCoverClass.TREE_COVER,
    LandCoverClass.SHRUBLAND,
    LandCoverClass.GRASSLAND,
    LandCoverClass.CROPLAND,
    LandCoverClass.BUILT_UP,
    LandCoverClass.BARE_SPARSE,
    LandCoverClass.SNOW_ICE,
    LandCoverClass.WATER,
    LandCoverClass.WETLAND,
    LandCoverClass.MOSS_LICHEN,
]


@njit(cache=True)
def _longest_true_arcs(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Longest circular run of True per row, in fractions of the full circle."""
    n_rows, n_az = mask.shape
    result = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        total = 0
        for j in range(n_az):
            if mask[i, j]:
                total += 1
        if total == n_az:
            result[i] = 1.0
            continue
        best = 0
        run = 0
        for j in range(2 * n_az):
            if mask[i, j % n_az]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        if best > n_az:
            best = n_az
        result[i] = best / n_az
    return result


def _circular_smooth(values: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    assert window % 2 == 1, "smoothing window is odd"
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.concatenate([values[:, -(window // 2) :], values, values[:, : window // 2]], axis=1)
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return smoothed.astype(np.float32)


def reduce_sweep(
    candidate_ids: list[CandidateId],
    observer_ground_z: NDArray[np.float64],
    observer_landcover: list[str],
    sweep: SweepResult,
) -> list[ViewMetrics]:
    n = len(candidate_ids)
    assert sweep.horizon_rad.shape[0] == n, "sweep batch matches candidate list"
    assert len(observer_landcover) == n, "observer land-cover labels match candidate list"

    horizon_deg = np.rad2deg(sweep.horizon_rad)
    alpha0_deg = np.rad2deg(sweep.alpha0_rad)

    plan_area_km2 = sweep.plan_area * AZIMUTH_STEP_RAD / 1.0e6
    ang_area_deg2 = sweep.ang_area * AZIMUTH_STEP_RAD * STERADIAN_TO_DEG2
    lc_ang = sweep.landcover_ang.astype(np.float64)

    mean_h = np.mean(horizon_deg, axis=1)
    median_h = np.median(horizon_deg, axis=1)
    p90_h = np.percentile(horizon_deg, 90, axis=1)
    open_fraction = np.mean(horizon_deg < OPEN_HORIZON_DEG, axis=1)

    far_mask = sweep.d_far_m > FAR_VIEW_DISTANCE_M
    far_fraction = np.mean(far_mask, axis=1)
    longest_arc_deg = _longest_true_arcs(far_mask) * 360.0

    far_mask_veg = sweep.d_far_veg_m > FAR_VIEW_DISTANCE_M
    far_fraction_veg = np.mean(far_mask_veg, axis=1)
    longest_arc_veg_deg = _longest_true_arcs(far_mask_veg) * 360.0
    d_far_veg_p90 = np.percentile(sweep.d_far_veg_m / 1000.0, 90, axis=1)
    plan_area_veg_km2 = sweep.plan_area_veg * AZIMUTH_STEP_RAD / 1.0e6
    total_area = np.sum(plan_area_km2, axis=1)
    total_area_veg = np.sum(plan_area_veg_km2, axis=1)
    veg_retention = np.where(total_area > 0.0, total_area_veg / np.maximum(total_area, 1e-9), 1.0)

    d_far_km = sweep.d_far_m / 1000.0
    d_far_median = np.median(d_far_km, axis=1)
    d_far_p90 = np.percentile(d_far_km, 90, axis=1)
    d_far_max = np.max(d_far_km, axis=1)

    smoothed = _circular_smooth(horizon_deg, window=SKYLINE_SMOOTH_WINDOW)
    diffs = np.abs(np.diff(np.concatenate([smoothed, smoothed[:, :1]], axis=1), axis=1))
    skyline_tv = np.sum(diffs, axis=1)

    depression = np.mean(np.maximum(0.0, -alpha0_deg), axis=1)
    near_tree_fraction = np.mean(sweep.tree_blocked, axis=1)

    lc_totals = np.sum(lc_ang, axis=1)
    visible_relief = np.maximum(0.0, sweep.vis_z_max - sweep.vis_z_min)
    max_sector_drop = observer_ground_z - np.min(sweep.sector_mean_z, axis=1)

    n_bands = len(DISTANCE_BAND_EDGES_M) - 1
    metrics: list[ViewMetrics] = []
    for i in range(n):
        total_lc = float(lc_totals[i])
        fractions: dict[str, float] = {}
        shannon = 0.0
        if total_lc > 0.0:
            probs = lc_ang[i] / total_lc
            for bin_index in range(N_LANDCOVER_BINS):
                p = float(probs[bin_index])
                if p > 0.0:
                    label = LAND_COVER_LABELS[_BIN_TO_CLASS[bin_index]]
                    fractions[label] = fractions.get(label, 0.0) + round(p, 5)
                    if bin_index != 0:
                        shannon -= p * math.log(p)
            shannon /= math.log(SHANNON_CLASS_COUNT)

        tree_f = float(lc_ang[i][1] / total_lc) if total_lc > 0.0 else 0.0
        built_f = float(lc_ang[i][5] / total_lc) if total_lc > 0.0 else 0.0
        water_f = float(lc_ang[i][8] / total_lc) if total_lc > 0.0 else 0.0

        metrics.append(
            ViewMetrics(
                candidate_id=candidate_ids[i],
                visible_area_km2_by_band=[float(plan_area_km2[i, b]) for b in range(n_bands)],
                total_visible_area_km2=float(total_area[i]),
                angular_area_deg2_by_band=[float(ang_area_deg2[i, b]) for b in range(n_bands)],
                total_terrain_angular_deg2=float(np.sum(ang_area_deg2[i])),
                mean_horizon_deg=float(mean_h[i]),
                median_horizon_deg=float(median_h[i]),
                p90_horizon_deg=float(p90_h[i]),
                skyline_total_variation_deg=float(skyline_tv[i]),
                open_fraction=float(open_fraction[i]),
                far_fraction=float(far_fraction[i]),
                longest_far_arc_deg=float(longest_arc_deg[i]),
                d_far_median_km=float(d_far_median[i]),
                d_far_p90_km=float(d_far_p90[i]),
                d_far_max_km=float(d_far_max[i]),
                far_fraction_veg=float(far_fraction_veg[i]),
                longest_far_arc_veg_deg=float(longest_arc_veg_deg[i]),
                d_far_veg_p90_km=float(d_far_veg_p90[i]),
                visible_area_veg_km2=float(total_area_veg[i]),
                veg_retention=float(min(1.0, veg_retention[i])),
                mean_depression_deg=float(depression[i]),
                max_sector_drop_m=float(max_sector_drop[i]),
                visible_relief_m=float(visible_relief[i]),
                landcover_angular_fractions=fractions,
                shannon_diversity=float(shannon),
                water_fraction=water_f,
                built_fraction=built_f,
                tree_fraction=tree_f,
                near_tree_fraction=float(near_tree_fraction[i]),
                observer_landcover=observer_landcover[i],
            )
        )
    return metrics
