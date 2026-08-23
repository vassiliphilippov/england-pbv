"""Engine validation on synthetic terrain with known geometry."""

import math

import numpy as np
from numpy.typing import NDArray

from england_pbv.constants import EARTH_RADIUS_M, GRID_CELL_M, REFRACTION_K
from england_pbv.enums import LandCoverClass
from england_pbv.models import ViewMetrics
from england_pbv.viewshed.horizon import build_sampling_plan, sweep_batch
from england_pbv.viewshed.metrics import reduce_sweep

GRID_SIDE_CELLS: int = 2400  # 120 km x 120 km synthetic world
CENTRE_M: float = GRID_SIDE_CELLS * GRID_CELL_M / 2.0


def _flat_world(elevation_m: float) -> NDArray[np.float32]:
    return np.full((GRID_SIDE_CELLS, GRID_SIDE_CELLS), elevation_m, dtype=np.float32)


def _grass_cover() -> NDArray[np.uint8]:
    return np.full(
        (GRID_SIDE_CELLS, GRID_SIDE_CELLS),
        int(LandCoverClass.GRASSLAND),
        dtype=np.uint8,
    )


def _run_single(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    easting: float,
    northing: float,
) -> ViewMetrics:
    plan = build_sampling_plan()
    sweep = sweep_batch(
        dem,
        landcover,
        eastings=np.array([easting], dtype=np.float64),
        northings=np.array([northing], dtype=np.float64),
        plan=plan,
    )
    ground = np.array([float(dem[int(northing // 50), int(easting // 50)])], dtype=np.float64)
    metrics = reduce_sweep(
        candidate_ids=["test"],
        observer_ground_z=ground,
        observer_landcover=["grassland"],
        sweep=sweep,
    )
    return metrics[0]


def test_flat_plain_horizon_matches_curvature_physics() -> None:
    dem = _flat_world(0.0)
    metrics = _run_single(dem, _grass_cover(), easting=CENTRE_M, northing=CENTRE_M)

    # Eye 1.7 m above a flat plain: geometric horizon at sqrt(2 * R_eff * h) ~ 5.0 km.
    effective_radius = EARTH_RADIUS_M / (1.0 - REFRACTION_K)
    expected_horizon_km = math.sqrt(2.0 * effective_radius * 1.7) / 1000.0
    assert abs(metrics.d_far_median_km - expected_horizon_km) < 1.0, (
        f"flat-plain horizon ~{expected_horizon_km:.1f} km, got {metrics.d_far_median_km:.1f}"
    )
    assert metrics.open_fraction == 1.0, "flat plain is open in every direction"
    assert metrics.far_fraction == 0.0, "no terrain visible beyond 10 km on a flat plain"
    assert metrics.mean_horizon_deg < 0.1, "flat-plain horizon angle is about zero"
    assert metrics.max_sector_drop_m < 1.0, "flat plain has no drop"


def test_escarpment_edge_sees_far_on_one_side() -> None:
    dem = _flat_world(0.0)
    plateau_edge_col = int(CENTRE_M / GRID_CELL_M)
    dem[:, :plateau_edge_col] = 200.0  # plateau to the west, vale at sea level to the east
    # Observer on the last plateau cell centre, standing right at the edge.
    observer_e = (plateau_edge_col - 1) * GRID_CELL_M + 25.0
    metrics = _run_single(dem, _grass_cover(), easting=observer_e, northing=CENTRE_M)

    assert metrics.far_fraction > 0.35, "the vale side sees far"
    assert metrics.far_fraction < 0.65, "the plateau side does not see far"
    assert 140.0 < metrics.longest_far_arc_deg < 230.0, "one continuous open sector of ~180 deg"
    assert metrics.max_sector_drop_m > 150.0, "directional drop into the vale is detected"
    assert metrics.mean_depression_deg > 3.0, "the vale side looks steeply down at the ground"
    assert metrics.d_far_max_km > 30.0, "long sightlines across the vale"


def test_enclosing_ridge_blocks_the_view() -> None:
    dem = _flat_world(0.0)
    centre_cell = int(CENTRE_M / GRID_CELL_M)
    # A 60 m high square ring of terrain at ~500 m from the observer.
    inner = centre_cell - 10
    outer = centre_cell + 10
    dem[inner - 2 : outer + 2, inner - 2 : outer + 2] = 60.0
    dem[inner : outer + 1, inner : outer + 1] = 0.0
    metrics = _run_single(dem, _grass_cover(), easting=CENTRE_M, northing=CENTRE_M)

    assert metrics.far_fraction == 0.0, "ring blocks all distant terrain"
    assert metrics.mean_horizon_deg > 2.0, "horizon is pushed up by the enclosing ridge"
    assert metrics.open_fraction < 0.2, "few open bearings inside a ring"


def test_landcover_composition_of_visible_scene() -> None:
    dem = _flat_world(0.0)
    hill_cells = 40
    centre_cell = int(CENTRE_M / GRID_CELL_M)
    # A gentle 100 m cone under the observer so the plain stays visible to 40 km.
    rows, cols = np.mgrid[0:GRID_SIDE_CELLS, 0:GRID_SIDE_CELLS]
    dist_cells = np.sqrt((rows - centre_cell) ** 2 + (cols - centre_cell) ** 2)
    cone = np.maximum(0.0, 100.0 * (1.0 - dist_cells / hill_cells)).astype(np.float32)
    dem = dem + cone

    landcover = np.full(
        (GRID_SIDE_CELLS, GRID_SIDE_CELLS), int(LandCoverClass.WATER), dtype=np.uint8
    )
    metrics = _run_single(dem, landcover, easting=CENTRE_M, northing=CENTRE_M)

    assert metrics.water_fraction > 0.9, "a water world panorama is nearly all water"
    assert metrics.shannon_diversity < 0.1, "single-class panorama has near-zero diversity"
    assert metrics.far_fraction == 1.0, "100 m summit sees beyond 10 km everywhere"
    assert metrics.mean_depression_deg > 1.0, "summit looks down at nearby ground"


def test_near_tree_ring_flags_obstruction() -> None:
    dem = _flat_world(0.0)
    landcover = _grass_cover()
    centre_cell = int(CENTRE_M / GRID_CELL_M)
    ring = 8  # 400 m box of woodland around the observer
    landcover[
        centre_cell - ring : centre_cell + ring + 1,
        centre_cell - ring : centre_cell + ring + 1,
    ] = int(LandCoverClass.TREE_COVER)
    metrics = _run_single(dem, landcover, easting=CENTRE_M, northing=CENTRE_M)

    assert metrics.near_tree_fraction > 0.9, "woodland ring flags nearly every bearing"


def test_vegetation_blocks_summit_view_but_not_scarp_slope() -> None:
    # A 100 m cone summit: with a woodland ring around the summit, the leaf-on view dies.
    centre_cell = int(CENTRE_M / GRID_CELL_M)
    rows, cols = np.mgrid[0:GRID_SIDE_CELLS, 0:GRID_SIDE_CELLS]
    dist_cells = np.sqrt((rows - centre_cell) ** 2 + (cols - centre_cell) ** 2)
    dem = np.maximum(0.0, 100.0 * (1.0 - dist_cells / 40)).astype(np.float32)

    landcover = _grass_cover()
    ring = 8
    landcover[
        centre_cell - ring : centre_cell + ring + 1,
        centre_cell - ring : centre_cell + ring + 1,
    ] = int(LandCoverClass.TREE_COVER)
    blocked = _run_single(dem, landcover, easting=CENTRE_M, northing=CENTRE_M)
    assert blocked.far_fraction == 1.0, "bare-earth view reaches far in every direction"
    assert blocked.far_fraction_veg < 0.1, "summit woodland ring kills the leaf-on view"
    assert blocked.veg_retention < 0.3, "little visible area survives the woodland ring"

    # Same woodland placed on a steep scarp slope BELOW an escarpment-edge observer:
    # tree tops stay below the sightline, so the leaf-on view survives.
    dem2 = _flat_world(0.0)
    edge_col = centre_cell
    dem2[:, :edge_col] = 200.0
    landcover2 = _grass_cover()
    landcover2[:, edge_col + 2 : edge_col + 8] = int(LandCoverClass.TREE_COVER)  # 100-400 m out
    observer_e = (edge_col - 1) * GRID_CELL_M + 25.0
    open_view = _run_single(dem2, landcover2, easting=observer_e, northing=CENTRE_M)
    assert open_view.far_fraction_veg > 0.3, "trees below the escarpment lip do not block"
