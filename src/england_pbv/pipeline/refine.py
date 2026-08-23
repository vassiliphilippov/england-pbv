"""Fine local search: find the best exact standing spot around each promising candidate.

Two grid passes per candidate (coarse +/-200 m at 50 m, then fine +/-50 m at 12.5 m around
the coarse winner), scored against the frozen national percentile transforms so results are
comparable everywhere. Only positions within MAX_TOTAL_OFFSET_M of the original seed are
considered, so every written coordinate was actually evaluated and stays matchable.

Run EXACTLY ONCE after screening (rerunning would measure the offset budget from
already-moved positions); pipeline.screening regenerates the seeds. Rerun
compute_metrics + scoring afterwards.

Run: uv run python -m england_pbv.pipeline.refine
"""

import time

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from england_pbv import paths
from england_pbv.constants import (
    REFINE_COARSE_SPAN_M,
    REFINE_COARSE_STEP_M,
    REFINE_DEDUPE_RADIUS_M,
    REFINE_FINE_SPAN_M,
    REFINE_FINE_STEP_M,
    REFINE_TOP_N,
)
from england_pbv.models import CandidatePoint, ScoredViewpoint
from england_pbv.pipeline.score_inputs import FrozenPercentiles, metric_inputs
from england_pbv.terrain.grid import bng_to_latlon, load_dem_grid, load_uint8_grid
from england_pbv.verification.evaluate import load_scored
from england_pbv.viewshed.horizon import build_sampling_plan, sample_dem_at, sweep_batch
from england_pbv.viewshed.metrics import reduce_sweep

CHUNK_CANDIDATES: int = 120
MAX_TOTAL_OFFSET_M: float = 280.0  # keep refined spots matchable to their origin


def _offset_grid(span_m: float, step_m: float) -> NDArray[np.float64]:
    steps = np.arange(-span_m, span_m + step_m / 2, step_m, dtype=np.float64)
    de, dn = np.meshgrid(steps, steps)
    return np.column_stack([de.ravel(), dn.ravel()])


def _dedupe_by_rank(scored: list[ScoredViewpoint]) -> list[ScoredViewpoint]:
    """Keep the best-ranked candidate within each 250 m neighbourhood."""
    ordered = sorted(scored, key=lambda s: s.national_rank)
    kept: list[ScoredViewpoint] = []
    kept_coords: list[tuple[float, float]] = []
    tree: cKDTree | None = None
    tail_start = 0
    rebuild_every = 500
    radius_sq = REFINE_DEDUPE_RADIUS_M**2
    for item in ordered:
        point = (item.candidate.easting, item.candidate.northing)
        if tree is not None and len(tree.query_ball_point(point, r=REFINE_DEDUPE_RADIUS_M)) > 0:
            continue
        if any(
            (point[0] - e) ** 2 + (point[1] - n) ** 2 < radius_sq
            for e, n in kept_coords[tail_start:]
        ):
            continue
        kept.append(item)
        kept_coords.append(point)
        if len(kept_coords) - tail_start >= rebuild_every:
            tree = cKDTree(np.array(kept_coords))
            tail_start = len(kept_coords)
    return kept


def _best_offsets(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    frozen: FrozenPercentiles,
    centres_e: NDArray[np.float64],
    centres_n: NDArray[np.float64],
    offsets: NDArray[np.float64],
    budget_origin_e: NDArray[np.float64],
    budget_origin_n: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate every centre x offset position; return the best in-budget position.

    Positions farther than MAX_TOTAL_OFFSET_M from the budget origin are excluded from
    the argmax, so every returned coordinate was actually evaluated and every candidate
    stays within matching distance of its original seed. Offset (0,0) is always in
    budget, so the result never degrades below the centre position.
    """
    plan = build_sampling_plan()
    n_origins = len(centres_e)
    n_offsets = len(offsets)
    all_e = (centres_e[:, None] + offsets[None, :, 0]).ravel()
    all_n = (centres_n[:, None] + offsets[None, :, 1]).ravel()

    sweep = sweep_batch(dem, landcover, eastings=all_e, northings=all_n, plan=plan)
    ground = sample_dem_at(dem, eastings=all_e, northings=all_n)
    ids = [f"r{i}" for i in range(len(all_e))]
    lc_labels = [""] * len(all_e)
    metrics = reduce_sweep(
        candidate_ids=ids, observer_ground_z=ground, observer_landcover=lc_labels, sweep=sweep
    )
    vectors = np.array([metric_inputs(m) for m in metrics], dtype=np.float64)
    scores = frozen.composite(vectors).reshape(n_origins, n_offsets)

    drift = np.sqrt(
        (all_e.reshape(n_origins, n_offsets) - budget_origin_e[:, None]) ** 2
        + (all_n.reshape(n_origins, n_offsets) - budget_origin_n[:, None]) ** 2
    )
    scores = np.where(drift <= MAX_TOTAL_OFFSET_M, scores, -np.inf)
    assert bool(np.all(np.isfinite(np.max(scores, axis=1)))), "every origin keeps a valid position"

    best = np.argmax(scores, axis=1)
    best_e = centres_e + offsets[best, 0]
    best_n = centres_n + offsets[best, 1]
    best_scores = scores[np.arange(n_origins), best]
    return best_e, best_n, best_scores


def main() -> None:
    dem = load_dem_grid(paths.DEM_GRID_NPY)
    landcover = load_uint8_grid(paths.LANDCOVER_GRID_NPY)
    scored = load_scored()

    input_matrix = np.array([metric_inputs(s.metrics) for s in scored], dtype=np.float64)
    frozen = FrozenPercentiles(input_matrix)

    deduped = _dedupe_by_rank(scored)
    selected = deduped[:REFINE_TOP_N]
    selected_ids = {s.candidate.candidate_id for s in selected}
    print(f"{len(scored)} scored -> {len(deduped)} after dedupe -> refining {len(selected)}")

    coarse = _offset_grid(REFINE_COARSE_SPAN_M, REFINE_COARSE_STEP_M)
    fine = _offset_grid(REFINE_FINE_SPAN_M, REFINE_FINE_STEP_M)

    refined_positions: dict[str, tuple[float, float]] = {}
    moved_count = 0
    improvements: list[float] = []
    started = time.time()
    for chunk_start in range(0, len(selected), CHUNK_CANDIDATES):
        chunk = selected[chunk_start : chunk_start + CHUNK_CANDIDATES]
        origins_e = np.array([s.candidate.easting for s in chunk], dtype=np.float64)
        origins_n = np.array([s.candidate.northing for s in chunk], dtype=np.float64)

        coarse_e, coarse_n, _ = _best_offsets(
            dem,
            landcover,
            frozen,
            centres_e=origins_e,
            centres_n=origins_n,
            offsets=coarse,
            budget_origin_e=origins_e,
            budget_origin_n=origins_n,
        )
        fine_e, fine_n, fine_scores = _best_offsets(
            dem,
            landcover,
            frozen,
            centres_e=coarse_e,
            centres_n=coarse_n,
            offsets=fine,
            budget_origin_e=origins_e,
            budget_origin_n=origins_n,
        )

        drift = np.sqrt((fine_e - origins_e) ** 2 + (fine_n - origins_n) ** 2)
        for index, item in enumerate(chunk):
            refined_positions[item.candidate.candidate_id] = (
                float(fine_e[index]),
                float(fine_n[index]),
            )
            if drift[index] > 1.0:
                moved_count += 1
            improvements.append(float(fine_scores[index]) - item.view_potential)

        done = chunk_start + len(chunk)
        rate = done / (time.time() - started)
        print(
            f"refined {done}/{len(selected)} ({rate:.0f}/s, moved {moved_count})",
            flush=True,
        )

    print(
        f"moved {moved_count}/{len(selected)}; mean composite gain "
        f"{float(np.mean(improvements)):.2f} points"
    )

    updated: list[CandidatePoint] = []
    with open(paths.CANDIDATES_JSONL, encoding="utf-8") as handle:
        for line in handle:
            candidate = CandidatePoint.model_validate_json(line)
            position = refined_positions.get(candidate.candidate_id)
            if position is not None:
                latlon = bng_to_latlon(position[0], position[1])
                ground = sample_dem_at(
                    dem,
                    eastings=np.array([position[0]]),
                    northings=np.array([position[1]]),
                )
                candidate = candidate.model_copy(
                    update={
                        "easting": position[0],
                        "northing": position[1],
                        "lat": latlon.lat,
                        "lon": latlon.lon,
                        "elevation_m": float(ground[0]),
                    }
                )
            updated.append(candidate)

    kept_ids = {c.candidate_id for c in updated}
    assert selected_ids <= kept_ids, "every refined candidate is still present"
    with open(paths.CANDIDATES_JSONL, "w", encoding="utf-8") as handle:
        for candidate in updated:
            handle.write(candidate.model_dump_json(exclude_none=True) + "\n")
    print(f"updated {paths.CANDIDATES_JSONL}; rerun compute_metrics and scoring")


if __name__ == "__main__":
    main()
