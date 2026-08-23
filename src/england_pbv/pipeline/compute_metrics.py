"""Run the horizon sweep over all candidates and write per-candidate view metrics.

Run: uv run python -m england_pbv.pipeline.compute_metrics
"""

import time

import numpy as np

from england_pbv import paths
from england_pbv.enums import LAND_COVER_LABELS, LandCoverClass
from england_pbv.models import CandidatePoint
from england_pbv.terrain.grid import load_dem_grid, load_uint8_grid
from england_pbv.viewshed.horizon import build_sampling_plan, sample_dem_at, sweep_batch
from england_pbv.viewshed.metrics import reduce_sweep

CHUNK_SIZE: int = 10000


def load_candidates() -> list[CandidatePoint]:
    candidates: list[CandidatePoint] = []
    with open(paths.CANDIDATES_JSONL, encoding="utf-8") as handle:
        for line in handle:
            candidates.append(CandidatePoint.model_validate_json(line))
    assert len(candidates) > 0, "candidates exist"
    return candidates


def _observer_landcover_label(code: int) -> str:
    try:
        return LAND_COVER_LABELS[LandCoverClass(code)]
    except ValueError:
        return LAND_COVER_LABELS[LandCoverClass.NODATA]


def main() -> None:
    dem = load_dem_grid(paths.DEM_GRID_NPY)
    landcover = load_uint8_grid(paths.LANDCOVER_GRID_NPY)
    candidates = load_candidates()
    plan = build_sampling_plan()
    print(f"{len(candidates)} candidates, {len(plan.distances_m)} samples/ray")

    started = time.time()
    with open(paths.METRICS_JSONL, "w", encoding="utf-8") as handle:
        for chunk_start in range(0, len(candidates), CHUNK_SIZE):
            chunk = candidates[chunk_start : chunk_start + CHUNK_SIZE]
            eastings = np.array([c.easting for c in chunk], dtype=np.float64)
            northings = np.array([c.northing for c in chunk], dtype=np.float64)
            sweep = sweep_batch(dem, landcover, eastings=eastings, northings=northings, plan=plan)
            ground = sample_dem_at(dem, eastings=eastings, northings=northings)
            observer_lc = [
                _observer_landcover_label(
                    int(landcover[int(c.northing // 50.0), int(c.easting // 50.0)])
                )
                for c in chunk
            ]
            metrics = reduce_sweep(
                candidate_ids=[c.candidate_id for c in chunk],
                observer_ground_z=ground,
                observer_landcover=observer_lc,
                sweep=sweep,
            )
            for row in metrics:
                handle.write(row.model_dump_json() + "\n")
            done = chunk_start + len(chunk)
            rate = done / (time.time() - started)
            print(f"{done}/{len(candidates)} ({rate:.0f}/s)", flush=True)
    print(f"wrote {paths.METRICS_JSONL} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
