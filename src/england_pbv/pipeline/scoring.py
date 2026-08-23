"""Percentile-normalised component scores and the composite view-potential ranking.

Every component is a national percentile (0..100) among all candidates; the composite is
their mean — a documented convention, never a claim of measured beauty.

Run: uv run python -m england_pbv.pipeline.scoring
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from england_pbv import paths
from england_pbv.constants import REGIONAL_PERCENTILE_RADIUS_M
from england_pbv.data.pois import load_osm_points
from england_pbv.models import CandidatePoint, ComponentScores, ScoredViewpoint, ViewMetrics
from england_pbv.pipeline.compute_metrics import load_candidates
from england_pbv.pipeline.score_inputs import FrozenPercentiles, metric_inputs
from england_pbv.terrain.grid import latlon_to_bng

NEAR_PLACE_MAX_M: float = 6000.0


def load_metrics() -> dict[str, ViewMetrics]:
    metrics: dict[str, ViewMetrics] = {}
    with open(paths.METRICS_JSONL, encoding="utf-8") as handle:
        for line in handle:
            row = ViewMetrics.model_validate_json(line)
            metrics[row.candidate_id] = row
    return metrics


def regional_percentiles(
    eastings: NDArray[np.float64],
    northings: NDArray[np.float64],
    potential: NDArray[np.float64],
) -> NDArray[np.float64]:
    tree = cKDTree(np.column_stack([eastings, northings]))
    result = np.zeros(len(potential), dtype=np.float64)
    neighbour_lists = tree.query_ball_tree(tree, r=REGIONAL_PERCENTILE_RADIUS_M)
    for index, neighbours in enumerate(neighbour_lists):
        local = potential[neighbours]
        below = float(np.sum(local < potential[index]))
        result[index] = below / max(1, len(local) - 1) * 100.0
    return result


def nearest_place_names(
    eastings: NDArray[np.float64],
    northings: NDArray[np.float64],
) -> list[str | None]:
    places = load_osm_points(paths.OSM_PLACES_JSON, source_prefix="place")
    named = [p for p in places if p.name is not None]
    coords = np.array(
        [[latlon_to_bng(p.lat, p.lon).easting, latlon_to_bng(p.lat, p.lon).northing] for p in named]
    )
    tree = cKDTree(coords)
    distances, indices = tree.query(np.column_stack([eastings, northings]))
    results: list[str | None] = []
    for distance, place_index in zip(distances.tolist(), indices.tolist(), strict=True):
        if distance <= NEAR_PLACE_MAX_M:
            results.append(named[place_index].name)
        else:
            results.append(None)
    return results


def main() -> None:
    candidates = load_candidates()
    metrics_by_id = load_metrics()
    joined: list[tuple[CandidatePoint, ViewMetrics]] = []
    for candidate in candidates:
        metric = metrics_by_id.get(candidate.candidate_id)
        if metric is not None:
            joined.append((candidate, metric))
    print(f"scoring {len(joined)} candidates")

    input_matrix = np.array([metric_inputs(m) for _, m in joined], dtype=np.float64)
    frozen = FrozenPercentiles(input_matrix)
    potential = frozen.composite(input_matrix)

    eastings = np.array([c.easting for c, _ in joined], dtype=np.float64)
    northings = np.array([c.northing for c, _ in joined], dtype=np.float64)
    regional = regional_percentiles(eastings, northings, potential=potential)
    places = nearest_place_names(eastings, northings)

    order = np.argsort(-potential)
    ranks = np.empty(len(joined), dtype=np.int64)
    ranks[order] = np.arange(1, len(joined) + 1)

    with open(paths.SCORED_JSONL, "w", encoding="utf-8") as handle:
        for index, (candidate, metric) in enumerate(joined):
            place = places[index]
            if candidate.name is not None:
                display = candidate.name
            elif place is not None:
                display = f"Viewpoint near {place}"
            else:
                display = f"Viewpoint {candidate.candidate_id}"
            scored_components = frozen.score_one(metric_inputs(metric))
            scored = ScoredViewpoint(
                candidate=candidate,
                metrics=metric,
                components=ComponentScores(
                    prospect=round(scored_components.components["prospect"], 2),
                    openness=round(scored_components.components["openness"], 2),
                    drop=round(scored_components.components["drop"], 2),
                    depth=round(scored_components.components["depth"], 2),
                    diversity=round(scored_components.components["diversity"], 2),
                    clearness=round(scored_components.components["clearness"], 2),
                ),
                view_potential=round(float(potential[index]), 2),
                national_rank=int(ranks[index]),
                regional_percentile=round(float(regional[index]), 2),
                display_name=display,
                region_hint=place,
            )
            handle.write(scored.model_dump_json(exclude_none=True) + "\n")
    print(f"wrote {paths.SCORED_JSONL}")

    top_indices = order[:25]
    for index in top_indices:
        candidate = joined[int(index)][0]
        place = places[int(index)]
        label = candidate.name or (f"near {place}" if place is not None else candidate.candidate_id)
        print(f"{potential[index]:6.2f}  {label}  ({candidate.lat:.4f},{candidate.lon:.4f})")


if __name__ == "__main__":
    main()
