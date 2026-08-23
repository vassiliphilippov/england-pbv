"""Shared definition of the score-input vector and frozen percentile transforms.

The scoring stage builds percentiles over the national candidate population; the
refinement stage re-scores alternative observer positions against those same frozen
distributions so that "is this spot better" means the same thing everywhere.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from england_pbv.models import ViewMetrics

INPUT_NAMES: list[str] = [
    "ang_beyond_2km",
    "visible_area",
    "far_veg",
    "arc_veg",
    "drop",
    "depression",
    "d90_veg",
    "depth_entropy",
    "shannon",
    "retention",
    "built_penalty",
]

# Component -> input indices (each component is the mean of its inputs' percentiles).
COMPONENT_INPUTS: dict[str, list[int]] = {
    "prospect": [0, 1],
    "openness": [2, 3],
    "drop": [4, 5],
    "depth": [6, 7],
    "diversity": [8],
    "clearness": [9, 10],
}
COMPONENT_ORDER: list[str] = ["prospect", "openness", "drop", "depth", "diversity", "clearness"]

# The percentile composite measures terrain POTENTIAL; the retention factor grounds it
# in what a standing person actually sees. Raw leaf-on retention (not its percentile —
# the national distribution compresses 9% and 20% retention onto neighbouring
# percentiles) scales the composite through a smoothstep: full credit at >=30%
# retention, floor at <=3%. Calibrated 2026-08-24 against the 60-viewpoint
# verification set: positives hold at 51/60, negatives improve 17->20 to 18/20, and
# Box Hill's open Salomons Memorial viewpoint (a failing expected-high positive)
# overtakes the wooded Box Hill summit. Steeper hi values break wooded-gap
# viewpoints (Surprise View, Newlands Corner, ~30-35% retention) — keep hi at 0.30.
RETENTION_FACTOR_FLOOR: float = 0.65
RETENTION_SMOOTHSTEP_LO: float = 0.03
RETENTION_SMOOTHSTEP_HI: float = 0.30
_RETENTION_INPUT_INDEX: int = INPUT_NAMES.index("retention")


def retention_factor(retention: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smoothstep multiplier 0.65..1.0 from raw leaf-on retention."""
    t = np.clip(
        (retention - RETENTION_SMOOTHSTEP_LO) / (RETENTION_SMOOTHSTEP_HI - RETENTION_SMOOTHSTEP_LO),
        0.0,
        1.0,
    )
    smooth = 3.0 * t * t - 2.0 * t * t * t
    return RETENTION_FACTOR_FLOOR + (1.0 - RETENTION_FACTOR_FLOOR) * smooth


def band_depth_entropy(angular_by_band: list[float]) -> float:
    total = sum(angular_by_band)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in angular_by_band:
        p = value / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy / math.log(len(angular_by_band))


def metric_inputs(metrics: ViewMetrics) -> list[float]:
    return [
        float(sum(metrics.angular_area_deg2_by_band[1:])),
        metrics.total_visible_area_km2,
        metrics.far_fraction_veg,
        metrics.longest_far_arc_veg_deg,
        metrics.max_sector_drop_m,
        metrics.mean_depression_deg,
        metrics.d_far_veg_p90_km,
        band_depth_entropy(metrics.angular_area_deg2_by_band),
        metrics.shannon_diversity,
        metrics.veg_retention,
        1.0 - metrics.built_fraction,
    ]


@dataclass(frozen=True, slots=True)
class ScoredComponents:
    components: dict[str, float]
    composite: float


class FrozenPercentiles:
    """Percentile transforms frozen from the national candidate population."""

    def __init__(self, input_matrix: NDArray[np.float64]) -> None:
        assert input_matrix.shape[1] == len(INPUT_NAMES), "one column per input"
        self._sorted: list[NDArray[np.float64]] = [
            np.sort(input_matrix[:, index]) for index in range(len(INPUT_NAMES))
        ]
        self._n = input_matrix.shape[0]

    def percentiles(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """(k, n_inputs) raw values -> (k, n_inputs) percentiles 0..100."""
        result = np.empty_like(vectors)
        for index in range(len(INPUT_NAMES)):
            positions = np.searchsorted(self._sorted[index], vectors[:, index], side="right")
            result[:, index] = positions / self._n * 100.0
        return result

    def composite(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        pct = self.percentiles(vectors)
        totals = np.zeros(vectors.shape[0], dtype=np.float64)
        for component in COMPONENT_ORDER:
            indices = COMPONENT_INPUTS[component]
            totals += np.mean(pct[:, indices], axis=1)
        return totals / len(COMPONENT_ORDER) * retention_factor(vectors[:, _RETENTION_INPUT_INDEX])

    def score_one(self, vector: list[float]) -> ScoredComponents:
        pct = self.percentiles(np.array([vector], dtype=np.float64))[0]
        components: dict[str, float] = {}
        for component in COMPONENT_ORDER:
            indices = COMPONENT_INPUTS[component]
            components[component] = float(np.mean(pct[indices]))
        factor = float(
            retention_factor(np.array([vector[_RETENTION_INPUT_INDEX]], dtype=np.float64))[0]
        )
        composite = float(np.mean([components[c] for c in COMPONENT_ORDER])) * factor
        return ScoredComponents(components=components, composite=composite)
