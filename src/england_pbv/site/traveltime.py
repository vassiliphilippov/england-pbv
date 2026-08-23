"""Estimated car travel times between points in England, and Pareto frontiers.

No routing engine fits a static site, so drive time is estimated from the
straight-line (great-circle) distance with a power law calibrated against
8,008 OSRM car routes between random England points:

    minutes = COEFF * crow_km ** EXPONENT

The fit (R^2 = 0.965, median abs error 5.8% vs OSRM free-flow durations)
absorbs both circuity (road/crow distance, ~1.6 for short hops falling to
~1.25 for long trips, consistent with the published England factor of ~1.4)
and the rise of average speed with trip length (local roads -> motorways,
consistent with DfT speed statistics). A traffic factor of 1.15 converts
OSRM's free-flow durations to typical real-world times.

The same formula runs in the site's JavaScript (parameters are injected via
``drive_params``) so build-time tables and client-side estimates agree.
Client pages may refine estimates with a public OSRM table request when the
network allows; the formula is the always-available baseline.
"""

import math

import numpy as np
from numpy.typing import NDArray

# minutes = COEFF * crow_km ** EXPONENT, calibrated against OSRM (see module docstring).
FREE_FLOW_COEFF: float = 2.61
EXPONENT: float = 0.809
TRAFFIC_FACTOR: float = 1.15
COEFF: float = FREE_FLOW_COEFF * TRAFFIC_FACTOR

EARTH_RADIUS_KM: float = 6371.0


def haversine_km(
    *,
    lat1: NDArray[np.float64] | float,
    lon1: NDArray[np.float64] | float,
    lat2: NDArray[np.float64] | float,
    lon2: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Great-circle distance in km; broadcasts like numpy."""
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=np.float64)) - np.radians(
        np.asarray(lon1, dtype=np.float64)
    )
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    result: NDArray[np.float64] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return result


def estimate_drive_minutes(crow_km: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimated driving minutes for straight-line distances in km (vectorised)."""
    distances = np.maximum(np.asarray(crow_km, dtype=np.float64), 0.0)
    minutes: NDArray[np.float64] = COEFF * distances**EXPONENT
    return minutes


def estimate_drive_minutes_scalar(crow_km: float) -> float:
    """Scalar convenience wrapper around estimate_drive_minutes."""
    return float(estimate_drive_minutes(np.array([crow_km], dtype=np.float64))[0])


def drive_params() -> dict[str, float]:
    """Formula parameters for injection into client-side JavaScript."""
    return {"coeff": COEFF, "exponent": EXPONENT}


def pareto_frontier_indices(
    *,
    drive_minutes: NDArray[np.float64],
    scores: NDArray[np.float64],
    min_score_gain: float = 0.0,
) -> list[int]:
    """Indices forming the (drive time, score) Pareto frontier.

    Sorted by drive time ascending, an item joins the frontier when its score
    beats every closer item by more than min_score_gain. The result is ordered
    from nearest to farthest, with strictly increasing scores.
    """
    order = np.argsort(drive_minutes, kind="stable")
    frontier: list[int] = []
    best = -math.inf
    for index in order:
        score = float(scores[index])
        if score > best + min_score_gain:
            frontier.append(int(index))
            best = score
    return frontier
