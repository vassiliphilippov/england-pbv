"""Tests for the drive-time estimate and Pareto frontier used by the site."""

import numpy as np

from england_pbv.site.svg import horizon_panorama_svg, polar_reach_svg
from england_pbv.site.traveltime import (
    estimate_drive_minutes,
    estimate_drive_minutes_scalar,
    haversine_km,
    pareto_frontier_indices,
)


def test_haversine_london_manchester() -> None:
    distance = haversine_km(lat1=51.5074, lon1=-0.1278, lat2=53.4808, lon2=-2.2426)
    assert 255.0 < float(distance) < 270.0


def test_drive_minutes_plausible() -> None:
    # A 10 km hop is a ~15-25 minute local drive; 250 km is roughly 3-4.5 hours.
    assert 12.0 < estimate_drive_minutes_scalar(10.0) < 25.0
    assert 170.0 < estimate_drive_minutes_scalar(250.0) < 280.0


def test_drive_minutes_monotonic_and_vectorised() -> None:
    distances = np.linspace(0.0, 400.0, 200)
    minutes = estimate_drive_minutes(distances)
    assert minutes.shape == distances.shape
    assert bool(np.all(np.diff(minutes) > 0.0))


def test_pareto_frontier_orders_and_filters() -> None:
    minutes = np.array([10.0, 20.0, 5.0, 30.0], dtype=np.float64)
    scores = np.array([50.0, 60.0, 40.0, 55.0], dtype=np.float64)
    frontier = pareto_frontier_indices(drive_minutes=minutes, scores=scores)
    # Nearest first, each strictly more beautiful; the 30-min/55 point is dominated.
    assert frontier == [2, 0, 1]


def test_pareto_frontier_min_gain() -> None:
    minutes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scores = np.array([50.0, 50.4, 60.0], dtype=np.float64)
    frontier = pareto_frontier_indices(drive_minutes=minutes, scores=scores, min_score_gain=0.5)
    assert frontier == [0, 2]


def test_charts_stay_small() -> None:
    # 10,000+ pages inline these charts; a size regression here bloats the site.
    rng = np.random.default_rng(seed=7)
    horizon = rng.uniform(-2.0, 8.0, 720).astype(np.float32)
    veg = horizon + rng.uniform(0.0, 1.5, 720).astype(np.float32)
    d_far = rng.uniform(0.0, 60.0, 720).astype(np.float32)
    assert len(horizon_panorama_svg(horizon, horizon_veg_deg=veg, d_far_km=d_far)) < 9000
    assert len(polar_reach_svg(d_far)) < 6000
