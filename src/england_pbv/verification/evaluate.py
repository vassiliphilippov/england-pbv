"""Evaluate the ranking against the curated verification dataset.

Pass conventions (documented, versioned):
- a famous viewpoint passes when it reaches the top 25% nationally among candidates
  (candidates are already the top ~5% of England cells) OR the top 10% of its 30 km region;
- a negative control passes when it stays in the bottom 50% nationally.

Run: uv run python -m england_pbv.verification.evaluate
"""

import json

import numpy as np
from scipy.spatial import cKDTree

from england_pbv import paths
from england_pbv.constants import ALGORITHM_VERSION
from england_pbv.models import (
    ScoredViewpoint,
    VerificationReport,
    VerificationResult,
    VerificationViewpoint,
)
from england_pbv.terrain.grid import latlon_to_bng

MATCH_RADIUS_M: float = 300.0
POSITIVE_NATIONAL_PCT: float = 75.0
POSITIVE_REGIONAL_PCT: float = 90.0
NEGATIVE_NATIONAL_PCT: float = 50.0


def load_scored() -> list[ScoredViewpoint]:
    scored: list[ScoredViewpoint] = []
    with open(paths.SCORED_JSONL, encoding="utf-8") as handle:
        for line in handle:
            scored.append(ScoredViewpoint.model_validate_json(line))
    assert len(scored) > 0, "scored viewpoints exist"
    return scored


def main() -> None:
    scored = load_scored()
    n_total = len(scored)
    coords = np.array(
        [[s.candidate.easting, s.candidate.northing] for s in scored], dtype=np.float64
    )
    tree = cKDTree(coords)

    verification = json.loads(paths.VERIFICATION_VIEWPOINTS_JSON.read_text(encoding="utf-8"))
    viewpoints = [VerificationViewpoint.model_validate(raw) for raw in verification["viewpoints"]]

    results: list[VerificationResult] = []
    for viewpoint in viewpoints:
        bng = latlon_to_bng(viewpoint.lat, viewpoint.lon)
        distance, index = tree.query([bng.easting, bng.northing])
        if float(distance) > MATCH_RADIUS_M:
            results.append(
                VerificationResult(
                    name=viewpoint.name,
                    expected_high=viewpoint.expected_high,
                    matched_candidate_id=None,
                    match_distance_m=float(distance),
                    view_potential=None,
                    national_percentile=None,
                    regional_percentile=None,
                    passed=None,
                    detail=f"no candidate within {MATCH_RADIUS_M:.0f} m",
                )
            )
            continue
        match = scored[int(index)]
        national_pct = (1.0 - match.national_rank / n_total) * 100.0
        if viewpoint.expected_high:
            passed = (
                national_pct >= POSITIVE_NATIONAL_PCT
                or match.regional_percentile >= POSITIVE_REGIONAL_PCT
            )
        else:
            passed = national_pct <= NEGATIVE_NATIONAL_PCT
        results.append(
            VerificationResult(
                name=viewpoint.name,
                expected_high=viewpoint.expected_high,
                matched_candidate_id=match.candidate.candidate_id,
                match_distance_m=float(distance),
                view_potential=match.view_potential,
                national_percentile=round(national_pct, 2),
                regional_percentile=match.regional_percentile,
                passed=passed,
                detail=match.display_name,
            )
        )

    positives = [r for r in results if r.expected_high]
    negatives = [r for r in results if not r.expected_high]
    report = VerificationReport(
        algorithm_version=ALGORITHM_VERSION,
        n_positive=len(positives),
        n_negative=len(negatives),
        positive_top10pct_national=sum(
            1
            for r in positives
            if r.national_percentile is not None and r.national_percentile >= 90
        ),
        positive_top5pct_regional=sum(
            1
            for r in positives
            if r.regional_percentile is not None and r.regional_percentile >= 95
        ),
        negative_below_median=sum(
            1
            for r in negatives
            if r.national_percentile is not None and r.national_percentile <= 50
        ),
        results=results,
    )
    paths.VERIFICATION_REPORT_JSON.write_text(report.model_dump_json(indent=2))

    print(f"{'name':52s} {'exp':4s} {'pot':>6s} {'nat%':>6s} {'reg%':>6s} pass")
    for r in results:
        pot = f"{r.view_potential:.1f}" if r.view_potential is not None else "-"
        nat = f"{r.national_percentile:.1f}" if r.national_percentile is not None else "-"
        reg = f"{r.regional_percentile:.1f}" if r.regional_percentile is not None else "-"
        flag = {True: "PASS", False: "FAIL", None: "SKIP"}[r.passed]
        expected = "HIGH" if r.expected_high else "low"
        print(f"{r.name[:52]:52s} {expected:4s} {pot:>6s} {nat:>6s} {reg:>6s} {flag}")

    pos_pass = sum(1 for r in positives if r.passed is True)
    neg_pass = sum(1 for r in negatives if r.passed is True)
    print(f"\npositives: {pos_pass}/{len(positives)} pass", end="; ")
    print(f"negatives: {neg_pass}/{len(negatives)} pass")
    print(f"report -> {paths.VERIFICATION_REPORT_JSON}")


if __name__ == "__main__":
    main()
