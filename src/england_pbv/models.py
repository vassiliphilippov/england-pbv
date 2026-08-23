"""Pydantic artifact models — the JSON/JSONL shapes written and read by pipeline stages."""

from pydantic import BaseModel, ConfigDict

from england_pbv.enums import CandidateSource, ViewKind

type CandidateId = str


class CandidatePoint(BaseModel):
    """A location worth running the full horizon analysis on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: CandidateId
    easting: float
    northing: float
    lat: float
    lon: float
    elevation_m: float
    source: CandidateSource
    name: str | None = None
    tpi_500m: float | None = None
    tpi_2km: float | None = None
    tpi_10km: float | None = None
    relief_2km: float | None = None


class ViewMetrics(BaseModel):
    """Objective view-geometry and composition metrics for one candidate.

    All "angular" quantities are visual-magnitude weights: the share of the observer's
    panorama occupied by terrain, computed from horizon-angle increments during the sweep.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: CandidateId

    # Visible plan area (km^2) per distance band; edges in constants.DISTANCE_BAND_EDGES_M.
    visible_area_km2_by_band: list[float]
    total_visible_area_km2: float

    # Terrain angular area (square degrees of the panorama) per distance band.
    angular_area_deg2_by_band: list[float]
    total_terrain_angular_deg2: float

    # Horizon profile summary (degrees).
    mean_horizon_deg: float
    median_horizon_deg: float
    p90_horizon_deg: float
    skyline_total_variation_deg: float

    # Openness and reach (bare terrain).
    open_fraction: float
    far_fraction: float
    longest_far_arc_deg: float
    d_far_median_km: float
    d_far_p90_km: float
    d_far_max_km: float

    # Vegetation-aware reach: trees (+15 m) and buildings (+8 m) act as blockers.
    far_fraction_veg: float
    longest_far_arc_veg_deg: float
    d_far_veg_p90_km: float
    visible_area_veg_km2: float
    veg_retention: float  # visible_area_veg / visible_area (1.0 when nothing to lose)

    # Downward prospect and relief.
    mean_depression_deg: float
    max_sector_drop_m: float
    visible_relief_m: float

    # Land-cover composition of the visible panorama (angular fractions, 0..1, by label).
    landcover_angular_fractions: dict[str, float]
    shannon_diversity: float
    water_fraction: float
    built_fraction: float
    tree_fraction: float

    # Near-field obstruction risk: fraction of bearings whose first ring is woodland.
    near_tree_fraction: float
    observer_landcover: str


class ComponentScores(BaseModel):
    """Percentile-normalised score components (0..100 each) among all England candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prospect: float
    openness: float
    drop: float
    depth: float
    diversity: float
    clearness: float


class ScoredViewpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidatePoint
    metrics: ViewMetrics
    components: ComponentScores
    view_potential: float  # 0..100, mean of components (a documented convention, not "beauty")
    national_rank: int
    regional_percentile: float  # percentile among candidates within 30 km
    display_name: str
    region_hint: str | None = None


class VerificationViewpoint(BaseModel):
    """A curated real-world viewpoint (or negative control) used to validate the ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    lat: float
    lon: float
    region: str
    view_kind: ViewKind
    why_famous: str
    expected_high: bool  # False for negative controls
    coord_source: str | None = None
    elevation_m: float | None = None
    main_view_direction: str | None = None
    geograph_photo_page: str | None = None
    geograph_image_url: str | None = None
    confidence: str | None = None
    photo_verified: bool | None = None
    photo_verdict: str | None = None
    notes: str | None = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    expected_high: bool
    matched_candidate_id: CandidateId | None
    match_distance_m: float | None
    view_potential: float | None
    national_percentile: float | None
    regional_percentile: float | None
    passed: bool | None
    detail: str


class VerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm_version: str
    n_positive: int
    n_negative: int
    positive_top10pct_national: int
    positive_top5pct_regional: int
    negative_below_median: int
    results: list[VerificationResult]
