"""Versioned model parameters for the viewpoint-discovery pipeline.

These are engineering conventions, not discovered laws of scenic quality; see
specifications/metrics_specification.md for the rationale behind each value.
"""

ALGORITHM_VERSION: str = "0.1.0"

# --- National grid (EPSG:27700, British National Grid) ---
GRID_CELL_M: float = 50.0
GRID_ORIGIN_EASTING: float = 0.0
GRID_ORIGIN_NORTHING: float = 0.0
GRID_WIDTH_CELLS: int = 14000  # eastings 0..700 km
GRID_HEIGHT_CELLS: int = 25000  # northings 0..1250 km
BNG_EPSG: int = 27700
WGS84_EPSG: int = 4326
SEA_LEVEL_M: float = 0.0  # missing OS Terrain 50 tiles are open sea

# --- Observer model ---
EYE_HEIGHT_M: float = 1.7

# --- Viewshed geometry ---
MAX_VIEW_DISTANCE_M: float = 40000.0
EARTH_RADIUS_M: float = 6371000.0
REFRACTION_K: float = 1.0 / 7.0  # conventional optical refraction coefficient
N_AZIMUTHS: int = 720  # 0.5 degree step

# Radial sampling plan: (max_distance_m, step_m) segments, applied in order.
RAY_SEGMENTS: list[tuple[float, float]] = [
    (2000.0, 50.0),
    (8000.0, 100.0),
    (MAX_VIEW_DISTANCE_M, 200.0),
]
MIN_SAMPLE_DISTANCE_M: float = 75.0  # skip the observer's own cell

# Distance bands for area/angular composition metrics (metres, band edges).
DISTANCE_BAND_EDGES_M: list[float] = [0.0, 2000.0, 8000.0, 20000.0, MAX_VIEW_DISTANCE_M]

# Horizon-angle thresholds (degrees) for openness metrics.
OPEN_HORIZON_DEG: float = 2.0
FAR_VIEW_DISTANCE_M: float = 10000.0  # a bearing "sees far" if terrain visible beyond this

# --- Screening (candidate generation) ---
TPI_RADII_M: list[float] = [500.0, 2000.0, 10000.0]
SCREENING_KEEP_FRACTION: float = 0.02  # keep top 2% of England cells by screening score
NMS_RADIUS_M: float = 300.0  # non-maximum suppression radius between candidates

# --- Scoring ---
REGIONAL_PERCENTILE_RADIUS_M: float = 30000.0

# --- Land cover (ESA WorldCover 2021 v200 class codes) ---
WORLDCOVER_NODATA: int = 0
TREE_OBSTRUCTION_NEAR_M: float = 300.0  # near-field ring where woodland likely blocks the view
