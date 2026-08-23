"""Single source of truth for every file location used by the pipeline."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
GRID_DIR: Path = DATA_DIR / "grids"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
SITE_DIR: Path = REPO_ROOT / "docs"
VERIFICATION_DIR: Path = REPO_ROOT / "verification"
TEMPLATES_DIR: Path = Path(__file__).resolve().parent / "site" / "templates"

# Raw downloads
TERRAIN50_ZIP: Path = RAW_DIR / "terr50_gagg_gb.zip"
WORLDCOVER_DIR: Path = RAW_DIR / "worldcover"
OSM_VIEWPOINTS_JSON: Path = RAW_DIR / "osm_viewpoints.json"
OSM_PEAKS_JSON: Path = RAW_DIR / "osm_peaks.json"
OSM_PLACES_JSON: Path = RAW_DIR / "osm_places.json"
DOBIH_CSV: Path = RAW_DIR / "dobih_hills.csv"
CALIBRATION_PATHS_JSON: Path = RAW_DIR / "calibration_paths.json"
CALIBRATION_HABITATS_JSON: Path = RAW_DIR / "calibration_habitats.json"
CALIBRATION_ROCKS_JSON: Path = RAW_DIR / "calibration_rocks.json"
ENGLAND_BOUNDARY_GEOJSON: Path = RAW_DIR / "england_boundary.geojson"

# National grids (EPSG:27700, 50 m)
DEM_GRID_NPY: Path = GRID_DIR / "gb_dem_50m.npy"
DEM10_GRID_NPY: Path = GRID_DIR / "england_dem_10m.npy"
LANDCOVER_GRID_NPY: Path = GRID_DIR / "gb_landcover_50m.npy"
LANDCOVER10_GRID_NPY: Path = GRID_DIR / "england_landcover_10m.npy"
ENGLAND_MASK_NPY: Path = GRID_DIR / "england_mask_50m.npy"

# Pipeline artifacts
CANDIDATES_JSONL: Path = OUTPUTS_DIR / "candidates.jsonl"
METRICS_JSONL: Path = OUTPUTS_DIR / "view_metrics.jsonl"
SCORED_JSONL: Path = OUTPUTS_DIR / "scored_viewpoints.jsonl"
HORIZON_PROFILES_NPZ: Path = OUTPUTS_DIR / "horizon_profiles.npz"
VERIFICATION_REPORT_JSON: Path = OUTPUTS_DIR / "verification_report.json"

# Curated inputs (committed)
VERIFICATION_VIEWPOINTS_JSON: Path = VERIFICATION_DIR / "viewpoints.json"
VOTE_PHOTOS_JSON: Path = VERIFICATION_DIR / "vote_photos.json"


def ensure_dirs() -> None:
    for directory in (RAW_DIR, GRID_DIR, OUTPUTS_DIR, WORLDCOVER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
