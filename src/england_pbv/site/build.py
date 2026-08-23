"""Generate the static website into docs/ (GitHub Pages).

Run: uv run python -m england_pbv.site.build
"""

import json
import re
import shutil
from dataclasses import dataclass

import numpy as np
from jinja2 import Environment, FileSystemLoader

from england_pbv import paths
from england_pbv.models import ScoredViewpoint, VerificationReport, VerificationViewpoint
from england_pbv.pipeline.refine import _dedupe_by_rank
from england_pbv.site.links import (
    geograph_square_url,
    google_maps_url,
    openstreetmap_url,
    os_grid_ref,
    osmaps_url,
    peakfinder_url,
)
from england_pbv.site.svg import horizon_panorama_svg, polar_reach_svg
from england_pbv.terrain.grid import load_dem_grid, load_uint8_grid
from england_pbv.verification.evaluate import load_scored
from england_pbv.viewshed.horizon import build_sampling_plan, sweep_batch

N_MAP_POINTS: int = 1500
N_PAGES: int = 250
N_TOP_LIST: int = 50
N_SUBLIST: int = 20
COASTAL_WATER_FRACTION: float = 0.08

LANDCOVER_COLORS: dict[str, str] = {
    "woodland": "#55803c",
    "shrubland": "#8faa5a",
    "grassland": "#a8c686",
    "cropland": "#d9c078",
    "built-up": "#a9756b",
    "bare ground": "#c9b8a0",
    "water": "#6da3c9",
    "wetland": "#7fb5a2",
    "snow/ice": "#e8ecef",
    "moss/lichen": "#b5c4a1",
}

COMPONENT_LABELS: list[tuple[str, str]] = [
    ("prospect", "Prospect"),
    ("openness", "Openness"),
    ("drop", "Drop"),
    ("depth", "Depth"),
    ("diversity", "Variety"),
    ("clearness", "Clearness"),
]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] if len(slug) > 0 else "viewpoint"


@dataclass(frozen=True, slots=True)
class ListRow:
    name: str
    region: str
    score: float
    rank: int
    slug: str | None
    is_discovery: bool


def _region_of(item: ScoredViewpoint) -> str:
    if item.region_hint is not None:
        return f"near {item.region_hint}"
    return "—"


def _is_discovery(item: ScoredViewpoint) -> bool:
    return item.candidate.source.value == "screening" and item.candidate.name is None


def _list_row(item: ScoredViewpoint, slugs: dict[str, str]) -> ListRow:
    return ListRow(
        name=item.display_name,
        region=_region_of(item),
        score=item.view_potential,
        rank=item.national_rank,
        slug=slugs.get(item.candidate.candidate_id),
        is_discovery=_is_discovery(item),
    )


def build_site() -> None:
    scored = load_scored()
    deduped = _dedupe_by_rank(scored)
    print(f"{len(scored)} scored -> {len(deduped)} deduplicated for presentation")

    pages = deduped[:N_PAGES]
    slugs: dict[str, str] = {}
    used: set[str] = set()
    for item in pages:
        base = slugify(item.display_name)
        slug = f"{base}-{item.national_rank}" if base in used else base
        used.add(base)
        slugs[item.candidate.candidate_id] = slug

    env = Environment(loader=FileSystemLoader(paths.TEMPLATES_DIR), autoescape=False)
    site_dir = paths.SITE_DIR
    (site_dir / "viewpoints").mkdir(parents=True, exist_ok=True)
    (site_dir / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy(paths.TEMPLATES_DIR / "style.css", site_dir / "assets" / "style.css")

    coombe_photo: str | None = None
    photo_source = site_dir / "assets" / "coombe-hill.jpg"
    if photo_source.exists():
        coombe_photo = "coombe-hill.jpg"

    # --- index ---
    map_points = []
    for item in deduped[:N_MAP_POINTS]:
        map_points.append(
            {
                "la": round(item.candidate.lat, 5),
                "lo": round(item.candidate.lon, 5),
                "s": item.view_potential,
                "n": item.display_name,
                "u": slugs.get(item.candidate.candidate_id),
                "w": round(item.metrics.water_fraction, 3),
            }
        )
    inland = [v for v in deduped if v.metrics.water_fraction < COASTAL_WATER_FRACTION]
    coastal = [v for v in deduped if v.metrics.water_fraction >= COASTAL_WATER_FRACTION]
    gems = [v for v in deduped if _is_discovery(v)]

    index_html = env.get_template("index.j2").render(
        root="",
        n_candidates=f"{len(scored):,}",
        n_map_points=len(map_points),
        map_points_json=json.dumps(map_points, separators=(",", ":")),
        top_overall=[_list_row(v, slugs) for v in deduped[:N_TOP_LIST]],
        top_inland=[_list_row(v, slugs) for v in inland[:N_SUBLIST]],
        top_coastal=[_list_row(v, slugs) for v in coastal[:N_SUBLIST]],
        hidden_gems=[_list_row(v, slugs) for v in gems[:N_SUBLIST]],
    )
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")
    print("index.html written")

    # --- viewpoint pages (re-run the engine for horizon profiles) ---
    dem = load_dem_grid(paths.DEM_GRID_NPY)
    landcover = load_uint8_grid(paths.LANDCOVER_GRID_NPY)
    plan = build_sampling_plan()
    eastings = np.array([v.candidate.easting for v in pages], dtype=np.float64)
    northings = np.array([v.candidate.northing for v in pages], dtype=np.float64)
    sweep = sweep_batch(dem, landcover, eastings=eastings, northings=northings, plan=plan)
    print(f"horizon profiles computed for {len(pages)} pages")

    template = env.get_template("viewpoint.j2")
    for index, item in enumerate(pages):
        metrics = item.metrics
        horizon_deg = np.rad2deg(sweep.horizon_rad[index])
        horizon_veg_deg = np.rad2deg(sweep.horizon_veg_rad[index])
        d_far_veg_km = sweep.d_far_veg_m[index] / 1000.0

        landcover_rows = []
        for label, fraction in sorted(
            metrics.landcover_angular_fractions.items(), key=lambda kv: -kv[1]
        ):
            if label == "unknown" or fraction < 0.005:
                continue
            landcover_rows.append(
                {
                    "label": label,
                    "pct": fraction * 100.0,
                    "color": LANDCOVER_COLORS.get(label, "#cccccc"),
                }
            )

        components = [
            {"name": label, "value": getattr(item.components, key)}
            for key, label in COMPONENT_LABELS
        ]
        vp = {
            "name": item.display_name,
            "region": _region_of(item),
            "rank": item.national_rank,
            "regional_pct": item.regional_percentile,
            "score": item.view_potential,
            "components": components,
            "horizon_svg": horizon_panorama_svg(
                horizon_deg, horizon_veg_deg=horizon_veg_deg, d_far_km=d_far_veg_km
            ),
            "polar_svg": polar_reach_svg(d_far_veg_km),
            "landcover": landcover_rows,
            "shannon": metrics.shannon_diversity,
            "elevation": item.candidate.elevation_m,
            "visible_area": metrics.total_visible_area_km2,
            "d_max": metrics.d_far_max_km,
            "arc": metrics.longest_far_arc_veg_deg,
            "drop": metrics.max_sector_drop_m,
            "depression": metrics.mean_depression_deg,
            "retention": metrics.veg_retention,
            "grid_ref": os_grid_ref(item.candidate.easting, item.candidate.northing),
            "lat": f"{item.candidate.lat:.5f}",
            "lon": f"{item.candidate.lon:.5f}",
            "source": item.candidate.source.value.replace("_", " "),
            "gmaps": google_maps_url(item.candidate.lat, item.candidate.lon),
            "osmaps": osmaps_url(item.candidate.lat, item.candidate.lon),
            "osm": openstreetmap_url(item.candidate.lat, item.candidate.lon),
            "peakfinder": peakfinder_url(item.candidate.lat, item.candidate.lon),
            "geograph": geograph_square_url(item.candidate.easting, item.candidate.northing),
        }
        slug = slugs[item.candidate.candidate_id]
        html = template.render(root="../", vp=vp)
        (site_dir / "viewpoints" / f"{slug}.html").write_text(html, encoding="utf-8")
    print(f"{len(pages)} viewpoint pages written")

    # --- verification page ---
    report = VerificationReport.model_validate_json(paths.VERIFICATION_REPORT_JSON.read_text())
    verification = json.loads(paths.VERIFICATION_VIEWPOINTS_JSON.read_text(encoding="utf-8"))
    by_name = {
        raw["name"]: VerificationViewpoint.model_validate(raw) for raw in verification["viewpoints"]
    }
    positives = []
    negatives = []
    for result in report.results:
        viewpoint = by_name.get(result.name)
        nat = f"{result.national_percentile:.0f}" if result.national_percentile is not None else "—"
        reg = f"{result.regional_percentile:.0f}" if result.regional_percentile is not None else "—"
        if result.expected_high:
            photo = "—"
            if viewpoint is not None and viewpoint.photo_verified is True:
                photo = "view confirmed"
            elif viewpoint is not None and viewpoint.photo_verified is False:
                photo = "photo shows the hill/monument"
            positives.append(
                {
                    "name": result.name,
                    "region": viewpoint.region if viewpoint is not None else "",
                    "nat": nat,
                    "reg": reg,
                    "photo": photo,
                    "photo_page": viewpoint.geograph_photo_page if viewpoint is not None else None,
                    "passed": result.passed is True,
                }
            )
        else:
            why = ""
            if viewpoint is not None:
                why_full = viewpoint.why_famous.replace("NEGATIVE CONTROL", "")
                why = why_full.split("Tests failure mode:")[-1].strip()[:130]
            negatives.append(
                {
                    "name": result.name,
                    "why": why,
                    "nat": nat,
                    "passed": result.passed is True,
                }
            )
    verification_html = env.get_template("verification.j2").render(
        root="",
        n_positive=report.n_positive,
        n_negative=report.n_negative,
        pos_pass=sum(1 for p in positives if p["passed"]),
        neg_pass=sum(1 for n in negatives if n["passed"]),
        positives=positives,
        negatives=negatives,
    )
    (site_dir / "verification.html").write_text(verification_html, encoding="utf-8")

    # --- methodology ---
    methodology_html = env.get_template("methodology.j2").render(root="", coombe_photo=coombe_photo)
    (site_dir / "methodology.html").write_text(methodology_html, encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site written to {site_dir}")


def main() -> None:
    build_site()


if __name__ == "__main__":
    main()
