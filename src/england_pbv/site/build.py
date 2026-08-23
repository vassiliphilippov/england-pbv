"""Generate the static website into docs/ (GitHub Pages).

Run: uv run python -m england_pbv.site.build
"""

import argparse
import html
import json
import re
import shutil
from dataclasses import dataclass

import numpy as np
from jinja2 import Environment, FileSystemLoader
from numpy.typing import NDArray
from PIL import Image

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
from england_pbv.site.traveltime import (
    drive_params,
    estimate_drive_minutes,
    haversine_km,
    pareto_frontier_indices,
)
from england_pbv.terrain.grid import load_dem_grid, load_uint8_grid
from england_pbv.verification.evaluate import load_scored
from england_pbv.viewshed.horizon import build_sampling_plan, sweep_batch
from england_pbv.viewshed.render import (
    best_view_directions,
    compass_label,
    render_panorama,
    render_view,
)

N_MAP_POINTS: int = 1500
N_PAGES_NATIONAL: int = 600
REGIONAL_CELL_M: float = 25000.0
REGIONAL_PAGES_PER_CELL: int = 5
# Finer fill so even low-scoring pockets link their local champions; these "lite"
# pages carry one camera view instead of two to keep the published site small.
FILL_CELL_M: float = 10000.0
FILL_PAGES_PER_CELL: int = 2
LITE_JPEG_QUALITY: int = 75
# Micro tier: the local top of every 5 km cell gets a page, taking the site past
# 10,000 pages. Micro pages carry one downscaled panorama and no camera views so
# the published site stays inside GitHub Pages' size budget.
MICRO_CELL_M: float = 5000.0
MICRO_PAGES_PER_CELL: int = 3
MICRO_JPEG_QUALITY: int = 56
MICRO_PANORAMA_WIDTH: int = 900
N_TOP_LIST: int = 50
N_SUBLIST: int = 20
COASTAL_WATER_FRACTION: float = 0.08
# "More beautiful places nearby": Pareto frontier over (drive time, score).
N_NEARBY_ROWS: int = 8
NEARBY_MIN_SCORE_GAIN: float = 0.1

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

COMPASS_POINTS: list[str] = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]  # fmt: skip


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] if len(slug) > 0 else "viewpoint"


def compass_point(bearing_deg: float) -> str:
    return COMPASS_POINTS[int(round(bearing_deg / 22.5)) % 16]


@dataclass(frozen=True, slots=True)
class ListRow:
    name: str
    region: str
    score: float
    rank: int
    slug: str | None
    is_discovery: bool


@dataclass(frozen=True, slots=True)
class NearbyRow:
    """One step of the 'more beautiful places nearby' Pareto frontier."""

    name: str
    slug: str
    score: float
    crow_km: float
    drive_minutes: float
    direction: str


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


def _select_pages(
    deduped: list[ScoredViewpoint],
) -> tuple[list[ScoredViewpoint], set[str], set[str]]:
    """Pick the pages to publish; returns (pages, lite_ids, micro_ids)."""
    page_ids: set[str] = {v.candidate.candidate_id for v in deduped[:N_PAGES_NATIONAL]}
    cell_counts: dict[tuple[int, int], int] = {}
    for item in deduped:
        cell = (
            int(item.candidate.easting // REGIONAL_CELL_M),
            int(item.candidate.northing // REGIONAL_CELL_M),
        )
        if cell_counts.get(cell, 0) >= REGIONAL_PAGES_PER_CELL:
            continue
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        page_ids.add(item.candidate.candidate_id)
    lite_ids: set[str] = set()
    fill_counts: dict[tuple[int, int], int] = {}
    for item in deduped:
        cell = (
            int(item.candidate.easting // FILL_CELL_M),
            int(item.candidate.northing // FILL_CELL_M),
        )
        if fill_counts.get(cell, 0) >= FILL_PAGES_PER_CELL:
            continue
        fill_counts[cell] = fill_counts.get(cell, 0) + 1
        if item.candidate.candidate_id not in page_ids:
            lite_ids.add(item.candidate.candidate_id)
            page_ids.add(item.candidate.candidate_id)
    micro_ids: set[str] = set()
    micro_counts: dict[tuple[int, int], int] = {}
    for item in deduped:
        cell = (
            int(item.candidate.easting // MICRO_CELL_M),
            int(item.candidate.northing // MICRO_CELL_M),
        )
        if micro_counts.get(cell, 0) >= MICRO_PAGES_PER_CELL:
            continue
        micro_counts[cell] = micro_counts.get(cell, 0) + 1
        if item.candidate.candidate_id not in page_ids:
            micro_ids.add(item.candidate.candidate_id)
            page_ids.add(item.candidate.candidate_id)
    pages = [v for v in deduped if v.candidate.candidate_id in page_ids]
    print(
        f"pages: {len(pages)} ({N_PAGES_NATIONAL} national "
        f"+ local top-{REGIONAL_PAGES_PER_CELL} per 25 km cell "
        f"+ {len(lite_ids)} lite (top-{FILL_PAGES_PER_CELL} per 10 km cell) "
        f"+ {len(micro_ids)} micro (top-{MICRO_PAGES_PER_CELL} per 5 km cell)"
    )
    return pages, lite_ids, micro_ids


def _nearby_frontier(
    *,
    item: ScoredViewpoint,
    lats: NDArray[np.float64],
    lons: NDArray[np.float64],
    scores: NDArray[np.float64],
    names: list[str],
    slug_list: list[str],
    self_index: int,
) -> list[NearbyRow]:
    """Pareto frontier of strictly-more-beautiful places by estimated drive time."""
    crow_km = haversine_km(lat1=item.candidate.lat, lon1=item.candidate.lon, lat2=lats, lon2=lons)
    better = (scores > item.view_potential + NEARBY_MIN_SCORE_GAIN) & (
        np.array([s != "" for s in slug_list])
    )
    better[self_index] = False
    candidate_indices = np.nonzero(better)[0]
    if len(candidate_indices) == 0:
        return []
    minutes = estimate_drive_minutes(crow_km[candidate_indices])
    frontier = pareto_frontier_indices(
        drive_minutes=minutes,
        scores=scores[candidate_indices],
        min_score_gain=NEARBY_MIN_SCORE_GAIN,
    )[:N_NEARBY_ROWS]
    rows: list[NearbyRow] = []
    for position in frontier:
        index = int(candidate_indices[position])
        d_lon = lons[index] - item.candidate.lon
        d_lat = lats[index] - item.candidate.lat
        bearing = float(
            np.degrees(
                np.arctan2(
                    np.sin(np.radians(d_lon)) * np.cos(np.radians(lats[index])),
                    np.sin(np.radians(d_lat)),
                )
            )
            % 360.0
        )
        rows.append(
            NearbyRow(
                name=names[index],
                slug=slug_list[index],
                score=float(scores[index]),
                crow_km=float(crow_km[index]),
                drive_minutes=float(minutes[position]),
                direction=compass_point(bearing),
            )
        )
    return rows


def _load_vote_photos() -> list[dict[str, object]]:
    if not paths.VOTE_PHOTOS_JSON.exists():
        return []
    raw = json.loads(paths.VOTE_PHOTOS_JSON.read_text(encoding="utf-8"))
    photos = raw["photos"]
    assert isinstance(photos, list)
    return photos


def build_site(*, max_pages: int | None = None, skip_pages: bool = False) -> None:
    scored = load_scored()
    deduped = _dedupe_by_rank(scored)
    print(f"{len(scored)} scored -> {len(deduped)} deduplicated for presentation")

    pages, lite_ids, micro_ids = _select_pages(deduped)
    if max_pages is not None:
        # Smoke-test mode: nearby links may dangle, never publish such a build.
        pages = pages[:max_pages]
        print(f"smoke test: building only the first {len(pages)} pages")
    slugs: dict[str, str] = {}
    used: set[str] = set()
    for item in pages:
        base = slugify(item.display_name)
        slug = f"{base}-{item.national_rank}" if base in used else base
        used.add(base)
        slugs[item.candidate.candidate_id] = slug

    # Autoescape everywhere; SVG/JSON blobs are explicitly |safe after their own hardening.
    env = Environment(loader=FileSystemLoader(paths.TEMPLATES_DIR), autoescape=True)
    site_dir = paths.SITE_DIR
    viewpoints_dir = site_dir / "viewpoints"
    renders_dir = site_dir / "renders"
    if not skip_pages:
        if viewpoints_dir.exists():
            # Slugs change between runs; stale pages must not accumulate.
            shutil.rmtree(viewpoints_dir)
        viewpoints_dir.mkdir(parents=True, exist_ok=True)
        if renders_dir.exists():
            shutil.rmtree(renders_dir)
        renders_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "assets").mkdir(parents=True, exist_ok=True)
    (site_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(paths.TEMPLATES_DIR / "style.css", site_dir / "assets" / "style.css")

    coombe_photo: str | None = None
    photo_source = site_dir / "assets" / "coombe-hill.jpg"
    if photo_source.exists():
        coombe_photo = "coombe-hill.jpg"

    # --- shared data blobs ---
    map_points = []
    for item in deduped[:N_MAP_POINTS]:
        map_points.append(
            {
                "la": round(item.candidate.lat, 5),
                "lo": round(item.candidate.lon, 5),
                "s": item.view_potential,
                # Escaped at build time: Leaflet popups render this as HTML.
                "n": html.escape(item.display_name),
                "u": slugs.get(item.candidate.candidate_id),
                "w": round(item.metrics.water_fraction, 3),
            }
        )
    # Full deduplicated dataset for the "best views in this area" selector.
    all_rows = [
        [
            round(item.candidate.lat, 5),
            round(item.candidate.lon, 5),
            item.view_potential,
            round(item.metrics.water_fraction, 2),
            html.escape(item.display_name),
            slugs.get(item.candidate.candidate_id) or "",
        ]
        for item in deduped
    ]
    points_js = (
        "const ALL_POINTS="
        + json.dumps(all_rows, separators=(",", ":")).replace("</", "<\\/")
        + ";"
    )
    (site_dir / "data" / "points.js").write_text(points_js, encoding="utf-8")
    print(f"points.js written ({len(all_rows)} points)")

    # Drive-time estimator shared by the map, postcode and viewpoint pages.
    drive_js = (
        "const DRIVE_PARAMS=" + json.dumps(drive_params(), separators=(",", ":")) + ";\n"
        "function driveMinutes(crowKm){\n"
        "  return DRIVE_PARAMS.coeff*Math.pow(Math.max(crowKm,0),DRIVE_PARAMS.exponent);\n"
        "}\n"
        "function haversineKm(la1,lo1,la2,lo2){\n"
        "  const r=Math.PI/180, p1=la1*r, p2=la2*r, dp=(la2-la1)*r, dl=(lo2-lo1)*r;\n"
        "  const a=Math.sin(dp/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;\n"
        "  return 2*6371*Math.asin(Math.sqrt(a));\n"
        "}\n"
    )
    (site_dir / "assets" / "traveltime.js").write_text(drive_js, encoding="utf-8")

    inland = [v for v in deduped if v.metrics.water_fraction < COASTAL_WATER_FRACTION]
    coastal = [v for v in deduped if v.metrics.water_fraction >= COASTAL_WATER_FRACTION]
    gems = [v for v in deduped if _is_discovery(v)]

    index_html = env.get_template("index.j2").render(
        root="",
        n_candidates=f"{len(scored):,}",
        n_map_points=len(map_points),
        n_all_points=f"{len(all_rows):,}",
        # "</" escaped so a name can never terminate the <script> element.
        map_points_json=json.dumps(map_points, separators=(",", ":")).replace("</", "<\\/"),
        top_overall=[_list_row(v, slugs) for v in deduped[:N_TOP_LIST]],
        top_inland=[_list_row(v, slugs) for v in inland[:N_SUBLIST]],
        top_coastal=[_list_row(v, slugs) for v in coastal[:N_SUBLIST]],
        hidden_gems=[_list_row(v, slugs) for v in gems[:N_SUBLIST]],
    )
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")
    print("index.html written")

    # --- postcode page ---
    postcode_html = env.get_template("postcode.j2").render(root="")
    (site_dir / "postcode.html").write_text(postcode_html, encoding="utf-8")
    print("postcode.html written")

    # --- voting page ---
    vote_photos = _load_vote_photos()
    if vote_photos:
        vote_html = env.get_template("vote.j2").render(
            root="",
            n_photos=len(vote_photos),
            photos_json=json.dumps(vote_photos, separators=(",", ":")).replace("</", "<\\/"),
        )
        (site_dir / "vote.html").write_text(vote_html, encoding="utf-8")
        print(f"vote.html written ({len(vote_photos)} photos)")
    else:
        print("vote.html skipped (verification/vote_photos.json missing or empty)")

    if skip_pages:
        print("viewpoint pages skipped (--skip-pages)")
    else:
        # --- viewpoint pages (re-run the engine for horizon profiles) ---
        dem = load_dem_grid(paths.DEM_GRID_NPY)
        landcover = load_uint8_grid(paths.LANDCOVER_GRID_NPY)
        plan = build_sampling_plan()
        eastings = np.array([v.candidate.easting for v in pages], dtype=np.float64)
        northings = np.array([v.candidate.northing for v in pages], dtype=np.float64)
        sweep = sweep_batch(dem, landcover, eastings=eastings, northings=northings, plan=plan)
        print(f"horizon profiles computed for {len(pages)} pages")

        # Arrays over the full deduplicated set for the nearby-frontier search.
        all_lats = np.array([v.candidate.lat for v in deduped], dtype=np.float64)
        all_lons = np.array([v.candidate.lon for v in deduped], dtype=np.float64)
        all_scores = np.array([v.view_potential for v in deduped], dtype=np.float64)
        all_names = [v.display_name for v in deduped]
        all_slugs = [slugs.get(v.candidate.candidate_id) or "" for v in deduped]
        index_of = {v.candidate.candidate_id: i for i, v in enumerate(deduped)}

        template = env.get_template("viewpoint.j2")
        for index, item in enumerate(pages):
            metrics = item.metrics
            slug = slugs[item.candidate.candidate_id]
            is_lite = item.candidate.candidate_id in lite_ids
            is_micro = item.candidate.candidate_id in micro_ids
            panorama = render_panorama(
                dem,
                landcover,
                easting=item.candidate.easting,
                northing=item.candidate.northing,
            )
            if is_micro:
                height = round(panorama.height * MICRO_PANORAMA_WIDTH / panorama.width)
                panorama = panorama.resize((MICRO_PANORAMA_WIDTH, height), Image.Resampling.LANCZOS)
                panorama.save(
                    renders_dir / f"{slug}.jpg", quality=MICRO_JPEG_QUALITY, optimize=True
                )
            else:
                panorama.save(
                    renders_dir / f"{slug}.jpg",
                    quality=LITE_JPEG_QUALITY if is_lite else 76,
                    optimize=True,
                )

            view_cards = []
            if not is_micro:
                directions = best_view_directions(
                    sweep.d_far_veg_m[index], max_directions=1 if is_lite else 2
                )
                for view_index, direction in enumerate(directions, start=1):
                    view_image = render_view(
                        dem,
                        landcover,
                        easting=item.candidate.easting,
                        northing=item.candidate.northing,
                        center_azimuth_deg=direction.azimuth_deg,
                    )
                    view_file = f"{slug}_view{view_index}.jpg"
                    view_image.save(
                        renders_dir / view_file,
                        quality=LITE_JPEG_QUALITY if is_lite else 78,
                        optimize=True,
                    )
                    view_cards.append(
                        {
                            "file": view_file,
                            "label": f"Looking {compass_label(direction.azimuth_deg)}",
                            "azimuth": round(direction.azimuth_deg),
                        }
                    )
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
            nearby = _nearby_frontier(
                item=item,
                lats=all_lats,
                lons=all_lons,
                scores=all_scores,
                names=all_names,
                slug_list=all_slugs,
                self_index=index_of[item.candidate.candidate_id],
            )
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
                "render_file": f"{slug}.jpg",
                "views": view_cards,
                "nearby": nearby,
            }
            page_html = template.render(root="../", vp=vp)
            (site_dir / "viewpoints" / f"{slug}.html").write_text(page_html, encoding="utf-8")
            if (index + 1) % 200 == 0:
                print(f"{index + 1}/{len(pages)} pages rendered", flush=True)
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
    parser = argparse.ArgumentParser(description="Build the static site into docs/")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="build only the first N viewpoint pages (smoke test; never publish)",
    )
    parser.add_argument(
        "--skip-pages",
        action="store_true",
        help="regenerate only index/postcode/vote/verification/methodology, keeping pages",
    )
    arguments = parser.parse_args()
    build_site(max_pages=arguments.max_pages, skip_pages=arguments.skip_pages)


if __name__ == "__main__":
    main()
