"""Build a free-licensed dataset of English landscape-view photographs for pair voting.

Sweeps a fixed set of seed areas spread across England — famous scenic uplands and
ordinary lowland farmland alike — querying the Wikimedia Commons API (CirrusSearch
``nearcoord`` + ``imageinfo``) for geolocated "view from" / panorama photographs.
Only CC BY / CC BY-SA / CC0 / public-domain images with camera coordinates inside
England are kept, capped per area so geographic coverage stays spread.

Run: uv run python -m england_pbv.data.vote_photos
"""

import argparse
import html
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import requests
from pydantic import BaseModel, ConfigDict

from england_pbv import paths

COMMONS_API_URL: str = "https://commons.wikimedia.org/w/api.php"
USER_AGENT: str = (
    "england-pbv/0.1 (vote-photos dataset builder; "
    "https://github.com/vassiliphilippov/england-pbv; vassiliphilippov@gmail.com)"
)
REQUEST_TIMEOUT_S: float = 60.0
REQUEST_INTERVAL_S: float = 0.2
RETRY_DELAYS_S: list[float] = [2.0, 5.0, 15.0]
RETRYABLE_STATUS_CODES: list[int] = [429, 500, 502, 503, 504]

SEARCH_RADIUS_KM: int = 12
RESULTS_PER_QUERY: int = 20
THUMB_WIDTH_PX: int = 1024
MIN_IMAGE_WIDTH_PX: int = 800
MAX_PHOTOS_PER_AREA: int = 4
DEDUPE_DISTANCE_M: float = 500.0
EARTH_RADIUS_M: float = 6371000.0

# Generous England bounding box (Scilly to Berwick) for accepting camera coordinates.
ENGLAND_LAT_MIN: float = 49.8
ENGLAND_LAT_MAX: float = 55.85
ENGLAND_LON_MIN: float = -6.5
ENGLAND_LON_MAX: float = 1.9

FILE_TITLE_PREFIX: str = "File:"
COORDINATE_KIND_CAMERA: str = "camera"
COORDINATE_KIND_OBJECT: str = "object"
EXIF_HEADING_FIELD: str = "GPSImgDirection"
UNKNOWN_AUTHOR: str = "Unknown"

SEARCH_QUERY_TEMPLATES: list[str] = [
    'intitle:"view from" nearcoord:{radius}km,{lat},{lon} filetype:bitmap',
    "panorama nearcoord:{radius}km,{lat},{lon} filetype:bitmap",
    'intitle:"looking" nearcoord:{radius}km,{lat},{lon} filetype:bitmap',
]

# Titles/categories matching these are almost never open landscape views.
_EXCLUDED_SUBJECT_PATTERN: re.Pattern[str] = re.compile(
    r"\b(interior|inside|church|chapel|cathedral|priory|station|museum|portrait|close-?up"
    r"|plaque|sign|signpost|bench|gate|door|doorway|train|locomotive|tram|bus|aircraft"
    r"|aeroplane|airplane|helicopter|ship|ferry|statue|memorial|gravestone|cemetery"
    r"|churchyard|shopfront|pub|caf[eé])\b",
    re.IGNORECASE,
)

_HTML_TAG_PATTERN: re.Pattern[str] = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class SeedArea:
    name: str
    lat: float
    lon: float


# ~70 sample areas covering England: celebrated upland/coastal scenery alongside
# deliberately ordinary lowland farmland, fen, marsh, and edge-of-town country.
SEED_AREAS: list[SeedArea] = [
    SeedArea(name="Lake District - Keswick", lat=54.60, lon=-3.13),
    SeedArea(name="Lake District - Langdale", lat=54.44, lon=-3.09),
    SeedArea(name="Lake District - Windermere", lat=54.37, lon=-2.92),
    SeedArea(name="Lake District - Wasdale", lat=54.44, lon=-3.28),
    SeedArea(name="Peak District - Mam Tor", lat=53.35, lon=-1.81),
    SeedArea(name="Peak District - Stanage Edge", lat=53.35, lon=-1.63),
    SeedArea(name="Peak District - Dovedale", lat=53.06, lon=-1.78),
    SeedArea(name="Staffordshire - The Roaches", lat=53.16, lon=-1.99),
    SeedArea(name="Yorkshire Dales - Malham", lat=54.07, lon=-2.16),
    SeedArea(name="Yorkshire Dales - Wensleydale", lat=54.30, lon=-2.20),
    SeedArea(name="Yorkshire Dales - Swaledale", lat=54.38, lon=-2.05),
    SeedArea(name="Yorkshire Dales - Ingleborough", lat=54.16, lon=-2.40),
    SeedArea(name="Howgill Fells", lat=54.37, lon=-2.55),
    SeedArea(name="North York Moors - Sutton Bank", lat=54.24, lon=-1.21),
    SeedArea(name="North York Moors - Rosedale", lat=54.35, lon=-0.88),
    SeedArea(name="Yorkshire Coast - Robin Hood's Bay", lat=54.44, lon=-0.55),
    SeedArea(name="Northumberland - Cheviot Hills", lat=55.45, lon=-2.15),
    SeedArea(name="Northumberland - Hadrian's Wall", lat=55.00, lon=-2.39),
    SeedArea(name="Northumberland Coast - Bamburgh", lat=55.60, lon=-1.72),
    SeedArea(name="North Pennines - Teesdale", lat=54.65, lon=-2.19),
    SeedArea(name="Forest of Bowland", lat=53.96, lon=-2.53),
    SeedArea(name="Cotswolds - Broadway", lat=52.03, lon=-1.86),
    SeedArea(name="Cotswolds - Cleeve Hill", lat=51.92, lon=-2.01),
    SeedArea(name="Cotswolds - Stroud Valleys", lat=51.75, lon=-2.20),
    SeedArea(name="Chilterns - Ivinghoe Beacon", lat=51.84, lon=-0.60),
    SeedArea(name="Chilterns - Wendover", lat=51.76, lon=-0.74),
    SeedArea(name="Malvern Hills", lat=52.10, lon=-2.34),
    SeedArea(name="Shropshire - Long Mynd", lat=52.55, lon=-2.86),
    SeedArea(name="Shropshire - The Wrekin", lat=52.67, lon=-2.55),
    SeedArea(name="Mendip Hills - Cheddar", lat=51.28, lon=-2.77),
    SeedArea(name="Quantock Hills", lat=51.12, lon=-3.19),
    SeedArea(name="Exmoor - Porlock", lat=51.20, lon=-3.60),
    SeedArea(name="Dartmoor - Haytor", lat=50.58, lon=-3.75),
    SeedArea(name="Dartmoor - Princetown", lat=50.55, lon=-3.99),
    SeedArea(name="Bodmin Moor", lat=50.53, lon=-4.61),
    SeedArea(name="Cornwall - St Ives", lat=50.20, lon=-5.49),
    SeedArea(name="Cornwall - Tintagel", lat=50.66, lon=-4.75),
    SeedArea(name="Cornwall - Lizard", lat=49.97, lon=-5.21),
    SeedArea(name="South Devon - Salcombe", lat=50.24, lon=-3.77),
    SeedArea(name="Dorset - Lulworth", lat=50.62, lon=-2.25),
    SeedArea(name="Dorset - Golden Cap", lat=50.73, lon=-2.83),
    SeedArea(name="Purbeck - Corfe Castle", lat=50.64, lon=-2.06),
    SeedArea(name="Somerset - Glastonbury Tor", lat=51.14, lon=-2.70),
    SeedArea(name="South Downs - Devil's Dyke", lat=50.88, lon=-0.21),
    SeedArea(name="South Downs - Seven Sisters", lat=50.76, lon=0.13),
    SeedArea(name="South Downs - Butser Hill", lat=50.98, lon=-0.98),
    SeedArea(name="North Downs - Box Hill", lat=51.25, lon=-0.31),
    SeedArea(name="Kent Downs - Wye", lat=51.18, lon=0.94),
    SeedArea(name="Kent - White Cliffs of Dover", lat=51.13, lon=1.33),
    SeedArea(name="Isle of Wight - Tennyson Down", lat=50.66, lon=-1.55),
    SeedArea(name="Wiltshire - Pewsey Vale", lat=51.36, lon=-1.77),
    SeedArea(name="Wiltshire - Westbury White Horse", lat=51.26, lon=-2.14),
    SeedArea(name="Lincolnshire Wolds", lat=53.35, lon=-0.10),
    SeedArea(name="Lincolnshire Fens - Boston", lat=52.98, lon=-0.02),
    SeedArea(name="Norfolk - Cromer", lat=52.92, lon=1.30),
    SeedArea(name="Norfolk Broads", lat=52.69, lon=1.50),
    SeedArea(name="Norfolk - Breckland", lat=52.45, lon=0.75),
    SeedArea(name="Suffolk - Dedham Vale", lat=51.96, lon=0.99),
    SeedArea(name="Suffolk Coast - Southwold", lat=52.33, lon=1.67),
    SeedArea(name="Fens - Ely", lat=52.40, lon=0.26),
    SeedArea(name="Fens - Wisbech", lat=52.66, lon=0.16),
    SeedArea(name="Somerset Levels", lat=51.10, lon=-2.88),
    SeedArea(name="Leicestershire Wolds", lat=52.60, lon=-0.95),
    SeedArea(name="Warwickshire - Feldon", lat=52.20, lon=-1.55),
    SeedArea(name="Northamptonshire Uplands", lat=52.25, lon=-1.00),
    SeedArea(name="Essex - Dengie Marshes", lat=51.70, lon=0.80),
    SeedArea(name="Essex - Epping Forest", lat=51.66, lon=0.05),
    SeedArea(name="Cheshire Plain - Beeston", lat=53.12, lon=-2.69),
    SeedArea(name="Vale of York", lat=53.95, lon=-1.20),
    SeedArea(name="Yorkshire Wolds", lat=54.00, lon=-0.60),
    SeedArea(name="Holderness", lat=53.85, lon=-0.20),
    SeedArea(name="Cambridgeshire - Gog Magog Hills", lat=52.16, lon=0.16),
    SeedArea(name="Nottinghamshire - Sherwood", lat=53.20, lon=-1.07),
    SeedArea(name="Herefordshire Farmland", lat=52.10, lon=-2.65),
    SeedArea(name="New Forest", lat=50.87, lon=-1.63),
    SeedArea(name="Berkshire Downs", lat=51.54, lon=-1.42),
    SeedArea(name="Oxfordshire - Wittenham Clumps", lat=51.63, lon=-1.18),
    SeedArea(name="North Kent Marshes", lat=51.42, lon=0.60),
    SeedArea(name="Romney Marsh", lat=51.02, lon=0.85),
    SeedArea(name="Lancashire - Fylde", lat=53.80, lon=-2.85),
    SeedArea(name="Sefton Coast - Formby", lat=53.55, lon=-3.07),
    SeedArea(name="County Durham", lat=54.75, lon=-1.60),
    SeedArea(name="Solway Coast - Silloth", lat=54.87, lon=-3.40),
    SeedArea(name="Trent Valley", lat=52.85, lon=-1.40),
    SeedArea(name="London - Hampstead Heath", lat=51.56, lon=-0.16),
    SeedArea(name="Birmingham - Lickey Hills", lat=52.38, lon=-2.00),
    SeedArea(name="Ashdown Forest", lat=51.07, lon=0.03),
    SeedArea(name="Mid Devon", lat=50.80, lon=-3.55),
    SeedArea(name="Bristol - Avon Gorge", lat=51.46, lon=-2.63),
]


@dataclass(frozen=True, slots=True)
class CameraLocation:
    lat: float
    lon: float


class VotePhoto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    lat: float
    lon: float
    heading_deg: float | None
    area: str
    image_url: str
    page_url: str
    author: str
    license: str
    width: int
    height: int


class VotePhotoCollection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    photos: list[VotePhoto]


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _as_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class CommonsClient:
    """Rate-limited Wikimedia Commons API client (one request per REQUEST_INTERVAL_S)."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._last_request_monotonic: float | None = None

    def _throttle(self) -> None:
        if self._last_request_monotonic is not None:
            wait_s = REQUEST_INTERVAL_S - (time.monotonic() - self._last_request_monotonic)
            if wait_s > 0.0:
                time.sleep(wait_s)
        self._last_request_monotonic = time.monotonic()

    def _query(self, params: dict[str, str]) -> dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            self._throttle()
            try:
                response = self._session.get(
                    COMMONS_API_URL, params=params, timeout=REQUEST_TIMEOUT_S
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise requests.HTTPError(f"retryable HTTP status {response.status_code}")
                response.raise_for_status()
                payload: object = response.json()
                payload_dict = _as_dict(payload)
                if "error" in payload_dict:
                    # Covers maxlag backoff requests and transient API-side errors.
                    raise requests.RequestException(f"API error: {payload_dict.get('error')}")
                return payload_dict
            except requests.RequestException as error:
                last_error = error
                if attempt < len(RETRY_DELAYS_S):
                    time.sleep(RETRY_DELAYS_S[attempt])
        raise RuntimeError(f"Commons API request failed after retries: {last_error}")

    def search_files(self, search_query: str) -> list[dict[str, object]]:
        params: dict[str, str] = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": search_query,
            "gsrnamespace": "6",
            "gsrlimit": str(RESULTS_PER_QUERY),
            "prop": "imageinfo|coordinates",
            "iiprop": "url|size|extmetadata|metadata",
            "iiurlwidth": str(THUMB_WIDTH_PX),
            "iiextmetadatafilter": (
                "LicenseShortName|Artist|Categories|GPSLatitude|GPSLongitude|GPSImgDirection"
            ),
            "coprop": "type",
            "coprimary": "all",
            "colimit": "max",
            "maxlag": "5",
        }
        payload = self._query(params=params)
        query_dict = _as_dict(payload.get("query"))
        pages: list[dict[str, object]] = []
        for raw_page in _as_list(query_dict.get("pages")):
            page = _as_dict(raw_page)
            if len(page) > 0:
                pages.append(page)
        return pages


def _license_allowed(license_short_name: str) -> bool:
    lowered = license_short_name.strip().lower()
    if "public domain" in lowered or lowered == "pd" or lowered.startswith("pd-"):
        return True
    if lowered.startswith("cc0"):
        return True
    if lowered.startswith("cc by"):
        # Non-commercial / no-derivatives variants are not acceptable for the voting site.
        return "-nc" not in lowered and "-nd" not in lowered
    return False


def _strip_html(raw: str) -> str:
    text = _HTML_TAG_PATTERN.sub(" ", raw)
    return " ".join(html.unescape(text).split())


def _display_title(title: str) -> str:
    cleaned = title.removeprefix(FILE_TITLE_PREFIX)
    stem, _, extension = cleaned.rpartition(".")
    if stem != "" and len(extension) <= 4:
        return stem
    return cleaned


def _ext_value(extmetadata: dict[str, object], field: str) -> str | None:
    value = _as_dict(extmetadata.get(field)).get("value")
    if isinstance(value, str) and len(value.strip()) > 0:
        return value.strip()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return None


def _parse_exif_number(value: object) -> float | None:
    direct = _as_float(value)
    if direct is not None:
        return direct
    if isinstance(value, str) and "/" in value:
        numerator_text, _, denominator_text = value.partition("/")
        numerator = _as_float(numerator_text)
        denominator = _as_float(denominator_text)
        if numerator is not None and denominator is not None and denominator != 0.0:
            return numerator / denominator
    return None


def _parse_heading(
    metadata_entries: list[object],
    extmetadata: dict[str, object],
) -> float | None:
    for raw_entry in metadata_entries:
        entry = _as_dict(raw_entry)
        if _as_str(entry.get("name")) != EXIF_HEADING_FIELD:
            continue
        heading = _parse_exif_number(entry.get("value"))
        if heading is not None:
            return heading % 360.0
    ext_heading = _parse_exif_number(_ext_value(extmetadata, EXIF_HEADING_FIELD))
    if ext_heading is not None:
        return ext_heading % 360.0
    return None


def _inside_england(location: CameraLocation) -> bool:
    return (
        ENGLAND_LAT_MIN <= location.lat <= ENGLAND_LAT_MAX
        and ENGLAND_LON_MIN <= location.lon <= ENGLAND_LON_MAX
    )


def _camera_location(
    page: dict[str, object],
    extmetadata: dict[str, object],
) -> CameraLocation | None:
    fallback: CameraLocation | None = None
    for raw_entry in _as_list(page.get("coordinates")):
        entry = _as_dict(raw_entry)
        lat = _as_float(entry.get("lat"))
        lon = _as_float(entry.get("lon"))
        if lat is None or lon is None:
            continue
        location = CameraLocation(lat=lat, lon=lon)
        kind = _as_str(entry.get("type"))
        is_primary = entry.get("primary") is True
        # On Commons the primary GeoData coordinate is the camera position; "object"
        # coordinates describe the pictured subject, not where the photographer stood.
        if kind == COORDINATE_KIND_CAMERA or (is_primary and kind != COORDINATE_KIND_OBJECT):
            return location
        if fallback is None and kind != COORDINATE_KIND_OBJECT:
            fallback = location
    gps_lat = _as_float(_ext_value(extmetadata, "GPSLatitude"))
    gps_lon = _as_float(_ext_value(extmetadata, "GPSLongitude"))
    if gps_lat is not None and gps_lon is not None:
        return CameraLocation(lat=gps_lat, lon=gps_lon)
    return fallback


def _distance_m(a: CameraLocation, b: CameraLocation) -> float:
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _photo_from_page(page: dict[str, object], area_name: str) -> VotePhoto | None:
    page_id = _as_int(page.get("pageid"))
    title = _as_str(page.get("title"))
    if page_id is None or title is None:
        return None
    imageinfo_entries = _as_list(page.get("imageinfo"))
    if len(imageinfo_entries) == 0:
        return None
    info = _as_dict(imageinfo_entries[0])
    width = _as_int(info.get("width"))
    height = _as_int(info.get("height"))
    if width is None or height is None or width < MIN_IMAGE_WIDTH_PX:
        return None
    extmetadata = _as_dict(info.get("extmetadata"))
    license_name = _ext_value(extmetadata, "LicenseShortName")
    if license_name is None or not _license_allowed(license_name):
        return None
    categories = _ext_value(extmetadata, "Categories")
    if _EXCLUDED_SUBJECT_PATTERN.search(title) is not None:
        return None
    if categories is not None and _EXCLUDED_SUBJECT_PATTERN.search(categories) is not None:
        return None
    location = _camera_location(page=page, extmetadata=extmetadata)
    if location is None or not _inside_england(location):
        return None
    image_url = _as_str(info.get("thumburl"))
    if image_url is None:
        image_url = _as_str(info.get("url"))
    page_url = _as_str(info.get("descriptionurl"))
    if image_url is None or page_url is None:
        return None
    author_html = _ext_value(extmetadata, "Artist")
    author = _strip_html(author_html) if author_html is not None else UNKNOWN_AUTHOR
    if author == "":
        author = UNKNOWN_AUTHOR
    heading = _parse_heading(_as_list(info.get("metadata")), extmetadata)
    return VotePhoto(
        id=str(page_id),
        title=_display_title(title),
        lat=location.lat,
        lon=location.lon,
        heading_deg=heading,
        area=area_name,
        image_url=image_url,
        page_url=page_url,
        author=author,
        license=license_name,
        width=width,
        height=height,
    )


def _portrait_last(photo: VotePhoto) -> int:
    return 0 if photo.width > photo.height else 1


def collect_photos(client: CommonsClient, max_per_area: int) -> list[VotePhoto]:
    accepted: list[VotePhoto] = []
    accepted_ids: set[str] = set()
    accepted_locations: list[CameraLocation] = []
    for index, seed in enumerate(SEED_AREAS, start=1):
        candidates: dict[str, VotePhoto] = {}
        for template in SEARCH_QUERY_TEMPLATES:
            search_query = template.format(radius=SEARCH_RADIUS_KM, lat=seed.lat, lon=seed.lon)
            for page in client.search_files(search_query=search_query):
                photo = _photo_from_page(page=page, area_name=seed.name)
                if photo is not None and photo.id not in candidates:
                    candidates[photo.id] = photo
        kept = 0
        # Landscape orientation first; the stable sort keeps search relevance within groups.
        for photo in sorted(candidates.values(), key=_portrait_last):
            if kept >= max_per_area:
                break
            if photo.id in accepted_ids:
                continue
            location = CameraLocation(lat=photo.lat, lon=photo.lon)
            too_close = any(
                _distance_m(location, other) < DEDUPE_DISTANCE_M for other in accepted_locations
            )
            if too_close:
                continue
            accepted.append(photo)
            accepted_ids.add(photo.id)
            accepted_locations.append(location)
            kept += 1
        print(f"[{index:2d}/{len(SEED_AREAS)}] {seed.name}: kept {kept} of {len(candidates)}")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch free-licensed English landscape-view photos from Wikimedia Commons"
    )
    parser.add_argument("--max-per-area", type=int, default=MAX_PHOTOS_PER_AREA)
    args = parser.parse_args()

    client = CommonsClient()
    photos = collect_photos(client=client, max_per_area=args.max_per_area)
    assert len(photos) > 0, "at least one photo was collected"

    paths.VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    collection = VotePhotoCollection(photos=photos)
    paths.VOTE_PHOTOS_JSON.write_text(collection.model_dump_json(indent=2), encoding="utf-8")

    area_counts: Counter[str] = Counter(photo.area for photo in photos)
    print(f"\nwrote {len(photos)} photos to {paths.VOTE_PHOTOS_JSON}")
    print(f"areas represented: {len(area_counts)} of {len(SEED_AREAS)}")
    for area, count in sorted(area_counts.items()):
        print(f"  {count}  {area}")


if __name__ == "__main__":
    main()
