"""Parse named points of interest: OSM viewpoints/peaks/places and DoBIH hills."""

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from england_pbv import paths


@dataclass(frozen=True, slots=True)
class NamedPoint:
    source_id: str
    name: str | None
    lat: float
    lon: float
    elevation_m: float | None = None
    prominence_m: float | None = None


def load_osm_points(osm_json: Path, source_prefix: str) -> list[NamedPoint]:
    payload = json.loads(osm_json.read_text(encoding="utf-8"))
    points: list[NamedPoint] = []
    for element in payload.get("elements", []):
        if "lat" in element and "lon" in element:
            lat, lon = element["lat"], element["lon"]
        elif "center" in element:
            lat, lon = element["center"]["lat"], element["center"]["lon"]
        else:
            continue
        tags = element.get("tags", {})
        elevation: float | None = None
        raw_ele = tags.get("ele")
        if raw_ele is not None:
            try:
                elevation = float(str(raw_ele).replace("m", "").strip())
            except ValueError:
                elevation = None
        points.append(
            NamedPoint(
                source_id=f"{source_prefix}{element['id']}",
                name=tags.get("name"),
                lat=float(lat),
                lon=float(lon),
                elevation_m=elevation,
            )
        )
    return points


def load_dobih_hills(min_drop_m: float) -> list[NamedPoint]:
    zip_path = paths.DOBIH_CSV.with_suffix(".zip")
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        assert len(csv_names) == 1, "DoBIH archive contains one CSV"
        raw = archive.read(csv_names[0]).decode("utf-8", errors="replace")

    points: list[NamedPoint] = []
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        try:
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            drop = float(row["Drop"]) if row.get("Drop") not in (None, "") else 0.0
            elevation = float(row["Metres"]) if row.get("Metres") not in (None, "") else None
        except (KeyError, ValueError):
            continue
        if drop < min_drop_m:
            continue
        points.append(
            NamedPoint(
                source_id=f"dobih{row['Number']}",
                name=row.get("Name") or None,
                lat=lat,
                lon=lon,
                elevation_m=elevation,
                prominence_m=drop,
            )
        )
    assert len(points) > 1000, "DoBIH yields a plausible hill count"
    return points
