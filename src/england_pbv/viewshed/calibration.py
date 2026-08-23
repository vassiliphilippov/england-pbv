"""Render the calibration photo set and compose photo-vs-render pairs.

Reads verification/render_calibration_photos.json (documented camera position, bearing,
EXIF lens, aspect, horizon placement), caches the photos, renders each scene with the
matched camera, and writes pairs to outputs/calibration/pairs. The iteration loop's
before/after comparisons run on these images.

Run: uv run python -m england_pbv.viewshed.calibration [--suffix vN]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng, load_dem_grid, load_uint8_grid
from england_pbv.viewshed.render import render_view

CALIBRATION_DIR: Path = paths.OUTPUTS_DIR / "calibration"
PAIR_WIDTH: int = 1100
USER_AGENT: str = "Mozilla/5.0 (Macintosh) england-pbv-calibration"


def hfov_from_focal_35mm(focal_35mm: float) -> float:
    assert focal_35mm > 0.0, "focal length is known"
    return 2.0 * math.degrees(math.atan(36.0 / (2.0 * focal_35mm)))


def fetch_photo(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 20000:
        return
    resp = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    destination.write_bytes(resp.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render calibration pairs")
    parser.add_argument("--suffix", default="", help="filename suffix for this iteration")
    parser.add_argument(
        "--photos",
        default="render_calibration_photos.json",
        help="photo-set file name inside verification/",
    )
    args = parser.parse_args()

    photos_dir = CALIBRATION_DIR / "photos"
    pairs_dir = CALIBRATION_DIR / "pairs"
    photos_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads((paths.VERIFICATION_DIR / args.photos).read_text(encoding="utf-8"))
    dem = load_dem_grid(paths.DEM_GRID_NPY)
    landcover = load_uint8_grid(paths.LANDCOVER_GRID_NPY)
    dem10 = None
    if paths.DEM10_GRID_NPY.exists():
        dem10 = np.load(paths.DEM10_GRID_NPY, mmap_mode="r")
        print("England 10 m DTM overlay active")
    landcover10 = None
    if paths.LANDCOVER10_GRID_NPY.exists():
        landcover10 = np.load(paths.LANDCOVER10_GRID_NPY, mmap_mode="r")
        print("England 10 m land-cover overlay active")
    satellite10 = None
    if paths.SATELLITE10_GRID_NPY.exists():
        satellite10 = np.load(paths.SATELLITE10_GRID_NPY, mmap_mode="r")
        print("England 10 m Sentinel-2 satellite overlay active")

    for entry in entries:
        key = entry["key"]
        photo_path = photos_dir / f"{key}.jpg"
        fetch_photo(entry["url"], destination=photo_path)

        hfov = hfov_from_focal_35mm(float(entry["focal35"]))
        aspect_w, aspect_h = entry["aspect"]
        height = round(PAIR_WIDTH * aspect_h / aspect_w)
        bng = latlon_to_bng(entry["lat"], entry["lon"])
        render = render_view(
            dem,
            landcover,
            easting=bng.easting,
            northing=bng.northing,
            center_azimuth_deg=float(entry["bearing"]),
            hfov_deg=hfov,
            width=PAIR_WIDTH,
            height=height,
            pitch_deg=0.0,
            horizon_fraction=float(entry["horizon"]),
            dem10=dem10,
            landcover10=landcover10,
            satellite10=satellite10,
        )

        photo = Image.open(photo_path).convert("RGB")
        photo = photo.resize((PAIR_WIDTH, round(photo.height * PAIR_WIDTH / photo.width)))

        label_height = 30
        canvas = Image.new(
            "RGB",
            (PAIR_WIDTH, photo.height + render.height + label_height * 2 + 10),
            (16, 20, 24),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{entry['name']} - PHOTO ({entry['credit']})", fill=(238, 238, 232))
        canvas.paste(photo, (0, label_height))
        y2 = label_height + photo.height + 10
        draw.text(
            (10, y2 + 8),
            f"OUR RENDER - bearing {entry['bearing']} deg, "
            f"{entry['focal35']}mm lens ({hfov:.0f} deg FOV)",
            fill=(238, 238, 232),
        )
        canvas.paste(render, (0, y2 + label_height))
        suffix = f"_{args.suffix}" if args.suffix else ""
        out_path = pairs_dir / f"pair_{key}{suffix}.jpg"
        canvas.save(out_path, quality=87)
        print(out_path.name)


if __name__ == "__main__":
    main()
