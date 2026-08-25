"""Render the next-gen prototype for the sampled sites and compose 3-row pairs.

Rows: real PHOTO / NEXT-GEN render (1 m LiDAR near field, HDRI sky, strict camera)
/ CURRENT renderer for reference. Written to outputs/calibration/pairs_nextgen/.

Run: uv run python -m england_pbv.viewshed.calibration2
"""

import json
import time

import numpy as np
import requests
from PIL import Image, ImageDraw

from england_pbv import paths
from england_pbv.terrain.grid import latlon_to_bng, load_dem_grid, load_uint8_grid
from england_pbv.viewshed.calibration import fetch_photo, hfov_from_focal_35mm
from england_pbv.viewshed.render import render_view
from england_pbv.viewshed.render2 import SUN_AZIMUTH_DEG, render_view2

PAIR_WIDTH: int = 1100
# CC0 daytime equirect skies (Poly Haven tonemapped previews); first that downloads wins.
HDRI_CANDIDATES: tuple[str, ...] = ("kloofendal_48d_partly_cloudy", "meadow_2", "sunflowers")
HDRI_URL: str = "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/{name}.jpg"


def load_hdri() -> np.ndarray:
    """Equirect sky as float32 RGB, horizontally rotated so its sun sits at 225°."""
    hdri_path = paths.OUTPUTS_DIR / "nextgen" / "sky_hdri.npy"
    if hdri_path.exists():
        return np.asarray(np.load(hdri_path), dtype=np.float32)
    for name in HDRI_CANDIDATES:
        try:
            resp = requests.get(HDRI_URL.format(name=name), timeout=300)
            resp.raise_for_status()
        except requests.RequestException:
            continue
        raw = paths.OUTPUTS_DIR / "nextgen" / f"{name}.jpg"
        raw.write_bytes(resp.content)
        img = Image.open(raw).convert("RGB")
        img = img.resize((4096, 2048), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        # Sun azimuth = brightest column in the upper half; roll so it lands at 225°.
        upper = arr[: arr.shape[0] // 2].sum(axis=(0, 2))
        sun_col = int(np.argmax(upper))
        target_col = int(SUN_AZIMUTH_DEG / 360.0 * arr.shape[1])
        arr = np.roll(arr, target_col - sun_col, axis=1)
        np.save(hdri_path, arr)
        print(f"HDRI: {name}, sun rolled {target_col - sun_col} px")
        return arr
    raise RuntimeError("no HDRI candidate downloaded")


def main() -> None:
    sites = json.loads((paths.OUTPUTS_DIR / "nextgen_sites.json").read_text(encoding="utf-8"))
    pairs_dir = paths.OUTPUTS_DIR / "calibration" / "pairs_nextgen"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    photos_dir = paths.OUTPUTS_DIR / "calibration" / "photos"

    dem50 = load_dem_grid(paths.DEM_GRID_NPY)
    lc50 = load_uint8_grid(paths.LANDCOVER_GRID_NPY)
    dem10 = np.load(paths.DEM10_GRID_NPY, mmap_mode="r")
    lc10 = np.load(paths.LANDCOVER10_GRID_NPY, mmap_mode="r")
    sat = np.load(paths.SATELLITE10_GRID_NPY, mmap_mode="r")
    hdri = load_hdri()
    tex_path = paths.OUTPUTS_DIR / "nextgen" / "ground_textures.npy"
    tex = (
        np.load(tex_path).astype(np.float32)
        if tex_path.exists()
        else np.ones((1, 1, 1, 1, 3), dtype=np.float32)
    )

    timings: dict[str, dict[str, float]] = {}
    for site in sites:
        key = site["key"]
        window_path = paths.OUTPUTS_DIR / "nextgen" / f"{key}_1m.npz"
        if not window_path.exists():
            print(f"{key}: LiDAR window missing, skipped")
            continue
        window = dict(np.load(window_path))
        photo_path = photos_dir / f"{key}.jpg"
        fetch_photo(site["url"], destination=photo_path)
        hfov = hfov_from_focal_35mm(float(site["focal35"]))
        aspect_w, aspect_h = site["aspect"]
        height = round(PAIR_WIDTH * aspect_h / aspect_w)
        bng = latlon_to_bng(float(site["lat"]), float(site["lon"]))

        t0 = time.perf_counter()
        new_render, cam_offset_m = render_view2(
            window,
            dem50,
            lc50,
            hdri,
            easting=bng.easting,
            northing=bng.northing,
            center_azimuth_deg=float(site["bearing"]),
            hfov_deg=hfov,
            width=PAIR_WIDTH,
            height=height,
            horizon_fraction=float(site["horizon"]),
            dem10=dem10,
            lc10=lc10,
            sat=sat,
            tex=tex,
        )
        new_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        old_render = render_view(
            dem50,
            lc50,
            easting=bng.easting,
            northing=bng.northing,
            center_azimuth_deg=float(site["bearing"]),
            hfov_deg=hfov,
            width=PAIR_WIDTH,
            height=height,
            pitch_deg=0.0,
            horizon_fraction=float(site["horizon"]),
            dem10=dem10,
            landcover10=lc10,
            satellite10=sat,
        )
        old_seconds = time.perf_counter() - t0
        timings[key] = {
            "new_s": round(new_seconds, 2),
            "old_s": round(old_seconds, 2),
            "cam_offset_m": round(cam_offset_m, 1),
        }
        print(f"{key}: new {new_seconds:.1f}s, old {old_seconds:.1f}s", flush=True)
        photo = Image.open(photo_path).convert("RGB")
        photo = photo.resize((PAIR_WIDTH, round(photo.height * PAIR_WIDTH / photo.width)))

        label_h = 30
        canvas = Image.new(
            "RGB",
            (PAIR_WIDTH, photo.height + new_render.height + old_render.height + label_h * 3 + 20),
            (16, 20, 24),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{site['name']} - PHOTO ({site['credit']})", fill=(238, 238, 232))
        canvas.paste(photo, (0, label_h))
        y = label_h + photo.height + 10
        draw.text(
            (10, y),
            f"NEXT-GEN RENDER - 1m LiDAR near field, real shadows, HDRI sky - "
            f"bearing {site['bearing']} deg, {site['focal35']}mm - camera moved "
            f"{cam_offset_m:.0f} m (grid-ref/vegetation reconciliation)",
            fill=(140, 235, 170),
        )
        canvas.paste(new_render, (0, y + label_h - 8))
        y = y + label_h - 8 + new_render.height + 10
        draw.text((10, y), "CURRENT RENDERER (for reference)", fill=(238, 238, 232))
        canvas.paste(old_render, (0, y + label_h - 8))
        out = pairs_dir / f"pair_{key}_n1.jpg"
        canvas.save(out, quality=88)
        print(out.name)
    (pairs_dir / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
