"""Synthetic 3D panorama renderer: what the view from a point should look like.

Ray-marches the leaf-on surface (terrain + tree canopies + buildings) per image column,
painting visible spans with land-cover colours, sun-and-slope shading and atmospheric
perspective. Calibrated by visual comparison against photographs of known viewpoints
(Coombe Hill, Mam Tor, Sutton Bank, ...) — see specifications/render_calibration.md.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from england_pbv.constants import EARTH_RADIUS_M, EYE_HEIGHT_M, REFRACTION_K
from england_pbv.viewshed.horizon import _bilinear, _landcover_at

RENDER_CURVATURE: float = (1.0 - REFRACTION_K) / (2.0 * EARTH_RADIUS_M)

PANORAMA_WIDTH: int = 1440  # 0.25 degree per column
PANORAMA_HEIGHT: int = 400
TOP_ANGLE_DEG: float = 8.0
BOTTOM_ANGLE_DEG: float = -12.0

TREE_HEIGHT_M: float = 15.0
BUILT_HEIGHT_M: float = 8.0

MAX_RENDER_DISTANCE_M: float = 60000.0
HAZE_DISTANCE_M: float = 13000.0  # distance at which ~63% of surface colour is lost to haze
SUN_AZIMUTH_DEG: float = 225.0  # afternoon sun from the south-west
SHADE_GAIN: float = 2.2
SHADE_AMBIENT: float = 0.88
# The viewpoint itself is a clearing: within this ring obstacle heights ramp from zero,
# absorbing the ~50 m position uncertainty of the land-cover grid.
CLEARING_RAMP_START_M: float = 120.0
CLEARING_RAMP_FULL_M: float = 260.0
EYE_BOOST_M: float = 0.8  # slight raise over the 1.7 m eye for canopy-edge robustness
MOOR_GRASS_START_M: float = 350.0  # grassland turns tawny acid-moor colours above this
MOOR_GRASS_FULL_M: float = 550.0

# Land-cover base colours (RGB 0..255), tuned against Geograph photos of known views.
# Index = WorldCover code // 10.
_BASE_COLORS: NDArray[np.float32] = np.array(
    [
        [176, 178, 168],  # 0 nodata -> neutral ground
        [86, 106, 56],  # 1 tree cover
        [126, 136, 84],  # 2 shrubland
        [150, 156, 96],  # 3 grassland (warm summer green)
        [192, 174, 116],  # 4 cropland (ripe straw)
        [158, 140, 130],  # 5 built-up (warm brick-grey)
        [186, 172, 142],  # 6 bare/sparse
        [240, 244, 248],  # 7 snow/ice
        [138, 168, 190],  # 8 water
        [128, 150, 118],  # 9 wetland
        [150, 158, 120],  # 10 moss/lichen
    ],
    dtype=np.float32,
)

# Sky gradient: zenith-ish blue at the top of the frame to pale warm haze at the horizon.
_SKY_TOP: NDArray[np.float32] = np.array([150, 180, 215], dtype=np.float32)
_SKY_HORIZON: NDArray[np.float32] = np.array([218, 228, 236], dtype=np.float32)
_HAZE_COLOR: NDArray[np.float32] = np.array([201, 214, 227], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class RenderSettings:
    width: int = PANORAMA_WIDTH
    height: int = PANORAMA_HEIGHT
    top_angle_deg: float = TOP_ANGLE_DEG
    bottom_angle_deg: float = BOTTOM_ANGLE_DEG


@njit(cache=True, inline="always")
def _cell_noise(x_m: float, y_m: float, scale_m: float) -> float:
    """Deterministic 0..1 noise keyed to a world-space cell of the given scale."""
    ix = int(x_m / scale_m)
    iy = int(y_m / scale_m)
    h = (ix * 73856093) ^ (iy * 19349663)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


@njit(cache=True, parallel=True, fastmath=True)
def _render_kernel(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    obs_e: float,
    obs_n: float,
    eye_z: float,
    width: int,
    height: int,
    top_rad: float,
    bottom_rad: float,
    base_colors: NDArray[np.float32],
    sky_top: NDArray[np.float32],
    sky_horizon: NDArray[np.float32],
    haze_color: NDArray[np.float32],
    image: NDArray[np.uint8],
) -> None:
    rad_per_row = (top_rad - bottom_rad) / height
    sun_az = math.radians(SUN_AZIMUTH_DEG)

    for col in prange(width):  # type: ignore[no-untyped-call, attr-defined]  # numba prange
        azimuth = 2.0 * math.pi * col / width
        dx = math.sin(azimuth)
        dy = math.cos(azimuth)
        sun_facing = math.cos(azimuth - sun_az)

        max_ang = bottom_rad
        prev_z = eye_z - EYE_HEIGHT_M
        prev_slope = 0.0
        r = 60.0
        step = 12.0
        while r < MAX_RENDER_DISTANCE_M:
            x = obs_e + dx * r
            y = obs_n + dy * r
            z = _bilinear(dem, x, y)
            lc_bin = _landcover_at(landcover, x, y)
            clearing = (r - CLEARING_RAMP_START_M) / (CLEARING_RAMP_FULL_M - CLEARING_RAMP_START_M)
            if clearing < 0.0:
                clearing = 0.0
            elif clearing > 1.0:
                clearing = 1.0
            z_surf = z
            if lc_bin == 1:
                # Broken canopy: tree height varies per ~35 m clump for a natural skyline.
                z_surf = z + clearing * TREE_HEIGHT_M * (0.55 + 0.9 * _cell_noise(x, y, 35.0))
            elif lc_bin == 5:
                z_surf = z + clearing * BUILT_HEIGHT_M * (0.5 + _cell_noise(x, y, 45.0))

            z_eff = z_surf - RENDER_CURVATURE * r * r
            ang = math.atan((z_eff - eye_z) / r)
            if ang > max_ang:
                # Paint the newly revealed span [max_ang, ang) in this column.
                radial_slope = 0.5 * (z - prev_z) / step + 0.5 * prev_slope
                prev_slope = radial_slope
                shade = SHADE_AMBIENT + SHADE_GAIN * radial_slope * sun_facing
                if lc_bin == 8:
                    shade = 1.02  # water keeps a flat sheen
                if shade < 0.6:
                    shade = 0.6
                elif shade > 1.15:
                    shade = 1.15
                # Field-scale patchwork variation plus fine ground texture.
                field = 0.88 + 0.24 * _cell_noise(x, y, 260.0)
                grain = 0.94 + 0.12 * _cell_noise(x, y, 18.0)
                if lc_bin == 1:
                    grain = 0.78 + 0.44 * _cell_noise(x, y, 22.0)  # foliage sparkle
                shade *= field * grain

                haze = 1.0 - math.exp(-r / HAZE_DISTANCE_M)
                # Patchwork/grain contrast fades with distance so the horizon stays calm.
                shade = 1.0 + (shade - 1.0) * (1.0 - 0.75 * haze)

                red = base_colors[lc_bin, 0]
                green = base_colors[lc_bin, 1]
                blue = base_colors[lc_bin, 2]
                if lc_bin == 3 and z > MOOR_GRASS_START_M:
                    # Acid moorland grass above the enclosure line: tawny-olive tops.
                    moor = (z - MOOR_GRASS_START_M) / (MOOR_GRASS_FULL_M - MOOR_GRASS_START_M)
                    if moor > 1.0:
                        moor = 1.0
                    red = red + (152.0 - red) * moor
                    green = green + (142.0 - green) * moor
                    blue = blue + (92.0 - blue) * moor
                red *= shade
                green *= shade
                blue *= shade
                red = red + (haze_color[0] - red) * haze
                green = green + (haze_color[1] - green) * haze
                blue = blue + (haze_color[2] - blue) * haze

                row_hi = int((top_rad - max_ang) / rad_per_row)
                row_lo = int((top_rad - ang) / rad_per_row)
                if row_lo < 0:
                    row_lo = 0
                if row_hi > height:
                    row_hi = height
                # Per-pixel texture keeps close spans from smearing into flat slabs;
                # it fades with haze so the distance stays soft.
                texture_amp = (1.0 - haze) * (0.16 if lc_bin != 1 else 0.30)
                span_rows = row_hi - row_lo
                for row in range(row_lo, row_hi):
                    hsh = ((col * 7919) ^ (row * 104729)) & 0x7FFFFFFF
                    hsh = (hsh ^ (hsh >> 11)) * 2654435761
                    noise = ((hsh >> 8) & 0xFFFF) / 65535.0 - 0.5
                    tex = 1.0 + texture_amp * noise
                    if lc_bin == 1 and span_rows > 3:
                        # Canopy lighting: sunlit tops, shadowed undersides.
                        depth_in_span = (row - row_lo) / span_rows
                        tex *= (1.14 - 0.42 * depth_in_span) * (1.0 - haze) + haze
                    rr = red * tex
                    gg = green * tex
                    bb = blue * tex
                    if rr > 255.0:
                        rr = 255.0
                    if gg > 255.0:
                        gg = 255.0
                    if bb > 255.0:
                        bb = 255.0
                    image[row, col, 0] = np.uint8(rr)
                    image[row, col, 1] = np.uint8(gg)
                    image[row, col, 2] = np.uint8(bb)
                max_ang = ang

            prev_z = z
            r += step
            # Angular step control: distance step grows so each step spans ~0.05 deg.
            step = r * 0.012
            if step < 12.0:
                step = 12.0
            elif step > 400.0:
                step = 400.0

        # Sky above the final skyline, with haze pooling at the horizon.
        horizon_row = int((top_rad - max_ang) / rad_per_row)
        if horizon_row > height:
            horizon_row = height
        for row in range(0, horizon_row):
            t = row / max(1, height * 0.9)
            f = math.exp(-2.2 * (1.0 - t))  # 0 near top, ~1 near horizon
            for channel in range(3):
                value = sky_top[channel] + (sky_horizon[channel] - sky_top[channel]) * f
                image[row, col, channel] = np.uint8(value)


def render_panorama(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    easting: float,
    northing: float,
    settings: RenderSettings | None = None,
) -> NDArray[np.uint8]:
    if settings is None:
        settings = RenderSettings()
    # Snap the camera to the highest ground within ~75 m: renders should stand on the
    # actual crest, not half a cell down the slope.
    best_e, best_n, ground = easting, northing, _bilinear(dem, easting, northing)
    for de in (-50.0, 0.0, 50.0):
        for dn in (-50.0, 0.0, 50.0):
            z_here = _bilinear(dem, easting + de, northing + dn)
            if z_here > ground:
                ground = z_here
                best_e, best_n = easting + de, northing + dn
    easting, northing = best_e, best_n
    eye_z = float(ground) + EYE_HEIGHT_M + EYE_BOOST_M
    image = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
    _render_kernel(
        dem,
        landcover,
        easting,
        northing,
        eye_z,
        settings.width,
        settings.height,
        math.radians(settings.top_angle_deg),
        math.radians(settings.bottom_angle_deg),
        _BASE_COLORS,
        _SKY_TOP,
        _SKY_HORIZON,
        _HAZE_COLOR,
        image,
    )
    return image
