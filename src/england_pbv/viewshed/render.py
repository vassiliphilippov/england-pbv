"""Synthetic renderers: 360-degree panoramas and rectilinear "camera" views.

Ray-marches the leaf-on surface (terrain + tree canopies + buildings) per image column,
painting visible spans with land-cover colours, sun-and-slope shading, per-pixel texture,
hedgerow hints at field boundaries, procedural clouds and atmospheric perspective. Both
projections share one kernel: panoramas map elevation linearly to rows; camera views use a
rectilinear (tan) projection like a real lens. Everything renders at 2x and downsamples for
anti-aliasing. Calibrated against photographs of known viewpoints — see
specifications/render_calibration.md.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from PIL import Image, ImageEnhance

from england_pbv.constants import EARTH_RADIUS_M, EYE_HEIGHT_M, REFRACTION_K
from england_pbv.viewshed.horizon import _bilinear, _landcover_at

RENDER_CURVATURE: float = (1.0 - REFRACTION_K) / (2.0 * EARTH_RADIUS_M)

PANORAMA_WIDTH: int = 1440
PANORAMA_HEIGHT: int = 400
PANO_TOP_DEG: float = 8.0
PANO_BOTTOM_DEG: float = -12.0

VIEW_WIDTH: int = 920
VIEW_HEIGHT: int = 518
VIEW_HFOV_DEG: float = 68.0
VIEW_PITCH_DEG: float = -3.0  # camera tilted slightly down: more landscape, less sky
VIEW_HORIZON_FRACTION: float = 0.42  # image row of the level horizon, from the top

SUPERSAMPLE: int = 2

TREE_HEIGHT_M: float = 15.0
BUILT_HEIGHT_M: float = 8.0

MAX_RENDER_DISTANCE_M: float = 60000.0
HAZE_DISTANCE_M: float = 16000.0
SUN_AZIMUTH_DEG: float = 225.0
SUN_ELEVATION_DEG: float = 38.0
SHADE_AMBIENT: float = 0.52
SHADE_DIFFUSE: float = 0.78
CLOUD_SHADOW_SCALE_M: float = 1250.0
CLOUD_SHADOW_STRENGTH: float = 0.30
CLEARING_RAMP_START_M: float = 120.0
CLEARING_RAMP_FULL_M: float = 260.0
EYE_BOOST_M: float = 0.8
MOOR_GRASS_START_M: float = 350.0
MOOR_GRASS_FULL_M: float = 550.0
FIELD_CELL_M: float = 260.0
HEDGE_SHADE: float = 0.74  # darkening applied to the one-sample strip at a field boundary

PROJECTION_PANORAMA: int = 0
PROJECTION_RECTILINEAR: int = 1

# England 10 m DTM overlay (int16 decimetres, row 0 = south); values <= threshold are
# nodata (sea, Wales, Scotland) and fall back to the 50 m national grid.
DEM10_X0: float = 80000.0
DEM10_Y0: float = 4000.0
DEM10_CELL: float = 10.0
DEM10_INVALID: int = -30000

# Land-cover base colours (RGB 0..255); index = WorldCover code // 10.
_BASE_COLORS: NDArray[np.float32] = np.array(
    [
        [176, 178, 168],  # 0 nodata
        [86, 106, 56],  # 1 tree cover
        [126, 136, 84],  # 2 shrubland
        [150, 156, 96],  # 3 grassland
        [192, 174, 116],  # 4 cropland
        [158, 140, 130],  # 5 built-up
        [186, 172, 142],  # 6 bare/sparse
        [240, 244, 248],  # 7 snow/ice
        [138, 168, 190],  # 8 water
        [128, 150, 118],  # 9 wetland
        [150, 158, 120],  # 10 moss/lichen
    ],
    dtype=np.float32,
)

_SKY_TOP: NDArray[np.float32] = np.array([150, 180, 215], dtype=np.float32)
_SKY_HORIZON: NDArray[np.float32] = np.array([218, 228, 236], dtype=np.float32)
_HAZE_COLOR: NDArray[np.float32] = np.array([201, 214, 227], dtype=np.float32)
_CLOUD_COLOR: NDArray[np.float32] = np.array([249, 250, 251], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ViewDirection:
    azimuth_deg: float
    quality: float  # mean capped visible distance over the window (km)


@njit(cache=True, inline="always")
def _sample_elev(
    dem50: NDArray[np.float32],
    dem10: NDArray[np.int16],
    has10: bool,
    x: float,
    y: float,
) -> float:
    """Elevation from the England 10 m grid, falling back to the 50 m national grid."""
    if has10:
        fc = (x - DEM10_X0) / DEM10_CELL - 0.5
        fr = (y - DEM10_Y0) / DEM10_CELL - 0.5
        h10, w10 = dem10.shape
        if fc >= 0.0 and fc < w10 - 1.0 and fr >= 0.0 and fr < h10 - 1.0:
            c0 = int(fc)
            r0 = int(fr)
            a = dem10[r0, c0]
            b = dem10[r0, c0 + 1]
            c = dem10[r0 + 1, c0]
            d = dem10[r0 + 1, c0 + 1]
            if a > DEM10_INVALID and b > DEM10_INVALID and c > DEM10_INVALID and d > DEM10_INVALID:
                fx = fc - c0
                fy = fr - r0
                return float(
                    0.1
                    * (
                        a * (1.0 - fx) * (1.0 - fy)
                        + b * fx * (1.0 - fy)
                        + c * (1.0 - fx) * fy
                        + d * fx * fy
                    )
                )
    return _bilinear(dem50, x, y)


@njit(cache=True, inline="always")
def _hash01(ix: int, iy: int) -> float:
    h = (ix * 73856093) ^ (iy * 19349663)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


@njit(cache=True, inline="always")
def _cell_noise(x_m: float, y_m: float, scale_m: float) -> float:
    return _hash01(int(x_m / scale_m), int(y_m / scale_m))


@njit(cache=True, inline="always")
def _smooth_noise(u: float, v: float) -> float:
    """Bilinear value noise on the integer lattice of (u, v)."""
    iu = int(math.floor(u))
    iv = int(math.floor(v))
    fu = u - iu
    fv = v - iv
    fu = fu * fu * (3.0 - 2.0 * fu)
    fv = fv * fv * (3.0 - 2.0 * fv)
    a = _hash01(iu, iv)
    b = _hash01(iu + 1, iv)
    c = _hash01(iu, iv + 1)
    d = _hash01(iu + 1, iv + 1)
    return a + (b - a) * fu + (c - a) * fv + (d - c - b + a) * fu * fv


@njit(cache=True, inline="always")
def _cloud_density(azimuth_deg: float, elev_deg: float, seed_u: float, seed_v: float) -> float:
    """Two-octave stretched noise: cumulus clumps elongated along the horizon."""
    u = azimuth_deg * 0.055 + seed_u
    v = elev_deg * 0.55 + seed_v
    return 0.65 * _smooth_noise(u, v) + 0.35 * _smooth_noise(u * 2.7 + 13.7, v * 2.7 + 71.3)


@njit(cache=True, parallel=True, fastmath=True)
def _render_kernel(
    dem: NDArray[np.float32],
    dem10: NDArray[np.int16],
    has10: bool,
    landcover: NDArray[np.uint8],
    obs_e: float,
    obs_n: float,
    eye_z: float,
    col_azimuth: NDArray[np.float64],
    projection: int,
    height: int,
    top_rad: float,
    bottom_rad: float,
    f_v_px: float,
    cy_px: float,
    pitch_rad: float,
    cloud_seed_u: float,
    cloud_seed_v: float,
    base_colors: NDArray[np.float32],
    sky_top: NDArray[np.float32],
    sky_horizon: NDArray[np.float32],
    haze_color: NDArray[np.float32],
    cloud_color: NDArray[np.float32],
    image: NDArray[np.uint8],
) -> None:
    width = col_azimuth.shape[0]
    rad_per_row = (top_rad - bottom_rad) / height
    sun_az = math.radians(SUN_AZIMUTH_DEG)
    sun_el = math.radians(SUN_ELEVATION_DEG)
    sun_x = math.sin(sun_az) * math.cos(sun_el)
    sun_y = math.cos(sun_az) * math.cos(sun_el)
    sun_z = math.sin(sun_el)

    for col in prange(width):  # type: ignore[no-untyped-call, attr-defined]  # numba prange
        azimuth = col_azimuth[col]
        dx = math.sin(azimuth)
        dy = math.cos(azimuth)
        max_ang = bottom_rad
        prev_z = eye_z - EYE_HEIGHT_M
        prev_slope = 0.0
        prev_lateral = 0.0
        prev_visible = True
        prev_field_ix = -(10**9)
        prev_field_iy = -(10**9)
        # Start at the observer's feet: the bottom of a down-tilted frame is ground
        # only metres away and must be finely sampled, not extrapolated as one span.
        r = 4.0
        step = 0.05
        while r < MAX_RENDER_DISTANCE_M:
            x = obs_e + dx * r
            y = obs_n + dy * r
            z = _sample_elev(dem, dem10, has10, x, y)
            lc_bin = _landcover_at(landcover, x, y)
            clearing = (r - CLEARING_RAMP_START_M) / (CLEARING_RAMP_FULL_M - CLEARING_RAMP_START_M)
            if clearing < 0.0:
                clearing = 0.0
            elif clearing > 1.0:
                clearing = 1.0
            z_surf = z
            tree_here = False
            if lc_bin == 1:
                # The clearing ramp thins tree DENSITY, not height: near the observer only
                # scattered full-height trees stand (like a real clearing edge). Scaling
                # height instead would grow a staircase of canopy fronts with distance.
                clump = _cell_noise(x, y, 35.0)
                if clump < clearing:
                    tree_here = True
                    z_surf = z + TREE_HEIGHT_M * (
                        0.55 + 0.9 * _cell_noise(x + 17.0, y + 31.0, 35.0)
                    )
            elif lc_bin == 5:
                z_surf = z + clearing * BUILT_HEIGHT_M * (0.5 + _cell_noise(x, y, 45.0))

            z_eff = z_surf - RENDER_CURVATURE * r * r
            ang = math.atan((z_eff - eye_z) / r)
            if ang > max_ang:
                # Slope drives shading, so it must come from consecutively VISIBLE
                # samples: across an occluded dip, (z - prev_z) spans the hidden hollow
                # and would paint a bright false stripe at every re-emergence line.
                if prev_visible:
                    slope_alpha = 0.5 - 0.42 * math.exp(-r / 350.0)
                    radial_slope = (
                        slope_alpha * (z - prev_z) / step + (1.0 - slope_alpha) * prev_slope
                    )
                else:
                    radial_slope = prev_slope
                prev_slope = radial_slope
                prev_visible = True
                # Lateral slope completes the true surface normal for sun modelling.
                lat_w = 15.0 if step < 15.0 else step
                z_left = _sample_elev(dem, dem10, has10, x - dy * lat_w, y + dx * lat_w)
                z_right = _sample_elev(dem, dem10, has10, x + dy * lat_w, y - dx * lat_w)
                lateral_raw = (z_right - z_left) / (2.0 * lat_w)
                lateral = 0.5 * lateral_raw + 0.5 * prev_lateral
                prev_lateral = lateral
                # Colour is sampled at a world-anchored jittered position near the
                # observer, so 50 m cell borders dither into organic transitions
                # instead of crisp terrace lines across the foreground.
                jitter = 30.0 * math.exp(-r / 450.0)
                if jitter > 1.5:
                    xc = x + (_smooth_noise(x * 0.11 + 7.7, y * 0.11) - 0.5) * 2.0 * jitter
                    yc = y + (_smooth_noise(x * 0.11, y * 0.11 + 3.3) - 0.5) * 2.0 * jitter
                    lc_color = _landcover_at(landcover, xc, yc)
                else:
                    xc = x
                    yc = y
                    lc_color = lc_bin
                if tree_here:
                    # A standing tree is foliage regardless of what colour jitter found.
                    lc_color = 1
                # True-normal diffuse sun: gradient from radial + lateral slopes.
                grad_x = radial_slope * dx - lateral * dy
                grad_y = radial_slope * dy + lateral * dx
                inv_len = 1.0 / math.sqrt(grad_x * grad_x + grad_y * grad_y + 1.0)
                n_dot_sun = (-grad_x * sun_x - grad_y * sun_y + sun_z) * inv_len
                if n_dot_sun < 0.0:
                    n_dot_sun = 0.0
                shade = SHADE_AMBIENT + SHADE_DIFFUSE * n_dot_sun
                if lc_color == 8:
                    shade = 1.02
                # Cloud shadows dapple the ground the way every English photo shows.
                cloud_shadow = 0.65 * _smooth_noise(
                    x / CLOUD_SHADOW_SCALE_M + cloud_seed_u, y / CLOUD_SHADOW_SCALE_M + cloud_seed_v
                ) + 0.35 * _smooth_noise(
                    x / CLOUD_SHADOW_SCALE_M * 2.6 + 41.7, y / CLOUD_SHADOW_SCALE_M * 2.6 + 11.3
                )
                shadow_amount = (cloud_shadow - 0.52) * 2.8
                if shadow_amount > 0.0:
                    if shadow_amount > 1.0:
                        shadow_amount = 1.0
                    shade *= 1.0 - CLOUD_SHADOW_STRENGTH * shadow_amount
                if shade < 0.45:
                    shade = 0.45
                elif shade > 1.22:
                    shade = 1.22
                field = 0.88 + 0.24 * _cell_noise(xc, yc, FIELD_CELL_M)
                # The field you stand in is one field: patchwork contrast (and its
                # vertical column banding) only makes sense beyond a few hundred metres.
                field = 1.0 + (field - 1.0) * min(1.0, r / 800.0)
                grain = 0.94 + 0.12 * _cell_noise(xc, yc, 18.0)
                if lc_color == 1:
                    grain = 0.78 + 0.44 * _cell_noise(xc, yc, 22.0)
                elif r < 1200.0:
                    # Fine grass/crop mottling so the near foreground is not a smooth wall.
                    near_amp = 0.14 * math.exp(-r / 500.0)
                    grain *= 1.0 + near_amp * (2.0 * _smooth_noise(x * 0.22, y * 0.22) - 1.0)
                shade *= field * grain

                # Hedgerow hint: darken the strip where the field cell changes.
                field_ix = int(x / FIELD_CELL_M)
                field_iy = int(y / FIELD_CELL_M)
                if (
                    (field_ix != prev_field_ix or field_iy != prev_field_iy)
                    and prev_field_ix != -(10**9)
                    and (lc_color == 3 or lc_color == 4)
                ):
                    shade *= HEDGE_SHADE
                prev_field_ix = field_ix
                prev_field_iy = field_iy

                haze = 1.0 - math.exp(-r / HAZE_DISTANCE_M)
                shade = 1.0 + (shade - 1.0) * (1.0 - 0.75 * haze)

                red = base_colors[lc_color, 0]
                green = base_colors[lc_color, 1]
                blue = base_colors[lc_color, 2]
                if lc_color == 1 and lc_bin == 1 and not tree_here:
                    # Ground between the scattered near trees reads as rough scrub.
                    red = red + (base_colors[3, 0] - red) * 0.45
                    green = green + (base_colors[3, 1] - green) * 0.45
                    blue = blue + (base_colors[3, 2] - blue) * 0.45
                if lc_color == 3 and z > MOOR_GRASS_START_M:
                    moor = (z - MOOR_GRASS_START_M) / (MOOR_GRASS_FULL_M - MOOR_GRASS_START_M)
                    if moor > 1.0:
                        moor = 1.0
                    red = red + (152.0 - red) * moor
                    green = green + (142.0 - green) * moor
                    blue = blue + (92.0 - blue) * moor
                if lc_color == 8:
                    # Open water mirrors the sky's brightness.
                    red = red + (sky_horizon[0] - red) * 0.45
                    green = green + (sky_horizon[1] - green) * 0.45
                    blue = blue + (sky_horizon[2] - blue) * 0.45
                red *= shade
                green *= shade
                blue *= shade
                red = red + (haze_color[0] - red) * haze
                green = green + (haze_color[1] - green) * haze
                blue = blue + (haze_color[2] - blue) * haze

                if projection == PROJECTION_PANORAMA:
                    row_hi = int((top_rad - max_ang) / rad_per_row)
                    row_lo = int((top_rad - ang) / rad_per_row)
                else:
                    row_hi = int(cy_px - f_v_px * math.tan(max_ang - pitch_rad))
                    row_lo = int(cy_px - f_v_px * math.tan(ang - pitch_rad))
                if row_lo < 0:
                    row_lo = 0
                if row_hi > height:
                    row_hi = height
                near_boost = 0.12 * math.exp(-r / 400.0)
                texture_amp = (1.0 - haze) * ((0.16 + near_boost) if lc_color != 1 else 0.30)
                span_rows = row_hi - row_lo
                for row in range(row_lo, row_hi):
                    hsh = ((col * 7919) ^ (row * 104729)) & 0x7FFFFFFF
                    hsh = (hsh ^ (hsh >> 11)) * 2654435761
                    noise = ((hsh >> 8) & 0xFFFF) / 65535.0 - 0.5
                    tex = 1.0 + texture_amp * noise
                    if lc_color == 1 and span_rows > 3:
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
            else:
                prev_visible = False

            prev_z = z
            r += step
            # Two resolution limits: azimuthal (step ~ r * 0.7 deg) far out, and VERTICAL
            # angular resolution near the observer, where a span of dtheta covers
            # dr = dtheta * r^2 / eye-height — without this, each near sample paints a
            # tall flat terrace band across the foreground.
            step = r * 0.012
            vertical_step = 0.0008 * r * r
            if vertical_step < step:
                step = vertical_step
            if step < 0.05:
                step = 0.05
            elif step > 400.0:
                step = 400.0

        # Sky with procedural clouds above the final skyline.
        if projection == PROJECTION_PANORAMA:
            horizon_row = int((top_rad - max_ang) / rad_per_row)
        else:
            horizon_row = int(cy_px - f_v_px * math.tan(max_ang - pitch_rad))
        if horizon_row > height:
            horizon_row = height
        az_deg = math.degrees(azimuth)
        for row in range(0, horizon_row):
            if projection == PROJECTION_PANORAMA:
                elev = top_rad - row * rad_per_row
            else:
                elev = math.atan((cy_px - row) / f_v_px) + pitch_rad
            elev_deg = math.degrees(elev)
            elev_norm = elev / top_rad if top_rad > 0.0 else 0.0
            if elev_norm < 0.0:
                elev_norm = 0.0
            elif elev_norm > 1.0:
                elev_norm = 1.0
            f = math.exp(-2.2 * elev_norm)
            red = sky_top[0] + (sky_horizon[0] - sky_top[0]) * f
            green = sky_top[1] + (sky_horizon[1] - sky_top[1]) * f
            blue = sky_top[2] + (sky_horizon[2] - sky_top[2]) * f

            cloud = _cloud_density(az_deg, elev_deg, cloud_seed_u, cloud_seed_v)
            alpha = (cloud - 0.56) * 3.2
            if alpha > 0.0:
                if alpha > 1.0:
                    alpha = 1.0
                # Clouds thin towards the zenith and melt into the horizon haze.
                alpha *= 0.30 + 0.45 * math.exp(-2.5 * elev_norm)
                red = red + (cloud_color[0] - red) * alpha
                green = green + (cloud_color[1] - green) * alpha
                blue = blue + (cloud_color[2] - blue) * alpha
            image[row, col, 0] = np.uint8(red)
            image[row, col, 1] = np.uint8(green)
            image[row, col, 2] = np.uint8(blue)


CREST_SNAP_MIN_GAIN_M: float = 2.5


def _crest_snap(
    dem: NDArray[np.float32],
    dem10: NDArray[np.int16],
    has10: bool,
    easting: float,
    northing: float,
) -> tuple[float, float, float]:
    """Move the camera to the local crest only when clearly below it.

    Rescues cone summits sampled off-centre (Glastonbury Tor) without retreating
    from slope tops and escarpment brows, where the cell behind is always slightly
    higher and snapping would add 50 m of foreground field to every render.
    """
    origin_ground = _sample_elev(dem, dem10, has10, easting, northing)
    best_e, best_n, ground = easting, northing, origin_ground
    for de in (-50.0, 0.0, 50.0):
        for dn in (-50.0, 0.0, 50.0):
            z_here = _sample_elev(dem, dem10, has10, easting + de, northing + dn)
            if z_here > ground:
                ground = z_here
                best_e, best_n = easting + de, northing + dn
    if ground - origin_ground < CREST_SNAP_MIN_GAIN_M:
        return easting, northing, float(origin_ground)
    return best_e, best_n, float(ground)


def _run_kernel(
    dem: NDArray[np.float32],
    dem10: NDArray[np.int16],
    has10: bool,
    landcover: NDArray[np.uint8],
    easting: float,
    northing: float,
    col_azimuth: NDArray[np.float64],
    projection: int,
    height: int,
    top_rad: float,
    bottom_rad: float,
    f_v_px: float,
    cy_px: float,
    pitch_rad: float,
) -> NDArray[np.uint8]:
    easting, northing, ground = _crest_snap(dem, dem10, has10, easting, northing)
    eye_z = ground + EYE_HEIGHT_M + EYE_BOOST_M
    image = np.zeros((height, len(col_azimuth), 3), dtype=np.uint8)
    _render_kernel(
        dem,
        dem10,
        has10,
        landcover,
        easting,
        northing,
        eye_z,
        col_azimuth,
        projection,
        height,
        top_rad,
        bottom_rad,
        f_v_px,
        cy_px,
        pitch_rad,
        easting * 0.00173,
        northing * 0.00119,
        _BASE_COLORS,
        _SKY_TOP,
        _SKY_HORIZON,
        _HAZE_COLOR,
        _CLOUD_COLOR,
        image,
    )
    return image


_DEM10_STUB: NDArray[np.int16] = np.full((1, 1), -32768, dtype=np.int16)


def _dem10_or_stub(dem10: NDArray[np.int16] | None) -> tuple[NDArray[np.int16], bool]:
    if dem10 is None:
        return _DEM10_STUB, False
    return dem10, True


def _downsample(image: NDArray[np.uint8], width: int, height: int) -> Image.Image:
    result = Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)
    # Photographs are more saturated and contrasty than raw painted colour.
    result = ImageEnhance.Color(result).enhance(1.16)
    result = ImageEnhance.Contrast(result).enhance(1.07)
    return result


def render_panorama(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    easting: float,
    northing: float,
    dem10: NDArray[np.int16] | None = None,
) -> Image.Image:
    scale = SUPERSAMPLE
    width = PANORAMA_WIDTH * scale
    height = PANORAMA_HEIGHT * scale
    col_azimuth = 2.0 * math.pi * (np.arange(width, dtype=np.float64) + 0.5) / width
    dem10_arr, has10 = _dem10_or_stub(dem10)
    image = _run_kernel(
        dem,
        dem10_arr,
        has10,
        landcover,
        easting,
        northing,
        col_azimuth,
        PROJECTION_PANORAMA,
        height,
        math.radians(PANO_TOP_DEG),
        math.radians(PANO_BOTTOM_DEG),
        0.0,
        0.0,
        0.0,
    )
    return _downsample(image, PANORAMA_WIDTH, PANORAMA_HEIGHT)


def render_view(
    dem: NDArray[np.float32],
    landcover: NDArray[np.uint8],
    easting: float,
    northing: float,
    center_azimuth_deg: float,
    hfov_deg: float = VIEW_HFOV_DEG,
    width: int = VIEW_WIDTH,
    height: int = VIEW_HEIGHT,
    pitch_deg: float = VIEW_PITCH_DEG,
    horizon_fraction: float = VIEW_HORIZON_FRACTION,
    dem10: NDArray[np.int16] | None = None,
) -> Image.Image:
    """Rectilinear 'camera' view toward one bearing.

    hfov_deg/width/height model the lens and frame; horizon_fraction places the level
    horizon in the frame (0 = top) when pitch_deg is 0, matching a reference photograph.
    """
    scale = SUPERSAMPLE
    render_w = width * scale
    render_h = height * scale
    f_h = (render_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    center = math.radians(center_azimuth_deg)
    offsets = (np.arange(render_w, dtype=np.float64) + 0.5) - render_w / 2.0
    col_azimuth = center + np.arctan(offsets / f_h)

    pitch = math.radians(pitch_deg)
    cy = render_h * horizon_fraction
    f_v = f_h  # square pixels
    top_rad = math.atan(cy / f_v) + pitch
    bottom_rad = math.atan((cy - render_h) / f_v) + pitch
    dem10_arr, has10 = _dem10_or_stub(dem10)
    image = _run_kernel(
        dem,
        dem10_arr,
        has10,
        landcover,
        easting,
        northing,
        col_azimuth,
        PROJECTION_RECTILINEAR,
        render_h,
        top_rad,
        bottom_rad,
        f_v,
        cy,
        pitch,
    )
    return _downsample(image, width, height)


def best_view_directions(
    d_far_veg_m: NDArray[np.float32],
    window_deg: float = 70.0,
    max_directions: int = 2,
    second_quality_fraction: float = 0.55,
    min_separation_deg: float = 75.0,
) -> list[ViewDirection]:
    """Pick the 1-2 bearings whose window contains the most far-reaching visible ground."""
    n = len(d_far_veg_m)
    capped = np.minimum(d_far_veg_m.astype(np.float64), 40000.0) / 1000.0
    half = int(round(window_deg / 2.0 / 360.0 * n))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    padded = np.concatenate([capped[-half:], capped, capped[:half]])
    quality = np.convolve(padded, kernel, mode="valid")
    assert len(quality) == n, "circular window quality has one value per bearing"

    directions: list[ViewDirection] = []
    remaining = quality.copy()
    min_sep = int(round(min_separation_deg / 360.0 * n))
    for _ in range(max_directions):
        index = int(np.argmax(remaining))
        value = float(remaining[index])
        if value <= 0.5:
            break
        if len(directions) > 0 and value < directions[0].quality * second_quality_fraction:
            break
        directions.append(ViewDirection(azimuth_deg=index / n * 360.0, quality=value))
        for offset in range(-min_sep, min_sep + 1):
            remaining[(index + offset) % n] = -1.0
    return directions


def compass_label(azimuth_deg: float) -> str:
    names = [
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    ]
    index = int((azimuth_deg + 22.5) // 45.0) % 8
    return names[index]
