"""Next-generation renderer prototype: measured near field, photographic sky.

Inside R_NEAR_MAX the ray-marcher walks the REAL surface — the EA 1 m first-return
DSM — so tree crowns, hedgerows, walls' vegetation and buildings appear where they
actually stand, at their measured heights. Materials come from data, not noise:
DSM−DTM (nDSM) separates canopy from ground, OSM footprints tag buildings, the
Sentinel-2 mosaic supplies ground albedo, and the 10 m land-cover path flags still
draw worn paths. Direct sun is a real light: a per-sample occlusion march over the
DSM casts true shadows from trees, hedges and buildings. The sky is a real
photograph (CC0 equirectangular HDRI preview) sampled per ray, rotated so its sun
matches the scene sun. Beyond R_NEAR_MAX the existing calibrated far field takes
over (10 m terrain, procedural canopy silhouettes, satellite albedo, haze).

Strict camera: no openness search, no crest snap, eye at 1.7 m — the render is
made from exactly the coordinate it claims.
"""

import math

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from PIL import Image

from england_pbv.constants import EARTH_RADIUS_M, REFRACTION_K
from england_pbv.viewshed.horizon import _bilinear
from england_pbv.viewshed.render import (
    DEM10_CELL,
    DEM10_INVALID,
    DEM10_X0,
    DEM10_Y0,
    _cell_noise,
    _landcover_at2,
    _landcover_code10,
    _sample_elev,
    _smooth_noise,
)

RENDER_CURVATURE: float = (1.0 - REFRACTION_K) / (2.0 * EARTH_RADIUS_M)
PROJECTION_PANORAMA: int = 0
PROJECTION_RECTILINEAR: int = 1
PANORAMA_TOP_DEG: float = 8.0
PANORAMA_BOTTOM_DEG: float = -12.0
R_NEAR_MAX: float = 1800.0
EYE_M: float = 1.7
MAX_DISTANCE_M: float = 60000.0
# Aerial perspective. Measured against the calibration photos, distant land was
# coming out far too dark and too saturated: real haze both washes chroma out and
# lifts distant terrain toward sky brightness. Per-channel extinction makes the
# distance shift BLUE (short wavelengths scatter most), which is the cue that
# separates 5 km from 25 km.
HAZE_DISTANCE_M: float = 9500.0
HAZE_R_SCALE: float = 1.30
HAZE_B_SCALE: float = 0.78
HAZE_DESATURATION: float = 0.62
SUN_AZIMUTH_DEG: float = 225.0
SUN_ELEVATION_DEG: float = 38.0
SHADOW_MARCH_MAX_M: float = 240.0
SHADOW_MARCH_STEP_M: float = 3.0
SHADOW_MAX_R_M: float = 900.0
SUPERSAMPLE: int = 2

_TREE_BASE: tuple[float, float, float] = (86.0, 106.0, 56.0)
_HEDGE_BASE: tuple[float, float, float] = (66.0, 86.0, 44.0)
_GRASS_BASE: tuple[float, float, float] = (148.0, 154.0, 96.0)

# Physical-ish lighting in LINEAR light, filmic tonemap at the end: photographs
# have highlight rolloff, renders that composite in 8-bit sRGB do not.
SUN_RGB: tuple[float, float, float] = (1.0, 0.95, 0.86)
SUN_INTENSITY: float = 2.15
AMBIENT_INTENSITY: float = 0.62
EXPOSURE: float = 0.54
# Sentinel-2 true colour is genuinely dark (vegetation reflects ~5-10% in visible);
# lift it into a plausible albedo range before it becomes surface colour.
SAT_ALBEDO_GAIN: float = 1.72
SAT_CHROMA: float = 0.78
# Ground detail octaves (wavelength m, amplitude): included only while the
# wavelength stays above ~2 pixel footprints, so texture never aliases.
DETAIL_WL: tuple[float, float, float, float] = (9.0, 2.8, 0.9, 0.3)
DETAIL_AMP: tuple[float, float, float, float] = (0.15, 0.13, 0.10, 0.07)
BUMP_MAX_R_M: float = 500.0
# Real turf is a mottle of green and dead/dry blades at ~0.5-4 m scale; a single
# flat green is the clearest "painted" tell after geometry is fixed.
DRY_GRASS: tuple[float, float, float] = (186.0, 168.0, 112.0)
# Photo-texture detail (CC0, england_pbv.data.ground_textures). One tile covers
# TEX_WORLD_M; mip level is chosen from the pixel's ground footprint, so the
# pattern band-limits itself to nothing at distance instead of aliasing.
TEX_SIZE: int = 512
TEX_LEVELS: int = 6
TEX_WORLD_M: float = 6.0
TEX_STRENGTH: float = 0.85
TEX_MAX_R_M: float = 1400.0
# Foliage is not a solid surface. A first-return DSM is a canopy blanket, so rays
# either hit it or miss it and crowns render as opaque rubber blobs. Perforating
# the canopy top with foliage-scale gaps lets rays through between leaf clumps,
# which is what gives a real tree its ragged, sky-showing silhouette. Band-limited:
# the gaps fade out once a pixel footprint is wider than the gaps themselves.
CANOPY_GAP_DEPTH_M: float = 5.0
CANOPY_GAP_FOOT_M: float = 0.55
CANOPY_CROWN_FOOT_M: float = 2.6
# You cannot stand inside a tree. Without a 1 m surface to say otherwise, land
# cover alone will put a 15 m crown a few metres from the eye, and that single
# sample then paints the whole frame. Thin procedural canopy to nothing within
# CLEARING_START_M, full height by CLEARING_FULL_M.
CLEARING_START_M: float = 130.0
CLEARING_FULL_M: float = 280.0
MICRO_RELIEF_R_M: float = 260.0


@njit(cache=True, inline="always")
def _bilin_nan(grid: NDArray[np.float32], fx: float, fy: float) -> float:
    """Bilinear sample returning NaN when outside or any corner is NaN."""
    h, w = grid.shape
    if fx < 0.0 or fy < 0.0 or fx >= w - 1.0 or fy >= h - 1.0:
        return np.nan
    c0 = int(fx)
    r0 = int(fy)
    a = grid[r0, c0]
    b = grid[r0, c0 + 1]
    c = grid[r0 + 1, c0]
    d = grid[r0 + 1, c0 + 1]
    if math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d):
        return np.nan
    tx = fx - c0
    ty = fy - r0
    value = a * (1.0 - tx) * (1.0 - ty) + b * tx * (1.0 - ty) + c * (1.0 - tx) * ty + d * tx * ty
    return float(value)


@njit(cache=True, inline="always")
def _sun_shadow(
    dsm1: NDArray[np.float32],
    fx: float,
    fy: float,
    z_start: float,
    sun_dx: float,
    sun_dy: float,
    sun_tan: float,
) -> float:
    """Soft sun visibility 0..1 — the sun is a 0.53 deg disc, so penumbra widens
    with occluder distance; a binary test reads as a hard CGI shadow."""
    worst = -1.0
    step = SHADOW_MARCH_STEP_M
    d = step
    while d <= SHADOW_MARCH_MAX_M:
        z_ray = z_start + d * sun_tan
        z_here = _bilin_nan(dsm1, fx + sun_dx * d, fy + sun_dy * d)
        if not math.isnan(z_here):
            excess = (z_here - z_ray) / d
            if excess > worst:
                worst = excess
        d += step
        step *= 1.12  # coarser far out; near occluders matter most
    if worst <= -0.004:
        return 1.0
    if worst >= 0.010:
        return 0.0
    t = (worst + 0.004) / 0.014
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


@njit(cache=True, inline="always")
def _srgb_to_lin(c: float) -> float:
    v = c / 255.0
    if v <= 0.04045:
        return v / 12.92
    return float(((v + 0.055) / 1.055) ** 2.4)


@njit(cache=True, inline="always")
def _lin_to_srgb(v: float) -> float:
    c = v * 12.92 if v <= 0.0031308 else 1.055 * (v ** (1.0 / 2.4)) - 0.055
    return c * 255.0


@njit(cache=True, inline="always")
def _tonemap(v: float) -> float:
    """Narkowicz ACES filmic curve — highlight rolloff instead of hard clipping."""
    x = v * EXPOSURE
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    if y < 0.0:
        return 0.0
    if y > 1.0:
        return 1.0
    return y


@njit(cache=True, inline="always")
def _tex_bilinear(
    tex: NDArray[np.float32], idx: int, level: int, u: float, v: float
) -> tuple[float, float, float]:
    fu = (u - math.floor(u)) * TEX_SIZE
    fv = (v - math.floor(v)) * TEX_SIZE
    x0 = int(fu) % TEX_SIZE
    y0 = int(fv) % TEX_SIZE
    x1 = (x0 + 1) % TEX_SIZE
    y1 = (y0 + 1) % TEX_SIZE
    tx = fu - int(fu)
    ty = fv - int(fv)
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0
    for k in range(3):
        a = tex[idx, level, y0, x0, k]
        b = tex[idx, level, y0, x1, k]
        c = tex[idx, level, y1, x0, k]
        d = tex[idx, level, y1, x1, k]
        val = a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty
        if k == 0:
            out_r = val
        elif k == 1:
            out_g = val
        else:
            out_b = val
    return out_r, out_g, out_b


@njit(cache=True, inline="always")
def _ground_texture(
    tex: NDArray[np.float32],
    has_tex: bool,
    idx: int,
    x: float,
    y: float,
    footprint_m: float,
) -> tuple[float, float, float]:
    """Trilinear photo-texture pattern (mean 1.0) for a ground sample."""
    if not has_tex:
        return 1.0, 1.0, 1.0
    texels_per_px = footprint_m * TEX_SIZE / TEX_WORLD_M
    if texels_per_px < 1.0:
        texels_per_px = 1.0
    lod = math.log(texels_per_px) / math.log(2.0)
    if lod >= TEX_LEVELS - 1:
        return 1.0, 1.0, 1.0  # fully filtered away: no pattern, no aliasing
    l0 = int(lod)
    frac = lod - l0
    # Break the 6 m tile repeat with a CONTINUOUS low-frequency offset: a
    # per-cell hash offset seams at cell borders, and at grazing incidence those
    # borders project to vertical stripes across the foreground.
    ox = 4.0 * _smooth_noise(x / 240.0, y / 240.0)
    oy = 4.0 * _smooth_noise(x / 240.0 + 37.0, y / 240.0 + 11.0)
    u = x / TEX_WORLD_M + ox
    v = y / TEX_WORLD_M + oy
    ar, ag, ab = _tex_bilinear(tex, idx, l0, u, v)
    br, bg, bb = _tex_bilinear(tex, idx, min(l0 + 1, TEX_LEVELS - 1), u, v)
    return (
        ar + (br - ar) * frac,
        ag + (bg - ag) * frac,
        ab + (bb - ab) * frac,
    )


@njit(cache=True, inline="always")
def _far_canopy_h(x: float, y: float) -> float:
    """Continuous multi-scale canopy height used beyond the 1 m window."""
    return (
        15.0
        * (0.55 + 0.50 * _smooth_noise(x / 17.0, y / 17.0))
        * (0.72 + 0.34 * _smooth_noise(x / 5.5 + 21.0, y / 5.5))
    )


@njit(cache=True, inline="always")
def _detail_factor(x: float, y: float, footprint_m: float) -> float:
    """Multi-octave ground detail, band-limited to the pixel footprint."""
    factor = 1.0
    for i in range(4):
        wl = DETAIL_WL[i]
        if wl < 2.0 * footprint_m:
            continue
        fade = 1.0
        if wl < 4.0 * footprint_m:
            fade = (wl - 2.0 * footprint_m) / (2.0 * footprint_m)
        factor *= 1.0 + DETAIL_AMP[i] * fade * (2.0 * _smooth_noise(x / wl, y / wl) - 1.0)
    return factor


@njit(cache=True, inline="always")
def _sat_bilinear(
    sat: NDArray[np.uint8], has_sat: bool, x: float, y: float
) -> tuple[float, float, float, bool]:
    """Bilinear satellite colour — nearest-neighbour made 10 m blocks near the eye."""
    if not has_sat:
        return 0.0, 0.0, 0.0, False
    fc = (x - 80000.0) / 10.0 - 0.5
    fr = (y - 4000.0) / 10.0 - 0.5
    h, w = sat.shape[0], sat.shape[1]
    if fc < 0.0 or fr < 0.0 or fc >= w - 1.0 or fr >= h - 1.0:
        return 0.0, 0.0, 0.0, False
    c0 = int(fc)
    r0 = int(fr)
    tx = fc - c0
    ty = fr - r0
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0
    total = 0.0
    for k in range(3):
        a = float(sat[r0, c0, k])
        b = float(sat[r0, c0 + 1, k])
        c = float(sat[r0 + 1, c0, k])
        d = float(sat[r0 + 1, c0 + 1, k])
        v = a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) + c * (1 - tx) * ty + d * tx * ty
        v *= SAT_ALBEDO_GAIN
        if v > 255.0:
            v = 255.0
        if k == 0:
            out_r = v
        elif k == 1:
            out_g = v
        else:
            out_b = v
        total += v
    if total <= 24.0:
        return 0.0, 0.0, 0.0, False
    # Gained Sentinel true colour is more chromatic than real turf; pulling it back
    # toward its own luminance is what stops near ground reading fluorescent.
    luma = 0.299 * out_r + 0.587 * out_g + 0.114 * out_b
    out_r = luma + (out_r - luma) * SAT_CHROMA
    out_g = luma + (out_g - luma) * SAT_CHROMA
    out_b = luma + (out_b - luma) * SAT_CHROMA
    return out_r, out_g, out_b, True


@njit(cache=True, inline="always")
def _sat_albedo(
    sat: NDArray[np.uint8], has_sat: bool, x: float, y: float
) -> tuple[float, float, float, bool]:
    if not has_sat:
        return 0.0, 0.0, 0.0, False
    col = int((x - 80000.0) / 10.0)
    row = int((y - 4000.0) / 10.0)
    if col < 0 or row < 0 or col >= sat.shape[1] or row >= sat.shape[0]:
        return 0.0, 0.0, 0.0, False
    r = float(sat[row, col, 0])
    g = float(sat[row, col, 1])
    b = float(sat[row, col, 2])
    if r + g + b <= 12.0:
        return 0.0, 0.0, 0.0, False
    # Saturation + gain lift, as calibrated for the far field in render.py.
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    r = max(0.0, luma + (r - luma) * 1.35) * 1.22
    g = max(0.0, luma + (g - luma) * 1.35) * 1.22
    b = max(0.0, luma + (b - luma) * 1.35) * 1.22
    return r, g, b, True


@njit(cache=True, parallel=True, fastmath=True)
def _render2_kernel(
    dtm1: NDArray[np.float32],
    dsm1: NDArray[np.float32],
    bld: NDArray[np.uint8],
    win_x0: float,
    win_y0: float,
    dem50: NDArray[np.float32],
    dem10: NDArray[np.int16],
    has10: bool,
    lc50: NDArray[np.uint8],
    lc10: NDArray[np.uint8],
    has10_lc: bool,
    sat: NDArray[np.uint8],
    has_sat: bool,
    tex: NDArray[np.float32],
    has_tex: bool,
    has_window: bool,
    hdri: NDArray[np.float32],
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
    haze_color: NDArray[np.float32],
    ambient_color: NDArray[np.float32],
    image: NDArray[np.uint8],
) -> None:
    width = col_azimuth.shape[0]
    # Ambient is the sky's own colour (from the HDRI), so shadows go blue like a
    # photograph instead of neutral grey.
    amb_r = _srgb_to_lin(float(ambient_color[0]))
    amb_g = _srgb_to_lin(float(ambient_color[1]))
    amb_b = _srgb_to_lin(float(ambient_color[2]))
    haze_lin_r = _srgb_to_lin(float(haze_color[0]))
    haze_lin_g = _srgb_to_lin(float(haze_color[1]))
    haze_lin_b = _srgb_to_lin(float(haze_color[2]))
    rad_per_row = (top_rad - bottom_rad) / height
    # Angular size of one supersampled pixel: drives texture band-limiting.
    px_angle = rad_per_row if projection == PROJECTION_PANORAMA else 1.0 / f_v_px
    sun_az = math.radians(SUN_AZIMUTH_DEG)
    sun_el = math.radians(SUN_ELEVATION_DEG)
    sun_x = math.sin(sun_az) * math.cos(sun_el)
    sun_y = math.cos(sun_az) * math.cos(sun_el)
    sun_z = math.sin(sun_el)
    sun_tan = math.tan(sun_el)
    hs, ws = hdri.shape[0], hdri.shape[1]

    for col in prange(width):  # type: ignore[no-untyped-call, attr-defined]  # numba prange
        azimuth = col_azimuth[col]
        dx = math.sin(azimuth)
        dy = math.cos(azimuth)
        max_ang = bottom_rad
        prev_z = eye_z - EYE_M
        prev_slope = 0.0
        prev_visible = True
        # One march sample paints a whole span of pixel rows. At grazing incidence
        # that span grows and any texture inside it becomes a horizontal smear, so
        # the step adapts to keep spans near one pixel, and texture is attenuated
        # while a span is still tall.
        prev_span_px = 1.0
        r = 2.0
        step = 0.05
        while r < MAX_DISTANCE_M:
            x = obs_e + dx * r
            y = obs_n + dy * r
            near = False
            z_surf = 0.0
            z_ground = 0.0
            if r < R_NEAR_MAX:
                fx = x - win_x0
                fy = y - win_y0
                zs = _bilin_nan(dsm1, fx, fy)
                zg = _bilin_nan(dtm1, fx, fy)
                if not math.isnan(zs) and not math.isnan(zg):
                    near = True
                    z_surf = zs
                    z_ground = zg
                    canopy = zs - zg
                    if canopy > 1.5:
                        foot_here = r / f_v_px
                        # Two gap scales. Leaf-scale gaps (~0.15 m) are only
                        # resolvable very close; crown-scale gaps (~3 m, the spaces
                        # between neighbouring trees) survive to a kilometre and are
                        # what makes a distant wood read as trees, not a green sheet.
                        pen = 0.0
                        if foot_here < CANOPY_GAP_FOOT_M:
                            gap = 0.55 * _smooth_noise(x * 2.6, y * 2.6) + 0.45 * _smooth_noise(
                                x * 7.0 + 13.0, y * 7.0
                            )
                            fine = (gap - 0.38) * 1.9
                            if fine > 1.0:
                                fine = 1.0
                            if fine > 0.0:
                                pen = fine * (1.0 - foot_here / CANOPY_GAP_FOOT_M)
                        if foot_here < CANOPY_CROWN_FOOT_M:
                            crown_gap = _smooth_noise(x * 0.33 + 7.0, y * 0.33)
                            coarse = (crown_gap - 0.46) * 2.4
                            if coarse > 1.0:
                                coarse = 1.0
                            if coarse > 0.0:
                                coarse *= 1.0 - foot_here / CANOPY_CROWN_FOOT_M
                                if coarse > pen:
                                    pen = coarse
                        if pen > 0.0:
                            depth = CANOPY_GAP_DEPTH_M
                            if depth > canopy - 0.5:
                                depth = canopy - 0.5
                            z_surf = zs - pen * depth
            if not near:
                z_ground = _sample_elev(dem50, dem10, has10, x, y)
                z_surf = z_ground
                # Jitter the class lookup: a 10 m raster extruded straight up gives
                # rectangular green slabs with hard vertical sides across every
                # distant plain. Displacing the lookup by a smooth ~7 m offset turns
                # those raster edges into organic wood and hedgerow outlines.
                jx = 7.0 * (_smooth_noise(x * 0.06, y * 0.06) - 0.5)
                jy = 7.0 * (_smooth_noise(x * 0.06 + 31.0, y * 0.06) - 0.5)
                lc_bin_far = _landcover_at2(lc50, lc10, has10_lc, x + jx, y + jy)
                if lc_bin_far == 1:
                    # Continuous multi-scale canopy: a dome per 35 m grid cell put
                    # identical "broccoli" on a visible lattice across every plain.
                    canopy_h = _far_canopy_h(x, y)
                    clearing = (r - CLEARING_START_M) / (CLEARING_FULL_M - CLEARING_START_M)
                    if clearing < 0.0:
                        clearing = 0.0
                    elif clearing > 1.0:
                        clearing = 1.0
                    canopy_h *= clearing
                    # Taper at the wood edge instead of a vertical wall: sample two
                    # neighbours and drop the height where the class runs out.
                    edge = 1.0
                    if _landcover_at2(lc50, lc10, has10_lc, x + jx + 9.0, y + jy) != 1:
                        edge -= 0.32
                    if _landcover_at2(lc50, lc10, has10_lc, x + jx, y + jy + 9.0) != 1:
                        edge -= 0.32
                    z_surf = z_ground + canopy_h * edge
                elif lc_bin_far != 8 and r < MICRO_RELIEF_R_M and not has_window:
                    # A 10 m DEM is glass-smooth, so at grazing incidence the near
                    # ground paints as one stretched colour wall. Real turf is lumpy
                    # at 0.5-3 m; adding that as GEOMETRY (not just a normal) gives
                    # the ray-marcher something to resolve and breaks the smear.
                    relief = 1.0 - r / MICRO_RELIEF_R_M
                    z_surf = z_ground + relief * (
                        0.22 * (_smooth_noise(x * 0.55, y * 0.55) - 0.5)
                        + 0.09 * (_smooth_noise(x * 1.9 + 5.0, y * 1.9) - 0.5)
                    )
                if lc_bin_far == 5:
                    built = (r - CLEARING_START_M) / (CLEARING_FULL_M - CLEARING_START_M)
                    if built < 0.0:
                        built = 0.0
                    elif built > 1.0:
                        built = 1.0
                    z_surf = z_ground + built * 8.0 * (0.5 + _cell_noise(x, y, 45.0))

            z_eff = z_surf - RENDER_CURVATURE * r * r
            ang = math.atan((z_eff - eye_z) / r)
            if ang > max_ang:
                if prev_visible:
                    slope_alpha = 0.5 - 0.42 * math.exp(-r / 350.0)
                    radial_slope = (
                        slope_alpha * (z_surf - prev_z) / step + (1.0 - slope_alpha) * prev_slope
                    )
                else:
                    radial_slope = prev_slope
                prev_slope = radial_slope
                prev_visible = True

                red = 0.0
                green = 0.0
                blue = 0.0
                is_foliage = False
                # Pixel ground footprint: texture and bump are band-limited to it,
                # so detail never aliases into shimmer at distance.
                foot = r * px_angle
                incid = abs(ang - radial_slope) / math.sqrt(radial_slope * radial_slope + 1.0)
                if incid < 0.05:
                    incid = 0.05
                foot /= incid
                sun_vis = 1.0
                amb_vis = 1.0
                if near:
                    fx = x - win_x0
                    fy = y - win_y0
                    ndsm = z_surf - z_ground
                    ze = _bilin_nan(dsm1, fx + 1.0, fy)
                    zw = _bilin_nan(dsm1, fx - 1.0, fy)
                    zn = _bilin_nan(dsm1, fx, fy + 1.0)
                    zs2 = _bilin_nan(dsm1, fx, fy - 1.0)
                    gx = 0.0
                    gy = 0.0
                    if not (math.isnan(ze) or math.isnan(zw)):
                        gx = (ze - zw) * 0.5
                    if not (math.isnan(zn) or math.isnan(zs2)):
                        gy = (zn - zs2) * 0.5
                    is_bld = bld[int(fy), int(fx)] == 1 and ndsm > 1.5
                    sr, sg, sb, sat_ok = _sat_bilinear(sat, has_sat, x, y)
                    if is_bld:
                        tone = _cell_noise(x, y, 14.0)
                        if tone < 0.45:
                            red, green, blue = 96.0, 92.0, 98.0  # slate
                        elif tone < 0.8:
                            red, green, blue = 152.0, 98.0, 78.0  # tile
                        else:
                            red, green, blue = 138.0, 132.0, 124.0  # grey
                    elif ndsm > 2.0:
                        # Canopy colour from the satellite pixel over that crown:
                        # real species/season variation instead of one painted green.
                        if sat_ok:
                            red = 0.82 * sr + 0.18 * _TREE_BASE[0]
                            green = 0.82 * sg + 0.18 * _TREE_BASE[1]
                            blue = 0.82 * sb + 0.18 * _TREE_BASE[2]
                        else:
                            red, green, blue = _TREE_BASE
                        # Crown-scale species/age variation, continuous (no grid).
                        species = _smooth_noise(x / 14.0 + 61.0, y / 14.0) - 0.5
                        red *= 1.0 + 0.26 * species
                        green *= 1.0 + 0.14 * species
                        blue *= 1.0 + 0.08 * species
                        folg = 0.80 + 0.40 * _smooth_noise(x * 0.9, y * 0.9)
                        red *= folg
                        green *= folg
                        blue *= folg
                        is_foliage = True
                        # Crowns are rough: perturb the normal so foliage is not a
                        # smooth painted dome.
                        gx += 0.9 * (_smooth_noise(x * 1.1 + 3.0, y * 1.1) - 0.5)
                        gy += 0.9 * (_smooth_noise(x * 1.1, y * 1.1 + 3.0) - 0.5)
                    elif ndsm > 0.7:
                        if sat_ok:
                            red = 0.75 * sr + 0.25 * _HEDGE_BASE[0]
                            green = 0.75 * sg + 0.25 * _HEDGE_BASE[1]
                            blue = 0.75 * sb + 0.25 * _HEDGE_BASE[2]
                        else:
                            red, green, blue = _HEDGE_BASE
                        gx += 0.7 * (_smooth_noise(x * 1.3 + 7.0, y * 1.3) - 0.5)
                        gy += 0.7 * (_smooth_noise(x * 1.3, y * 1.3 + 7.0) - 0.5)
                    else:
                        # Ground: the satellite knows this field is tawny stubble or
                        # emerald pasture; a single painted green never can.
                        if sat_ok:
                            red = 0.86 * sr + 0.14 * _GRASS_BASE[0]
                            green = 0.86 * sg + 0.14 * _GRASS_BASE[1]
                            blue = 0.86 * sb + 0.14 * _GRASS_BASE[2]
                        else:
                            red, green, blue = _GRASS_BASE
                        detail = _detail_factor(x, y, foot)
                        red *= detail
                        green *= detail
                        blue *= detail
                        # Dry/green mottle, band-limited like the detail octaves.
                        # Band-limited with a CONTINUOUS falloff: a hard `foot < 2`
                        # gate put a visible edge wherever the incidence (and so the
                        # footprint) jumped at a break of slope.
                        mott = 0.55 * _smooth_noise(x / 3.5, y / 3.5) + 0.45 * _smooth_noise(
                            x / 1.1, y / 1.1
                        )
                        dryness = (mott - 0.44) * 1.7
                        if dryness < 0.0:
                            dryness = 0.0
                        elif dryness > 0.30:
                            dryness = 0.30
                        dryness /= 1.0 + (foot / 1.1) * (foot / 1.1)
                        red = red + (DRY_GRASS[0] - red) * dryness
                        green = green + (DRY_GRASS[1] - green) * dryness
                        blue = blue + (DRY_GRASS[2] - blue) * dryness
                        # Photo texture supplies STRUCTURE (blade contrast, clumps,
                        # bare patches); satellite still supplies the colour.
                        if r < TEX_MAX_R_M:
                            tex_idx = 1 if mott > 0.62 else 0
                            tr, tg, tb = _ground_texture(tex, has_tex, tex_idx, x, y, foot)
                            tlum = 0.299 * tr + 0.587 * tg + 0.114 * tb
                            span_fade = 1.0
                            if prev_span_px > 2.0:
                                span_fade = 2.0 / prev_span_px
                            tex_w = TEX_STRENGTH * span_fade
                            mix_r = 1.0 + tex_w * (0.82 * tlum + 0.18 * tr - 1.0)
                            mix_g = 1.0 + tex_w * (0.82 * tlum + 0.18 * tg - 1.0)
                            mix_b = 1.0 + tex_w * (0.82 * tlum + 0.18 * tb - 1.0)
                            red *= mix_r
                            green *= mix_g
                            blue *= mix_b
                        # Micro-relief (tussocks, sheep tracks, molehills) as a bump:
                        # perfectly smooth turf at grazing incidence is the strongest
                        # "CGI dune" tell in the previous round.
                        if r < BUMP_MAX_R_M and foot < 1.2:
                            bump = 0.30 * (1.0 - foot / 1.2)
                            gx += bump * (
                                _smooth_noise(x * 0.62 + 11.0, y * 0.62)
                                - _smooth_noise(x * 0.62 + 9.0, y * 0.62)
                            )
                            gy += bump * (
                                _smooth_noise(x * 0.62, y * 0.62 + 11.0)
                                - _smooth_noise(x * 0.62, y * 0.62 + 9.0)
                            )
                        code10 = _landcover_code10(lc10, has10_lc, x, y)
                        if code10 >= 0 and code10 % 10 == 1:
                            worn = 0.5
                            red = red + (196.0 - red) * worn
                            green = green + (184.0 - green) * worn
                            blue = blue + (156.0 - blue) * worn
                            if r < TEX_MAX_R_M:
                                pr, pg, pb = _ground_texture(tex, has_tex, 2, x, y, foot)
                                red *= 0.55 + 0.45 * pr
                                green *= 0.55 + 0.45 * pg
                                blue *= 0.55 + 0.45 * pb
                        if code10 >= 0 and code10 // 10 == 8:
                            red, green, blue = 120.0, 150.0, 175.0
                    inv_len = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
                    n_dot = (-gx * sun_x - gy * sun_y + sun_z) * inv_len
                    if n_dot < 0.0:
                        n_dot = 0.0
                    # Sky visibility: hollows and hedge bottoms see less sky than open
                    # ground. Constant ambient made every crease glow.
                    amb_vis = 0.45 + 0.55 * inv_len
                    if r < SHADOW_MAX_R_M:
                        zo = 0.0
                        for k in range(4):
                            ox = 6.0 if k == 0 else (-6.0 if k == 1 else 0.0)
                            oy = 6.0 if k == 2 else (-6.0 if k == 3 else 0.0)
                            zn2 = _bilin_nan(dsm1, fx + ox, fy + oy)
                            if not math.isnan(zn2) and zn2 > z_surf + 1.0:
                                zo += 0.14
                        amb_vis -= zo
                        if amb_vis < 0.25:
                            amb_vis = 0.25
                        sun_vis = _sun_shadow(dsm1, fx, fy, z_surf + 0.35, sun_x, sun_y, sun_tan)
                    n_dot_final = n_dot
                else:
                    lat_w = 15.0
                    zl = _sample_elev(dem50, dem10, has10, x - dy * lat_w, y + dx * lat_w)
                    zr = _sample_elev(dem50, dem10, has10, x + dy * lat_w, y - dx * lat_w)
                    lateral = (zr - zl) / (2.0 * lat_w)
                    gx = radial_slope * dx - lateral * dy
                    gy = radial_slope * dy + lateral * dx
                    tree_top = z_surf > z_ground + 2.0
                    if tree_top:
                        # Beyond ~700 m crown STRUCTURE is sub-pixel, but a wood still
                        # needs a lit side and a shaded side. Normal-mapping the canopy
                        # height gives that for nothing: no geometry, just relief
                        # shading, which is what stops distant woodland reading as
                        # flat green paint.
                        d = 4.0
                        cgx = (_far_canopy_h(x + d, y) - _far_canopy_h(x - d, y)) / (2.0 * d)
                        cgy = (_far_canopy_h(x, y + d) - _far_canopy_h(x, y - d)) / (2.0 * d)
                        bump = 1.0
                        if foot > 6.0:
                            bump = 6.0 / foot  # band-limit: fade as crowns go sub-pixel
                        gx += cgx * bump
                        gy += cgy * bump
                    inv_len = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
                    n_dot = (-gx * sun_x - gy * sun_y + sun_z) * inv_len
                    if n_dot < 0.0:
                        n_dot = 0.0
                    # De-shade with distance: the satellite already contains shadows
                    # from ITS sun angle, so re-lighting it with ours double-shades
                    # everything far away and makes distant land look muddy.
                    deshade = r / 14000.0
                    if deshade > 0.5:
                        deshade = 0.5
                    n_dot_final = n_dot * (1.0 - deshade) + 0.62 * deshade
                    amb_vis = 0.55 + 0.45 * inv_len
                    sr, sg, sb, sat_ok = _sat_bilinear(sat, has_sat, x, y)
                    if tree_top:
                        # The satellite pixel over a wood already encodes species,
                        # season and canopy density: use it, barely tinted.
                        if sat_ok:
                            red = 0.92 * sr + 0.08 * _TREE_BASE[0]
                            green = 0.92 * sg + 0.08 * _TREE_BASE[1]
                            blue = 0.92 * sb + 0.08 * _TREE_BASE[2]
                        else:
                            red, green, blue = _TREE_BASE
                        is_foliage = True
                    elif sat_ok:
                        red, green, blue = sr, sg, sb
                        detail = _detail_factor(x, y, foot)
                        red *= detail
                        green *= detail
                        blue *= detail
                        # Without a 1 m window (the national case) this branch also
                        # paints the near ground, so it needs the same photographic
                        # treatment: dry/green mottle, photo texture, micro-relief.
                        if foot < 2.5:
                            mott = 0.55 * _smooth_noise(x / 3.5, y / 3.5) + 0.45 * _smooth_noise(
                                x / 1.1, y / 1.1
                            )
                            dryness = (mott - 0.44) * 1.7
                            if dryness < 0.0:
                                dryness = 0.0
                            elif dryness > 0.30:
                                dryness = 0.30
                            dryness /= 1.0 + (foot / 1.1) * (foot / 1.1)
                            red = red + (DRY_GRASS[0] - red) * dryness
                            green = green + (DRY_GRASS[1] - green) * dryness
                            blue = blue + (DRY_GRASS[2] - blue) * dryness
                            tr, tg, tb = _ground_texture(
                                tex, has_tex, 1 if mott > 0.62 else 0, x, y, foot
                            )
                            tlum = 0.299 * tr + 0.587 * tg + 0.114 * tb
                            span_fade = 1.0
                            if prev_span_px > 2.0:
                                span_fade = 2.0 / prev_span_px
                            tw = TEX_STRENGTH * span_fade
                            red *= 1.0 + tw * (0.82 * tlum + 0.18 * tr - 1.0)
                            green *= 1.0 + tw * (0.82 * tlum + 0.18 * tg - 1.0)
                            blue *= 1.0 + tw * (0.82 * tlum + 0.18 * tb - 1.0)
                            if foot < 1.2:
                                bmp = 0.30 * (1.0 - foot / 1.2)
                                gx += bmp * (
                                    _smooth_noise(x * 0.62 + 11.0, y * 0.62)
                                    - _smooth_noise(x * 0.62 + 9.0, y * 0.62)
                                )
                                gy += bmp * (
                                    _smooth_noise(x * 0.62, y * 0.62 + 11.0)
                                    - _smooth_noise(x * 0.62, y * 0.62 + 9.0)
                                )
                                inv_len = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
                                n_dot = (-gx * sun_x - gy * sun_y + sun_z) * inv_len
                                if n_dot < 0.0:
                                    n_dot = 0.0
                                n_dot_final = n_dot * (1.0 - deshade) + 0.62 * deshade
                        code10 = _landcover_code10(lc10, has10_lc, x, y)
                        if code10 >= 0 and code10 % 10 == 1 and r < 2500.0:
                            worn = 0.45
                            red = red + (196.0 - red) * worn
                            green = green + (184.0 - green) * worn
                            blue = blue + (156.0 - blue) * worn
                    else:
                        red, green, blue = _GRASS_BASE

                # --- lighting in LINEAR light -----------------------------------
                alb_r = _srgb_to_lin(red)
                alb_g = _srgb_to_lin(green)
                alb_b = _srgb_to_lin(blue)
                sun_term = SUN_INTENSITY * n_dot_final * sun_vis
                if is_foliage:
                    # Leaves transmit light: crowns between the eye and the sun glow
                    # instead of going flat dark. Without this every wood reads as a
                    # matte cut-out.
                    toward_sun = dx * sun_x + dy * sun_y
                    if toward_sun > 0.0:
                        sun_term += (
                            SUN_INTENSITY * 0.42 * toward_sun * (1.0 - n_dot_final) * sun_vis
                        )
                amb_term = AMBIENT_INTENSITY * amb_vis
                lin_r = alb_r * (SUN_RGB[0] * sun_term + amb_r * amb_term)
                lin_g = alb_g * (SUN_RGB[1] * sun_term + amb_g * amb_term)
                lin_b = alb_b * (SUN_RGB[2] * sun_term + amb_b * amb_term)
                haze = 1.0 - math.exp(-r / HAZE_DISTANCE_M)
                # Scattering washes chroma before it washes brightness.
                lum = 0.299 * lin_r + 0.587 * lin_g + 0.114 * lin_b
                desat = haze * HAZE_DESATURATION
                lin_r = lum + (lin_r - lum) * (1.0 - desat)
                lin_g = lum + (lin_g - lum) * (1.0 - desat)
                lin_b = lum + (lin_b - lum) * (1.0 - desat)
                haze_r = 1.0 - math.exp(-r / (HAZE_DISTANCE_M * HAZE_R_SCALE))
                haze_b = 1.0 - math.exp(-r / (HAZE_DISTANCE_M * HAZE_B_SCALE))
                lin_r = lin_r + (haze_lin_r - lin_r) * haze_r
                lin_g = lin_g + (haze_lin_g - lin_g) * haze
                lin_b = lin_b + (haze_lin_b - lin_b) * haze_b
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
                span_now = row_hi - row_lo
                prev_span_px = float(span_now) if span_now > 0 else 1.0
                out_r = _lin_to_srgb(_tonemap(lin_r))
                out_g = _lin_to_srgb(_tonemap(lin_g))
                out_b = _lin_to_srgb(_tonemap(lin_b))
                for row in range(row_lo, row_hi):
                    hsh = ((col * 7919) ^ (row * 104729)) & 0x7FFFFFFF
                    hsh = (hsh ^ (hsh >> 11)) * 2654435761
                    noise = ((hsh >> 8) & 0xFFFF) / 65535.0 - 0.5
                    pix_tex = 1.0 + (0.05 * (1.0 - haze)) * noise
                    rr = out_r * pix_tex
                    gg = out_g * pix_tex
                    bb = out_b * pix_tex
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

            prev_z = z_surf
            r += step
            step = r * 0.012
            # Inside the 1 m window the surface has metre-scale features (single
            # crowns, walls): never stride past them, or trees alias into spikes.
            if r < R_NEAR_MAX and step > 1.5:
                step = 1.5
            vertical_step = (0.00045 if r < R_NEAR_MAX else 0.0008) * r * r
            if vertical_step < step:
                step = vertical_step
            if step < 0.02:
                step = 0.02
            elif step > 400.0:
                step = 400.0

        # Photographic sky from the equirectangular HDRI.
        if projection == PROJECTION_PANORAMA:
            horizon_row = int((top_rad - max_ang) / rad_per_row)
        else:
            horizon_row = int(cy_px - f_v_px * math.tan(max_ang - pitch_rad))
        if horizon_row > height:
            horizon_row = height
        for row in range(0, horizon_row):
            if projection == PROJECTION_PANORAMA:
                elev = top_rad - row * rad_per_row
            else:
                elev = math.atan((cy_px - row) / f_v_px) + pitch_rad
            u = (azimuth % (2.0 * math.pi)) / (2.0 * math.pi) * ws
            v = (0.5 - elev / math.pi) * hs
            ui = int(u) % ws
            vi = min(hs - 1, max(0, int(v)))
            sky_r = hdri[vi, ui, 0]
            sky_g = hdri[vi, ui, 1]
            sky_b = hdri[vi, ui, 2]
            # Blend toward haze at the horizon line for depth continuity.
            elev_deg = math.degrees(elev)
            if elev_deg < 3.0:
                t = (3.0 - max(elev_deg, 0.0)) / 3.0 * 0.55
                sky_r = sky_r + (haze_color[0] - sky_r) * t
                sky_g = sky_g + (haze_color[1] - sky_g) * t
                sky_b = sky_b + (haze_color[2] - sky_b) * t
            image[row, col, 0] = np.uint8(min(255.0, sky_r))
            image[row, col, 1] = np.uint8(min(255.0, sky_g))
            image[row, col, 2] = np.uint8(min(255.0, sky_b))


_DTM_STUB: NDArray[np.float32] = np.full((1, 1), np.nan, dtype=np.float32)
_BLD_STUB: NDArray[np.uint8] = np.zeros((1, 1), dtype=np.uint8)


def _window_arrays(
    window: dict[str, NDArray[np.generic]] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8], float, float]:
    """1 m LiDAR window, or stubs that make every near-field lookup miss.

    Per-viewpoint 1 m windows cost ~52 MB and a minute each, so the national site
    renders without them: every sample then takes the measured-terrain-plus-
    land-cover path, keeping the new lighting, colour, texture and haze.
    """
    if window is None:
        return _DTM_STUB, _DTM_STUB, _BLD_STUB, 0.0, 0.0
    return (
        np.asarray(window["dtm"], dtype=np.float32),
        np.asarray(window["dsm"], dtype=np.float32),
        np.asarray(window["buildings"], dtype=np.uint8),
        float(window["x0"]),
        float(window["y0"]),
    )


def _place_camera(
    dtm1: NDArray[np.float32],
    dsm1: NDArray[np.float32],
    win_x0: float,
    win_y0: float,
    easting: float,
    northing: float,
    center_azimuth_deg: float,
) -> tuple[float, float]:
    """Reconcile a recorded coordinate with the 1 m surface (no-op without one)."""
    if dtm1.shape[0] < 4:
        return easting, northing
    start_e, start_n = easting, northing
    cfx = easting - win_x0
    cfy = northing - win_y0
    if not (0 <= int(cfy) < dtm1.shape[0] and 0 <= int(cfx) < dtm1.shape[1]):
        return easting, northing
    cam_ndsm = float(dsm1[int(cfy), int(cfx)]) - float(dtm1[int(cfy), int(cfx)])
    if not math.isnan(cam_ndsm) and cam_ndsm > 2.0:
        best_d = 1e9
        for de in range(-25, 26, 2):
            for dn in range(-25, 26, 2):
                zg = float(dtm1[int(cfy + dn), int(cfx + de)])
                zs = float(dsm1[int(cfy + dn), int(cfx + de)])
                if math.isnan(zg) or math.isnan(zs) or zs - zg > 1.5:
                    continue
                d = de * de + dn * dn
                if d < best_d:
                    best_d = d
                    easting = win_x0 + cfx + de
                    northing = win_y0 + cfy + dn
    bx = math.sin(math.radians(center_azimuth_deg))
    by = math.cos(math.radians(center_azimuth_deg))
    best_block = 1e9
    best_e, best_n = easting, northing
    for forward in (0.0, 4.0, 8.0, 12.0):
        for lateral in (0.0, -8.0, 8.0):
            e2 = easting + bx * forward - by * lateral
            n2 = northing + by * forward + bx * lateral
            g2 = float(dtm1[int(n2 - win_y0), int(e2 - win_x0)])
            if math.isnan(g2):
                continue
            eye2 = g2 + EYE_M
            worst = -0.35
            probe = 2.0
            while probe <= 120.0:
                zp = float(dsm1[int(n2 - win_y0 + by * probe), int(e2 - win_x0 + bx * probe)])
                if not math.isnan(zp):
                    a = math.atan((zp - eye2) / probe)
                    if a > worst:
                        worst = a
                probe += 2.0 if probe < 30.0 else 6.0
            cost = worst + (forward + abs(lateral)) * 0.0008
            if cost < best_block - 0.002:
                best_block = cost
                best_e, best_n = e2, n2
    easting, northing = best_e, best_n
    if best_block > 0.105:
        for forward in range(0, 41, 4):
            for lateral in range(-40, 41, 4):
                e2 = start_e + bx * forward - by * lateral
                n2 = start_n + by * forward + bx * lateral
                g2 = float(dtm1[int(n2 - win_y0), int(e2 - win_x0)])
                zs0 = float(dsm1[int(n2 - win_y0), int(e2 - win_x0)])
                if math.isnan(g2) or math.isnan(zs0) or zs0 - g2 > 1.5:
                    continue
                eye2 = g2 + EYE_M
                worst = -0.35
                probe = 2.0
                while probe <= 120.0:
                    zp = float(dsm1[int(n2 - win_y0 + by * probe), int(e2 - win_x0 + bx * probe)])
                    if not math.isnan(zp):
                        a = math.atan((zp - eye2) / probe)
                        if a > worst:
                            worst = a
                    probe += 2.0 if probe < 30.0 else 6.0
                cost = worst + (forward + abs(lateral)) * 0.0004
                if cost < best_block - 0.002:
                    best_block = cost
                    best_e, best_n = e2, n2
        easting, northing = best_e, best_n
    return easting, northing


def _run2(
    window: dict[str, NDArray[np.generic]] | None,
    dem50: NDArray[np.float32],
    lc50: NDArray[np.uint8],
    hdri: NDArray[np.float32],
    *,
    easting: float,
    northing: float,
    col_azimuth: NDArray[np.float64],
    projection: int,
    render_h: int,
    top_rad: float,
    bottom_rad: float,
    f_v_px: float,
    cy_px: float,
    pitch_rad: float,
    center_azimuth_deg: float,
    dem10: NDArray[np.int16],
    lc10: NDArray[np.uint8],
    sat: NDArray[np.uint8],
    tex: NDArray[np.float32],
) -> tuple[NDArray[np.uint8], float]:
    dtm1, dsm1, bld, win_x0, win_y0 = _window_arrays(window)
    start_e, start_n = easting, northing
    easting, northing = _place_camera(
        dtm1, dsm1, win_x0, win_y0, easting, northing, center_azimuth_deg
    )
    ground = np.nan
    if dtm1.shape[0] > 4:
        ground = float(dtm1[int(northing - win_y0), int(easting - win_x0)])
    if math.isnan(ground):
        ground = float(_bilinear(dem50, easting, northing))
        if lc10.shape[0] > 4:
            col = int((easting - DEM10_X0) / DEM10_CELL)
            row = int((northing - DEM10_Y0) / DEM10_CELL)
            if 0 <= col < dem10.shape[1] and 0 <= row < dem10.shape[0]:
                value = int(dem10[row, col])
                if value > DEM10_INVALID:
                    ground = value * 0.1
    eye_z = ground + EYE_M
    band = hdri[int(hdri.shape[0] * 0.48) : int(hdri.shape[0] * 0.50)]
    haze_color = band.reshape(-1, 3).mean(axis=0).astype(np.float32)
    ambient_color = hdri[: hdri.shape[0] // 2].reshape(-1, 3).mean(axis=0).astype(np.float32)
    image = np.zeros((render_h, len(col_azimuth), 3), dtype=np.uint8)
    _render2_kernel(
        dtm1, dsm1, bld, win_x0, win_y0, dem50, dem10, True, lc50, lc10, True,
        sat, True, tex, tex.shape[0] >= 3, dtm1.shape[0] > 4, hdri, easting, northing, eye_z,
        col_azimuth, projection, render_h, top_rad, bottom_rad, f_v_px, cy_px,
        pitch_rad, haze_color, ambient_color, image,
    )  # fmt: skip
    return image, math.hypot(easting - start_e, northing - start_n)


def _finish(image: NDArray[np.uint8], width: int, height: int) -> Image.Image:
    # No PIL colour/contrast bump: the filmic tonemap provides the response curve.
    return Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)


def render_panorama2(
    dem50: NDArray[np.float32],
    lc50: NDArray[np.uint8],
    hdri: NDArray[np.float32],
    *,
    easting: float,
    northing: float,
    width: int,
    height: int,
    dem10: NDArray[np.int16],
    lc10: NDArray[np.uint8],
    sat: NDArray[np.uint8],
    tex: NDArray[np.float32],
    window: dict[str, NDArray[np.generic]] | None = None,
) -> Image.Image:
    """360-degree panorama with the next-generation material and light model."""
    scale = SUPERSAMPLE
    render_w = width * scale
    render_h = height * scale
    col_azimuth = 2.0 * math.pi * (np.arange(render_w, dtype=np.float64) + 0.5) / render_w
    image, _ = _run2(
        window, dem50, lc50, hdri,
        easting=easting, northing=northing, col_azimuth=col_azimuth,
        projection=PROJECTION_PANORAMA, render_h=render_h,
        top_rad=math.radians(PANORAMA_TOP_DEG), bottom_rad=math.radians(PANORAMA_BOTTOM_DEG),
        f_v_px=1.0, cy_px=0.0, pitch_rad=0.0, center_azimuth_deg=-999.0,
        dem10=dem10, lc10=lc10, sat=sat, tex=tex,
    )  # fmt: skip
    return _finish(image, width, height)


def render_view2(
    window: dict[str, NDArray[np.generic]] | None,
    dem50: NDArray[np.float32],
    lc50: NDArray[np.uint8],
    hdri: NDArray[np.float32],
    *,
    easting: float,
    northing: float,
    center_azimuth_deg: float,
    hfov_deg: float,
    width: int,
    height: int,
    horizon_fraction: float,
    dem10: NDArray[np.int16],
    lc10: NDArray[np.uint8],
    sat: NDArray[np.uint8],
    tex: NDArray[np.float32],
) -> tuple[Image.Image, float]:
    """Rectilinear camera view; returns the image and how far the camera moved."""
    scale = SUPERSAMPLE
    render_w = width * scale
    render_h = height * scale
    f_h = (render_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    center = math.radians(center_azimuth_deg)
    offsets = (np.arange(render_w, dtype=np.float64) + 0.5) - render_w / 2.0
    col_azimuth = center + np.arctan(offsets / f_h)
    cy = render_h * horizon_fraction
    image, offset_m = _run2(
        window, dem50, lc50, hdri,
        easting=easting, northing=northing, col_azimuth=col_azimuth,
        projection=PROJECTION_RECTILINEAR, render_h=render_h,
        top_rad=math.atan(cy / f_h), bottom_rad=math.atan((cy - render_h) / f_h),
        f_v_px=f_h, cy_px=cy, pitch_rad=0.0, center_azimuth_deg=center_azimuth_deg,
        dem10=dem10, lc10=lc10, sat=sat, tex=tex,
    )  # fmt: skip
    return _finish(image, width, height), offset_m
