"""Inline SVG generators: horizon panorama, polar reach diagram, component bars."""

import math

import numpy as np
from numpy.typing import NDArray

CHART_W: int = 920
CHART_H: int = 240
MARGIN_L: int = 44
MARGIN_R: int = 12
MARGIN_T: int = 14
MARGIN_B: int = 34


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def horizon_panorama_svg(
    horizon_deg: NDArray[np.float32],
    horizon_veg_deg: NDArray[np.float32] | None,
    d_far_km: NDArray[np.float32],
) -> str:
    """Skyline chart: filled terrain silhouette over azimuth with a distance strip below."""
    n = len(horizon_deg)
    inner_w = CHART_W - MARGIN_L - MARGIN_R
    inner_h = CHART_H - MARGIN_T - MARGIN_B

    y_min = min(-2.0, float(np.min(horizon_deg)) - 0.5)
    y_max = max(4.0, float(np.max(horizon_deg)) + 0.8)
    if horizon_veg_deg is not None:
        y_max = max(y_max, float(np.max(horizon_veg_deg)) + 0.8)

    def x_at(index: int) -> float:
        return MARGIN_L + index / n * inner_w

    def y_at(value: float) -> float:
        return MARGIN_T + (y_max - value) / (y_max - y_min) * inner_h

    skyline = [(x_at(i), y_at(float(horizon_deg[i]))) for i in range(0, n, 2)]
    skyline.append((x_at(n), y_at(float(horizon_deg[0]))))
    base_y = y_at(y_min)
    fill_path = (
        f"M{MARGIN_L},{base_y:.1f} L"
        + _polyline(skyline)
        + f" L{MARGIN_L + inner_w},{base_y:.1f} Z"
    )

    parts: list[str] = [
        f'<svg viewBox="0 0 {CHART_W} {CHART_H}" xmlns="http://www.w3.org/2000/svg" '
        'class="horizon-chart" role="img" aria-label="Horizon panorama">'
    ]
    # zero-degree line
    zero_y = y_at(0.0)
    parts.append(
        f'<line x1="{MARGIN_L}" y1="{zero_y:.1f}" x2="{CHART_W - MARGIN_R}" '
        f'y2="{zero_y:.1f}" class="zero-line"/>'
    )
    parts.append(f'<path d="{fill_path}" class="terrain-fill"/>')
    parts.append(f'<polyline points="{_polyline(skyline)}" class="terrain-line" fill="none"/>')

    if horizon_veg_deg is not None:
        veg = [(x_at(i), y_at(float(horizon_veg_deg[i]))) for i in range(0, n, 2)]
        veg.append((x_at(n), y_at(float(horizon_veg_deg[0]))))
        parts.append(f'<polyline points="{_polyline(veg)}" class="veg-line" fill="none"/>')

    # distance strip: colour each azimuth by how far the ground is visible (thinned 3x)
    strip_y = CHART_H - MARGIN_B + 8
    strip_h = 8
    thin = 3
    step = inner_w / n * thin
    for i in range(0, n, thin):
        dist = float(np.max(d_far_km[i : i + thin]))
        t = min(1.0, dist / 50.0)
        # light grey (near) -> deep blue (far)
        r = int(226 - 170 * t)
        g = int(232 - 130 * t)
        b = int(240 - 60 * t)
        parts.append(
            f'<rect x="{x_at(i):.1f}" y="{strip_y}" width="{step + 0.5:.2f}" height="{strip_h}" '
            f'fill="rgb({r},{g},{b})"/>'
        )

    # compass labels and axis
    for frac, label in [(0.0, "N"), (0.25, "E"), (0.5, "S"), (0.75, "W"), (1.0, "N")]:
        x = MARGIN_L + frac * inner_w
        parts.append(
            f'<text x="{x:.1f}" y="{CHART_H - 6}" class="axis-label" '
            f'text-anchor="middle">{label}</text>'
        )
    for value in (0, 5, 10):
        if y_min < value < y_max:
            parts.append(
                f'<text x="{MARGIN_L - 6}" y="{y_at(value) + 4:.1f}" class="axis-label" '
                f'text-anchor="end">{value}&#176;</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def polar_reach_svg(d_far_km: NDArray[np.float32], max_km: float = 60.0) -> str:
    """Compass rose: wedge length = how far the ground is visible on that bearing."""
    size = 260
    cx = cy = size / 2
    r_max = size / 2 - 20
    n = len(d_far_km)
    parts: list[str] = [
        f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" '
        'class="polar-chart" role="img" aria-label="View reach by direction">'
    ]
    for ring_km in (10, 25, 50):
        r = r_max * math.sqrt(ring_km / max_km)
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" class="polar-ring"/>')
        parts.append(
            f'<text x="{cx + 3:.1f}" y="{cy - r - 2:.1f}" '
            f'class="polar-ring-label">{ring_km}km</text>'
        )
    thin = 3
    step = 2.0 * math.pi / n * thin
    for i in range(0, n, thin):
        dist = min(float(np.max(d_far_km[i : i + thin])), max_km)
        if dist <= 0.5:
            continue
        r = r_max * math.sqrt(dist / max_km)
        a0 = (i / thin) * step - math.pi / 2.0
        a1 = a0 + step * 1.05
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        t = dist / max_km
        rr = int(226 - 170 * t)
        gg = int(232 - 130 * t)
        bb = int(240 - 60 * t)
        parts.append(
            f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{r:.1f},{r:.1f} 0 0 1 {x1:.1f},{y1:.1f} Z" '
            f'fill="rgb({rr},{gg},{bb})"/>'
        )
    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        a = math.radians(angle) - math.pi / 2.0
        x = cx + (r_max + 11) * math.cos(a)
        y = cy + (r_max + 11) * math.sin(a) + 4
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="axis-label" text-anchor="middle">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
