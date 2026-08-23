# Panorama Render Calibration

`england_pbv.viewshed.render` produces a synthetic 360° panorama per viewpoint (1440×400,
0.25°/column, +8°..−12°) by ray-marching the leaf-on surface (terrain + tree canopies +15 m
with per-clump height noise, buildings +8 m) with Earth curvature/refraction, painting visible
spans with land-cover colours, sun-slope shading (sun from SW), per-pixel texture, and
atmospheric perspective.

## Calibration protocol

Renders of known viewpoints (Coombe Hill, Sutton Bank, Mam Tor, Worcestershire Beacon,
Glastonbury Tor, Devil's Dyke) were compared against Geograph photographs of the same views by
independent review agents; their concrete feedback (colour samples, haze onset, occlusion
errors) drove five iterations (v1→v5). Applied findings:

1. **The viewpoint is a clearing** — obstacle heights ramp from zero inside 120 m to full at
   260 m. Without this, 10 m→50 m land-cover sampling walls off signature open panoramas
   (Coombe NW, Sutton Bank W) with canopy prisms that don't exist at the real standing spot.
2. **Camera snaps to the local crest** (highest ground within ±50 m) + 0.8 m eye boost —
   otherwise cone summits (Glastonbury Tor) occlude their own view.
3. **Haze**: blue-white (201,214,227), 1−exp(−r/13 km) — visible fade from ~5 km, ~90% by
   30 km; patchwork/grain contrast additionally damped by 0.75·haze so the horizon stays calm.
4. **Sky**: desaturated English summer — zenith (150,180,215) → horizon (218,228,236).
5. **Palette** (WorldCover classes): warm summer grass (150,156,96) turning tawny acid-moor
   (152,142,92) above 350–550 m; ripe-straw cropland (192,174,116); warm brick-grey built
   (158,140,130); canopy (86,106,56) with sunlit-top/shadowed-base gradient and foliage
   sparkle.
6. **Facet banding** reduced by radial-slope smoothing (2-sample average) + per-pixel texture.

## v6 additions

- **2× supersampling** with Lanczos downscale — kills skyline aliasing and distant speckle.
- **Procedural clouds**: two-octave stretched value noise, seeded per viewpoint, thinning
  toward the zenith and melting into horizon haze.
- **Hedgerow hints**: a darkened strip wherever the 260 m field cell changes under
  grass/cropland — distant patchwork reads as bounded fields.
- **Near-ground texture**: distance-scaled micro-grain (4–5 m mottling within ~1 km) so
  camera-view foregrounds aren't smooth walls.
- **Rectilinear camera views** (`render_view`): 68° lens, slight down-tilt, horizon at 42%
  frame height — shares the panorama kernel via a tan projection.
- **Best-direction picker** (`best_view_directions`): 70° circular window maximising mean
  capped visible distance (veg-aware); second direction kept when ≥55% of the best and ≥75°
  apart.

## Known remaining gaps (honest)

- 50 m facets still band on smooth near domes; field boundaries are noise-driven, not mapped.
- No rock material for quarried/limestone scars; no season/weather variants.
- Trees are statistical clumps, not the real hedgerow layout; buildings are height noise.
- The render shows the *model's* world — the same world the scores are computed from, which is
  exactly what makes it useful: you see what the algorithm thinks the view is.
