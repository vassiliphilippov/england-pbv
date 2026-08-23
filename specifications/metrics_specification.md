# Metrics Specification

Normative definitions for every metric the pipeline computes. Source of truth alongside
`england_pbv/constants.py`; update both in the same change.

## Principles (from `research/`)

1. **No feedback-free beauty scalar exists.** We measure the geometry and content of the
   visual field and publish a *view potential* ranking as a documented convention. Raw
   components stay visible everywhere.
2. **Relative, not absolute altitude.** The primitive is `target angle − largest intervening
   terrain angle` along each bearing, not elevation.
3. **Angular (visual-magnitude) weighting.** A hectare 20 km away is not a hectare 200 m away.
   Horizon-angle *record increments* during the radial sweep are exact angular areas of the
   panorama and weight every composition metric.
4. **Vegetation blocks geometrically.** Trees only block the view when their tops rise above
   the sightline — woodland on a scarp slope below an escarpment viewpoint does not.

## Observation model

| Parameter | Value | Note |
|---|---|---|
| Grid | EPSG:27700, 50 m cells, GB-wide (Wales/Scotland present for border viewsheds) | OS Terrain 50 |
| Eye height | 1.7 m above bare-earth DTM | never place the observer on canopy |
| Max distance | 60 km | curvature makes further terrain marginal at English elevations |
| Curvature/refraction | `z_eff = z − (1−k)·r²/(2R)`, k = 1/7, R = 6371 km | matches GDAL default 0.85714 |
| Azimuths | 720 (0.5°) | |
| Radial sampling | 50 m steps to 2 km, 100 m to 8 km, 200 m to 20 km, 400 m to 60 km; first sample at ≥75 m | 259 samples/ray, bilinear DEM interpolation |
| Distance bands | 0–2, 2–8, 8–20, 20–60 km | near / middle / far / horizon |
| Obstacle model | tree cells +15 m, built cells +8 m, as *blockers* only (targets stay bare ground) | WorldCover classes 10/50 |

## Per-candidate metrics (`ViewMetrics`)

**Sweep bookkeeping per bearing θ**: running max elevation angle `H(θ)` (bare) and
`H_veg(θ)` (with obstacle heights); a sample is visible when its bare angle sets a new record
against the respective max. `α₀(θ)` = angle to the nearest sampled ground (75–100 m out).

| Metric | Definition |
|---|---|
| `visible_area_km2_by_band` | Σ r·Δr·Δθ over visible samples per band (plan area, bare) |
| `visible_area_veg_km2` | same total with tree/built blockers |
| `veg_retention` | visible_area_veg / visible_area (1 when nothing to lose) |
| `angular_area_deg2_by_band` | Σ (angle-record increments)·Δθ per band, in deg² of panorama — the visual-magnitude measure |
| `mean/median/p90_horizon_deg` | statistics of H(θ) |
| `skyline_total_variation_deg` | Σ\|ΔH\| after 2.5° circular smoothing (skyline roughness) |
| `open_fraction` | share of bearings with H(θ) < 2° |
| `far_fraction`, `far_fraction_veg` | share of bearings with visible ground beyond 10 km |
| `longest_far_arc_deg`, `_veg_` | longest contiguous such arc (circular) |
| `d_far_median/p90/max_km`, `d_far_veg_p90_km` | farthest-visible-ground distances |
| `mean_depression_deg` | mean of max(0, −α₀) — how steeply the ground falls away right at the point |
| `max_sector_drop_m` | observer − min over twelve 30° sectors of mean terrain elevation in the 0.5–3 km ring |
| `visible_relief_m` | max − min elevation of visible terrain |
| `landcover_angular_fractions` | angular share of each WorldCover class in the visible panorama |
| `shannon_diversity` | −Σ p ln p / ln 10 over those shares (nodata excluded) |
| `water/built/tree_fraction` | convenience shares |
| `near_tree_fraction` | share of bearings whose first 300 m ring is ≥50% tree cells (reported, not scored — superseded by the geometric veg pass) |

## Scoring (`ComponentScores`, `view_potential`)

Each raw input becomes its **percentile among all ~46k England candidates** (frozen
transforms in `score_inputs.py`); components are means of input percentiles; the composite
is the unweighted mean of the six components. Equal weighting is itself a documented human
convention, not a discovered law.

| Component | Inputs |
|---|---|
| prospect | angular area beyond 2 km; total visible plan area |
| openness | far_fraction_veg; longest_far_arc_veg |
| drop | max_sector_drop; mean_depression |
| depth | d_far_veg_p90; entropy of angular area across the 4 distance bands |
| diversity | shannon_diversity |
| clearness | veg_retention; (1 − built_fraction) |

`regional_percentile` = percentile of the composite among candidates within 30 km, so a
Chilterns escarpment competes with the Chilterns, not the Lake District.

## Refinement

Top 8,000 deduplicated candidates (250 m keep-best) get a two-pass local grid search for the
best exact standing spot: ±200 m at 50 m steps, then ±50 m at 12.5 m steps around the coarse
winner, scored with the frozen percentile transforms; total drift capped at 280 m so a
refined spot still represents its origin.

## Known limitations (v0.1)

- 50 m DEM smooths cliff lips: gritstone edges (Curbar, Stanage) and small limestone features
  (Malham Cove) under-score. Fix: EA 1 m LiDAR near-field refinement (phase 2).
- Tower viewpoints (Leith Hill, Broadway Tower) are scored at ground level; platform height
  is not modelled.
- WorldCover 10 m→50 m sampling misclassifies some rocky summits as built-up.
- Sea is terrain at 0 m: coastal clifftops legitimately dominate the national ranking; the
  site exposes `water_fraction` so inland and coastal views can be browsed separately.
- Metrics describe geometric *potential*; weather, light, foreground furniture and access are
  out of scope for scoring.
