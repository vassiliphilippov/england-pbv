# Render iteration log

Loop protocol: each cycle = visual compare (ten calibration pairs) → hypothesize → change →
re-render (`uv run python -m england_pbv.viewshed.calibration --suffix cN`) → compare vs
previous → accept or revert. Failures recorded so they are not repeated.

## Baseline (v7, pre-loop)

Ten-pair review findings, ranked: (1) 50 m terrain false brows eat mid-ground at sharp
summits/scarps (Mam Tor, Coaley, Uffington's Manger); (2) 50 m land-cover kills hedgerow
oaks that define English vales; (3) flat radial-only lighting — no sun modelling, no cloud
shadows; (4) cuboid trees; (5) flat blue water; (6) undersaturated palette, haze slightly
early; missing classes (heather, bare rock, buildings-as-shapes).

## Cycle 1 — lighting overhaul — ACCEPTED

Changes: true-normal diffuse sun (radial + lateral slope → surface normal, sun SW @38°,
ambient 0.52 + diffuse 0.78); world-space cloud shadows (2-octave noise, 1250 m scale,
strength 0.30); water blends 45% toward sky-horizon colour before shading; tone grade
(saturation ×1.16, contrast ×1.07); haze distance 13 → 16 km.

Verdict: clearly better on all inspected pairs. Ragged Stone: sun sculpts valley folds,
cloud-dappled patchwork matches the photo's character. Latrigg: fells gain modelling, lake
reads silver like the photo. No regressions seen.

Carry-over observations for next cycles: (a) near hillsides now look silk-smooth — the
stronger diffuse washes relative texture; boost close-range grain. (b) Terrain false brows
remain the dominant structural error → 10 m EA DTM (download in progress). (c) Trees still
cuboid. (d) Villages still mauve mist.

## Cycle 2 — England 10 m terrain — ACCEPTED

Changes: EA 10 m national DTM (5.3 GB zip -> int16 decimetre grid, 7.6 GB, row 0 = south,
nodata for sea/Wales/Scotland) sampled by the renderer with bilinear + fall-back to the
50 m GB grid (`_sample_elev`); used for targets, lateral normals, crest snap and eye height.

Verdict: major structural win. Uffington: the Manger coombe (the photo's entire subject,
absent at 50 m) renders with its folds. Coaley: scarp woods cascade correctly, Cam Long
Down present, foreground blobs gone. Mam Tor: sharper valley sides above the brow; the brow
itself persists — the 10 m profile confirms the crest genuinely descends ~30 m/250 m, and
the remaining mismatch is camera-position sensitivity at a break of slope (a 10 m nudge
changes the foreground drastically).

Carry-over for next cycles: (a) "photographer steps to the brow": nudge the render camera
up to ~15 m along the view bearing when ground immediately ahead falls — photographers
stand at the edge, grid refs quantize to 10 m; (b) land cover still 50 m — trees/fields
should sample native 10 m WorldCover (restores hedgerow oaks); (c) near-foreground still
silk-smooth under the new sun; (d) villages remain mauve mist (OSM buildings).

## Failures ledger (do not repeat)

- (from pre-loop) Fixed march steps near the observer paint terrace bands — step must scale
  with r² vertical angular resolution.
- (from pre-loop) Crest-snapping the camera to the local max retreats from brows — only snap
  when >2.5 m below the crest.
- (from pre-loop) Height-scaling trees near the viewpoint builds canopy staircases — thin
  density, keep full height.
