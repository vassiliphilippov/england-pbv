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

## Failures ledger (do not repeat)

- (from pre-loop) Fixed march steps near the observer paint terrace bands — step must scale
  with r² vertical angular resolution.
- (from pre-loop) Crest-snapping the camera to the local max retreats from brows — only snap
  when >2.5 m below the crest.
- (from pre-loop) Height-scaling trees near the viewpoint builds canopy staircases — thin
  density, keep full height.
