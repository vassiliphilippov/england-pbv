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

## Cycle 3 — native 10 m land cover + brow nudge — ACCEPTED

Changes: England 10 m WorldCover grid (uint8, same frame as the 10 m DTM) sampled for
geometry and colour with 50 m fallback (Wales keeps real 10 m cover; only terrain falls
back there). "Step to the brow": render_view camera advances up to 2x12 m along the view
bearing when the ground ahead drops >1 m.

Verdict: strong accept. Latrigg is now near-1:1 with the photo (Keswick as textured town
mass, plantation belt, fell relief). Coombe's vale gained hedgerow trees and farm
buildings; Mam Tor's Edale side shows walled fields like the photo. The Mam Tor crown
persists — the nudge threshold (1 m per 12 m step) never fires on the flat crest; the
photo camera's advantage is microrelief (path cut) below 10 m resolution.

Carry-over: (a) relax nudge to ~0.4 m/step x4 steps and re-test Mam Tor only; (b) fell
colours — bracken russet/heather on steep acid upland (photo Latrigg foreground is orange
bracken); (c) round tree canopy tops; (d) photo-like cumulus sky; (e) consider swapping in
a fresh photo set soon — Latrigg/Coaley/Uffington/Devil's Dyke are close.

## Cycle 4 — canopy domes, bracken/heather, perspective clouds, nudge retune — ACCEPTED after retune

Changes: rounded tree crowns (dome profile per 35 m clump); bracken russet on steep upland
grass (z>280 m, slope>0.17, capped ~0.55 strength) + heather dusk on high shrub; cumulus
projected onto a world-space 1500 m cloud plane (perspective shrink toward horizon) with a
3-octave density and smoothstep alpha; brow nudge relaxed to 0.4 m per 12 m step x4.

First attempt FAILED two ways (recorded): (1) bracken at z>220/slope>0.12 painted Edale's
green walled fields orange — improved-pasture vs open-fell cannot be told apart by slope
alone, so thresholds must stay conservative; (2) 2-octave plane clouds with hard alpha ramp
rendered as paper-cutout blobs — clouds need a fine erosion octave and smoothstep edges.
Both fixed in the same cycle; after retune, Latrigg shows russet fell accents + soft wisps,
Mam Tor's valley reads green-pasture-to-brown-moor like the photo.

Remaining niggles: concentric arc patterns inside near tree crowns; Mam Tor's crown
foreground (flat 10 m crest, nudge cannot fire — bounded known limitation).

## Cycle 5 — generalization check with a fresh photo set — PASS (no overfitting)

Set 2 (verification/render_calibration_photos_set2.json): ten new documented photos in
regions set 1 never touched (Exmoor coast, Swaledale/Wensleydale/Wharfedale, Simonside,
Hadrian's Wall, Shropshire, Kent Downs). Rendered with the engine UNCHANGED.

Verdict: improvements generalize — Wrekin (plain to hazy horizon), Swaledale's far dale
sides (walled patchwork, moor tops), clouds/haze/colours all transfer to unseen sites. The
one systematic residual appears in both sets: cameras standing ON steep hillsides or at
break-of-slope show too much of their own slope (Mam Tor set 1; Swaledale set 2) — a ±10 m
camera-position sensitivity, not a data or palette failure.

Queued for cycle 6 (consolidated): (1) micro-optimized render camera — search ±30 m for
the spot that minimizes near-ground angular occupancy toward the view bearing ("stand
where the view opens", what photographers actually do); (2) OSM footpaths drawn as worn
lines — the depth cue that makes descending foregrounds read as descending; (3)
grazing-angle darkening to break the "lit dome" reading of convex foregrounds.

## Cycle 6 — camera openness search, OSM footpaths, grazing shading, Wye Downs fix — ACCEPTED after retune

Changes: (1) Wye Downs metadata corrected — the stored grid ref was the SUBJECT (coombe rim),
not the camera (TR 0755 4526), 35 m apart and the difference between a green wall and the
real vale panorama; (2) `_open_camera` — the render camera searches 5 along-bearing x 5
lateral (±16 m) x 8 m offsets for the spot minimizing near-ground angular occupancy
(`_foreground_block`: 5 bearings ±30°, r=8..150 m, move cost 0.00035/m) — "stand where the
view opens", what photographers do; (3) OSM footpaths for the 20 calibration sites burned
into the 10 m land-cover grid as +1 flags (157,240 cells) and rendered as pale worn lines
fading to 2.5 km; (4) grazing-angle darkening on turf seen nearly edge-on.

First attempt FAILED two ways (recorded): (a) the path colour was sampled at the JITTERED
colour position (±30 m near the camera), smearing 10 m path cells into giant swirling blobs
across Mam Tor's crest — fixed by sampling paths at the raw march position and, within
500 m, confining paint to the middle of the cell (trodden line, not a 10 m swathe);
(b) crown-dome height contours painted as huge concentric amphitheatre rings when a wood
sat close below the Coombe camera — fixed with ~3 m fine foliage roughness (±18%) that
fragments the contours (also kills the "broccoli onion-ring" arcs inside near crowns).

Verdict after retune: Wye Downs transformed (worst pair → correct open-vale structure with
scrub cascading below the rim); Swaledale sees more of the dale side (partial — own-slope
still fills the lower half); Coombe's distant path reads as a believable chalk trail across
the fields; Mam Tor's swirls gone, faint worn hint remains; no regressions on the rest.

Queued for cycle 7: (1) palette — summer-tawny arable + the moor-grass ramp read wrong
against green-field photos (Mam Tor's 517 m crest is grazed GREEN sward; the ramp paints it
tawny); consider Natural England Priority Habitats Inventory (open data) to separate grass
moorland / upland heath / improved pasture properly instead of altitude proxies;
(2) canopy clump-edge vertical "fortress walls" — taper dome edges below the 0.15 clamp;
(3) hillside cameras still show much own-slope (Swaledale) — bounded, maybe accept;
(4) cumulus density still sparse vs photos on some pairs.

## Failures ledger (do not repeat)

- (from pre-loop) Fixed march steps near the observer paint terrace bands — step must scale
  with r² vertical angular resolution.
- (from pre-loop) Crest-snapping the camera to the local max retreats from brows — only snap
  when >2.5 m below the crest.
- (from pre-loop) Height-scaling trees near the viewpoint builds canopy staircases — thin
  density, keep full height.
- (c4) Bracken by slope+altitude alone at z>220 m paints improved valley pasture orange —
  keep z>280 m and steep-slope gates; farmland vs fell needs a better discriminator.
- (c4) World-plane cumulus with a hard 2-octave threshold = paper cutouts — always include
  a fine erosion octave and smoothstep the alpha.
- (c6) Path colour looked up at the JITTERED colour-sample position smears a 10 m path cell
  into 30 m swirling blobs near the camera — sample linear features at the raw march
  position, and within ~500 m confine the paint to the middle of the cell.
- (c6) Smooth crown-dome height contours paint giant concentric amphitheatre rings when a
  wood sits close below the camera — canopy height needs fine (~3 m) roughness so the
  contours fragment into foliage.
