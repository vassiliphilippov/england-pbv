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

## Cycle 7 — habitat-driven moorland palette (Priority Habitats Inventory) — ACCEPTED

Change: Natural England Priority Habitats Inventory polygons (ArcGIS FeatureServer, native
EPSG:27700) fetched for all 20 calibration sites and burned as a +2 flag on the 10 m
land-cover grid (1.56 M cells; `england_pbv.data.habitats`). The renderer's moor grass /
bracken / heather palette now fires on the habitat flag instead of altitude; altitude
thresholds remain only as fallback where no 10 m data exists. The per-site moor-class
counts alone validated the idea: chalk sites (Coombe, Coaley, Uffington, Devil's Dyke, Wye
Downs) correctly 0, Surprise View 425, High Shield 762, Valley of Rocks 110.

Verdict: clear accept. Mam Tor's crest turns grazed green (the altitude ramp painted its
517 m top tawny) while Kinder's moors behind band russet+heather like the photo. Valley of
Rocks transforms — sea-level bracken slopes in russet with green paths winding through,
where altitude gating had left uniform green. Wensleydale's foreground pasture goes green;
Swaledale's dale-edge moor banding appears. One accepted tradeoff: PHI misses bracken on
non-priority land (Latrigg's plantation-edge bracken band goes green).

Queued for cycle 8: (1) hillside/crag cameras (Surprise View's rim camera slides back onto
the plateau — the photo stands at the crag lip; Swaledale/Conistone own-slope): consider a
"cliff-edge" camera rule using the 10 m DEM's max forward drop; (2) canopy clump-edge
fortress walls; (3) distant moor tops too pale (Swaledale far rim vs photo's dark heather
line); (4) rock material for tors/limestone pavement (Valley of Rocks' Castle Rock,
Conistone Pie) — OSM natural=bare_rock/cliff as a data source.

## Cycle 8 — crag-lip cameras, deeper distant moor, crown-notch retune — ACCEPTED (one revert)

Changes: (1) crest-snap no longer runs when a view bearing is known — at a crag lip the
ground behind always rises, so snapping retreated the camera up-slope away from the edge
the photographer stands on; the openness search (extended to 64 m along-bearing) is the
right model there. Crest-snap still serves bearingless panoramas (cone summits).
(2) Moor-habitat cells darken shade ×0.92 (matte dark vegetation — the far heather rim was
hazing out pale). (3) Crown-notch clamp 0.15 → 0.10 (between "fortress walls" and the
saw-tooth that 0.05 produced).

Verdict: the camera fix is the cycle's win — Surprise View recovers the photo's whole
composition (Derwentwater filling the frame from the lip; it was a sliver from the
plateau), Swaledale finally clears its shoulder (valley floor, walled fields, winding
dale). Mam Tor unchanged; Coombe reframed slightly and lost its ring artifact.

REVERTED within the cycle: tapering crown fine-roughness by dome height (to cure
saw-tooth silhouettes) showed no benefit — an A/B against uniform roughness rendered the
same crown-shoulder contour bands on Wye Downs' near scrub (they are camera-geometry
dependent: the 64 m openness search now stands closer above the wood, where ±18%/3 m
roughness is too weak to fragment the contours). Uniform roughness kept for simplicity.
The saw-tooth silhouettes come from per-clump height variance (0.6+0.8×noise), not fine
roughness — a future cycle can narrow that range and strengthen close-range roughness.

Queued for cycle 9: (1) trial narrower per-clump canopy height variance for calmer wood
silhouettes; (2) near-foreground own-slope texture (Mam Tor/Conistone/Wensleydale smooth
domes remain the weakest structural element); (3) rock material for tors/limestone
(Valley of Rocks, Conistone Pie); (4) woods seen from above render as smooth dark aprons
(Surprise View foreground) — canopy-top texture at steep down-angles.

## Cycle 9 — canopy texture overhaul + two-scale ground mottling — ACCEPTED

Changes: (1) crown fine-roughness amplitude grows near the camera (0.36 + 0.50·e^(−r/250))
— uniform across each crown per the c8 lesson, but strong enough close-up to fragment the
contour rings; (2) per-clump height range narrowed 0.6–1.4 → 0.75–1.3 (saw-tooth wood
silhouettes came from neighbouring crowns jumping 2x); (3) per-clump species colour
variation (±15% red / ±9% green) — one green for every crown read as plastic; (4) ground
mottling gains a second coarser octave (~18 m tussock/rush patches to 1.5 km) so
own-slope foregrounds stop reading as polished domes.

Verdict: accept. Wye Downs' scrub mass is the showcase — species-varied, wavy, ring-free
crowns that finally match the photo's chaos; Mam Tor's crest breaks into tussocky
blotches; Swaledale calms its tree cutouts; Latrigg keeps its structure with richer woods.

Queued for cycle 10: (1) rock material for tors/limestone from OSM natural=bare_rock/
cliff/scree (Valley of Rocks' Castle Rock, Conistone Pie); (2) worn-path near-field
confinement paints dashed arcs on grazing fields (Latrigg bottom-right) — consider
painting by distance-to-polyline instead of cell-centre; (3) review the set-2 pairs not
recently inspected (High Shield, Old Stell Crag, Porlock, Lawley, Wrekin); (4) built-up
mauve sprawl (Keswick) still reads as speckle, not town.

## Cycle 10 — OSM rock material + corduroy fix — ACCEPTED after three near-field retunes

Changes: (1) OSM rock features (natural=bare_rock/scree/rock/stone/shingle areas,
natural=cliff lines) fetched for all 20 sites (`england_pbv.data.rocks`, 38,212 cells)
and burned as a +3 flag (rock wins over the moor flag; the path flag wins over rock);
(2) renderer paints flagged cells as grey crag blended through vegetation by near-binary
~8 m coverage noise, faded out below 150 m; (3) the c9 tussock-patch octave is gated by
view incidence — seen edge-on it had compressed into horizontal "corduroy" stripes
(High Shield, Old Stell Crag foregrounds).

Verdict: accept. Valley of Rocks is the showcase — Castle Rock and its neighbour tors
render as grey crag masses and the pair finally reads as the same place; High Shield and
Surprise View gain believable crag accents on the Whin Sill and Borrowdale fells.

Three near-field failures on the way (all recorded): 100% grey paint made Conistone's
limestone hillside a concrete dome; a smooth 0.35–0.9 blend still averaged into a
grey-green wash; near-binary clusters AND a 2 m geometry bump both collapsed at grazing
view — the bump actually widened the grey (bumped samples occlude more spans). Root
cause: below ~150 m at grazing incidence, adjacent rays sample the world centimetres
apart, so the whole foreground reflects a single noise strip — no world-space pattern
survives. Near rock paint now fades out; near-field outcrops (Conistone's pavement, Old
Stell Crag's framing tors at the camera) are a bounded limitation, honestly unrendered.

Queued for cycle 11: (1) Sentinel-2 real-imagery far-field texture (research in
progress — user-suggested; would replace procedural patchwork beyond a few km with the
real one); (2) built-up mauve sprawl; (3) consider photo set 3 — most set-2 pairs are
now structurally close.

## Cycle 11 — real Sentinel-2 far-field texture — ACCEPTED after lighting retune

Changes (user-suggested data source): a 10 m true-colour England mosaic
(`england_pbv.data.satellite`, 1.37 B cells) built from the least-cloudy May–September
Sentinel-2 scene per MGRS tile (AWS Open Data, anonymous TCI COGs; all 66 scenes ≤12%
cloud, most 0%). The renderer blends the real imagery in beyond 2 km (full by 3.5 km,
85% strength, ×1.22 gain) — nearer than that, grazing incidence collapses any
world-space texture (c10 lesson), so the procedural surface remains.

First attempt blended RAW satellite colour after our shading — distant hillsides went
flat and washed (the imagery's own baked shading is too subtle at distance to carry
relief). Retuned: satellite is treated as albedo lit by our sun — multiplied by the
sun+cloud shade WITHOUT the procedural field/grain noise it replaces.

Verdict (A/B crops c10 vs c11): accept. Mam Tor's Edale mid-distance swaps repeating
procedural patchwork for the real irregular field mosaic; Wensleydale's horizon gains
the actual dark moor line; Kinder's russet moor reads organically mottled. Near fields
stay procedural and identical.

Also this cycle: satellite/aerial imagery research (workflow, 18 sources, endpoints
probed live) recorded in specifications/data_sources.md — EA Vertical Aerial
Photography (10–25 cm, OGL) noted as a future near-field source; Google/Bing/Esri/
Mapbox ruled out on licence terms; EOX cloudless ruled out on NC+SA propagation.

Queued for cycle 12: (1) satellite tone match (TCI runs slightly pale vs the graded
palette — consider saturation lift on satellite pixels); (2) check remaining pairs for
scene-date seams; (3) built-up mauve sprawl; (4) photo set 3 — the set-2 pairs are now
structurally close across the board.

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
- (c8) Saw-tooth wood silhouettes are NOT caused by crown fine-roughness (A/B verified:
  tapering it by dome height changes nothing visible) — they come from per-clump height
  variance (0.6+0.8×noise). Contour rings on crowns seen close-from-above need STRONGER
  near-range roughness, not redistribution.
- (c10) Painting a mapped surface class at 100% over broad areas makes concrete-uniform
  ground — blend materials through the underlying vegetation with near-binary coverage
  noise instead.
- (c10) Below ~150 m at grazing incidence NO world-space pattern survives — colour
  patches, thresholded clusters and 2 m geometry bumps all collapse into a uniform wash
  (adjacent rays sample centimetres apart; bumps just widen occluding spans). Fade
  world-pattern materials out near the camera; don't fight this with amplitude.
