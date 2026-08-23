# Verification Protocol

The algorithm must find known great views and reject adversarial non-views — without human
ratings ever entering the scoring itself.

## Dataset (`verification/viewpoints.json`)

- **60 positives**: famous, widely-celebrated English viewpoints (Chilterns to Northumberland),
  researched 2026-08-23. Coordinates are the *actual viewing spot*, arbitrated per point across
  DoBIH surveyed grid refs, OSM `tourism=viewpoint`/`natural=peak`/trig nodes, Wikipedia
  geohack, and Historic England records; each entry records `coord_source` and `confidence`.
- **20 negative controls**, chosen adversarially by failure mode:
  (a) high but fully forested summits (Haddington Hill, Kielder/Thetford interiors);
  (b) flat lowlands with huge theoretical viewsheds (Fens, Somerset Levels, Romney Marsh);
  (c) valley floors directly below famous viewpoints (below Coombe Hill, Mam Tor, Sutton Bank,
      the Malverns) — near-maximal score gap expected against their paired positives;
  (d) altitude-without-prospect: plateau interiors at the same elevation as nearby celebrated
      edges (Dunsmore 1.9 km from Coombe Hill, Kinder summit, Cranmere Pool, Cotswold high wold).
- Photo evidence: Geograph page per point where found; `photo_verified`/`photo_verdict` are
  filled by the visual photo-check pass (see below).

## Matching and pass conventions

A verification point matches the nearest scored candidate within **300 m** (verification
points are also seeded as candidates, so matches are near-guaranteed). Conventions, versioned
with `ALGORITHM_VERSION`:

- positive passes ⇔ national percentile ≥ 75 among candidates **or** regional (30 km)
  percentile ≥ 90 — candidates are already the top ~10% of England cells, so "average
  candidate" is a high bar;
- negative passes ⇔ national percentile ≤ 50.

Output: `outputs/verification_report.json` + the site's verification page.

## Photo verification

Every positive's Geograph photo is downloaded and visually checked (multi-agent pass):
does it show the Coombe Hill archetype — an elevated open viewpoint with a big drop and a
wide panorama over lower country — and is the pictured view attractive? Verdicts are stored
back into the dataset (`photo_verified`, `photo_verdict`). Photos showing the hill *from
afar* rather than the view *from* it are marked as weaker evidence, not failures.

## Honest-failure notes

Some celebrated viewpoints fail geometric ranking for reasons the metrics correctly report:
Lake District "balcony" viewpoints (Latrigg, Surprise View Derwentwater) are regionally
outcompeted by every surrounding fell — their fame mixes accessibility and composition with
geometry; Tan Hill Inn is a moorland pub, not a prospect; Malham Cove's 80 m amphitheatre
vanishes at 50 m DEM resolution. These are documented rather than tuned away.
