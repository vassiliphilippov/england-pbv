# Potentially Beautiful Views of England

Automatic discovery of panoramic viewpoints in England from **objective, feedback-free terrain
metrics** — no human ratings, likes, or aesthetic models. The pipeline searches the whole country
for places like Coombe Hill Monument: elevated points with a steep drop, a wide unobstructed
horizon, long sightlines, and a diverse visible landscape — then publishes them as a static
website.

> A place is presented by its measurable **view potential** (openness, prospect, drop, depth,
> diversity, clearness), never by a claimed "beauty score". Whether the view is beautiful remains
> the visitor's judgement; the geometry that makes it possible is what we compute.

## How it works

1. **Data** — open datasets only: OS Terrain 50 (50 m national DEM), ESA WorldCover 10 m land
   cover, OpenStreetMap viewpoints/peaks/places, the Database of British and Irish Hills.
2. **Screening** — multi-scale Topographic Position Index and relief over the national grid pick
   candidate cells; non-maximum suppression thins them.
3. **Horizon sweep** — a numba engine casts 720 rays per candidate with Earth-curvature and
   refraction correction, producing the horizon profile, visible area by distance band, the
   angular (visual-magnitude) composition of the panorama, and land-cover diversity of the
   visible scene.
4. **Scoring** — each metric becomes a national percentile; six component scores average into a
   transparent "view potential" ranking, plus regional percentiles so a Chilterns escarpment can
   stand out without competing against the Lake District.
5. **Verification** — a curated dataset of 60 famous English viewpoints (photo-verified) and 20
   adversarial negative controls checks that the algorithm finds known great views and rejects
   forested summits, flat fens and valley floors. Current result: **49/60 positives and 17/20
   negatives pass**; Coombe Hill Monument scores in the 95th percentile nationally and beats its
   own vale floor by 58 view-potential points. See `specifications/verification_protocol.md`.
6. **Website** — a static site (map + per-viewpoint pages with horizon panoramas) is generated
   into `docs/` for GitHub Pages.

## Running

```bash
uv sync --locked --all-groups
uv run python -m england_pbv.data.download
uv run python -m england_pbv.pipeline.build_grid
uv run python -m england_pbv.pipeline.screening
uv run python -m england_pbv.pipeline.compute_metrics
uv run python -m england_pbv.pipeline.scoring
uv run python -m england_pbv.verification.evaluate
uv run python -m england_pbv.site.build
```

Quality gate: `uv run ruff format . && uv run ruff check . && uv run mypy src tests && uv run pytest`

## Project layout

- `src/england_pbv/` — the pipeline package
- `specifications/` — normative metric/data/pipeline specifications
- `research/` — background research on objective landscape metrics
- `verification/` — curated verification viewpoints dataset
- `docs/` — the generated website (GitHub Pages)
- `styleguide/` — Python style guide (normative)

Data sources are © their providers: OS Terrain 50 (OGL), ESA WorldCover (CC-BY 4.0),
OpenStreetMap (ODbL), DoBIH (CC-BY-SA). This project: Apache-2.0.
