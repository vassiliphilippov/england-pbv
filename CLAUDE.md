# england-pbv — Potentially Beautiful Views of England

Objective, feedback-free discovery of panoramic viewpoints in England from open terrain and
land-cover data, published as a static website.

## What this project does

1. Downloads open datasets (OS Terrain 50 DEM, ESA WorldCover land cover, OSM viewpoints/peaks,
   DoBIH hills) into `data/` — see `england_pbv.data.download`.
2. Screens the whole of England for candidate viewpoints using multi-scale terrain metrics
   (TPI, relief, drop) on a 50 m British National Grid raster.
3. Runs a numba horizon-sweep viewshed engine on every candidate: horizon profile, visible area
   by distance band, angular (visual-magnitude) composition of the panorama, openness, skyline
   metrics, land-cover diversity of the visible scene.
4. Scores candidates by percentile-normalised components (never "beauty" — "view potential"),
   nationally and regionally.
5. Verifies the ranking against a curated dataset of famous English viewpoints
   (`verification/viewpoints.json`) and adversarial negative controls.
6. Builds a static website into `docs/` (GitHub Pages).

## Commands

Everything runs through `uv`:

```bash
uv sync --locked --all-groups
uv run ruff format . && uv run ruff check .
uv run mypy src tests
uv run pytest
```

Pipeline stages (each is a `python -m` module, in order):

```bash
uv run python -m england_pbv.data.download          # fetch datasets into data/
uv run python -m england_pbv.pipeline.build_grid    # national 50 m DEM + land cover grids
uv run python -m england_pbv.pipeline.screening     # candidate generation
uv run python -m england_pbv.pipeline.compute_metrics  # horizon sweep on candidates
uv run python -m england_pbv.pipeline.scoring       # percentiles + composite scores
uv run python -m england_pbv.verification.evaluate  # verification report
uv run python -m england_pbv.site.build             # static site into docs/
```

Optional: `uv run python -m england_pbv.data.vote_photos` refreshes
`verification/vote_photos.json` (free-licensed view photos for the pair-voting page).

## Conventions

- Python 3.12, `uv`-managed. mypy `--strict` clean, ruff clean, pytest green before finishing.
- Style guide: `styleguide/python_styleguide.md` is normative (dataclasses over tuples, keyword
  arguments, enums over magic strings, `None` for missing data, absolute imports, paths only
  from `england_pbv.paths`).
- Specifications for metrics/data/pipeline live in `specifications/` and are the source of truth
  for what each metric means. Update the spec in the same change that changes a metric.
- `data/` and `outputs/` are gitignored (large, reproducible). `docs/` (the built site) and
  `verification/` (curated dataset) are committed.
- Scores are labelled "view potential", never "beauty". Raw metric components must stay visible
  in every ranking artifact.
