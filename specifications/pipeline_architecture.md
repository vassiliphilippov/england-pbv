# Pipeline Architecture

Stages are `python -m` modules; each reads/writes files under `data/` and `outputs/`
(both gitignored, fully reproducible). Paths come only from `england_pbv.paths`.

```
data.download          raw datasets -> data/raw/
pipeline.build_grid    national 50 m grids -> data/grids/ (DEM float32, landcover uint8, England mask)
pipeline.screening     multi-scale TPI top-10% + 150 m NMS + named-point seeding -> outputs/candidates.jsonl
pipeline.compute_metrics  numba horizon sweep (720 az x 259 samples) -> outputs/view_metrics.jsonl
pipeline.scoring       frozen percentile components + composite -> outputs/scored_viewpoints.jsonl
pipeline.refine        2-pass local grid search for top 8k spots -> updates candidates.jsonl
                       (then compute_metrics + scoring run again on refined positions)
verification.evaluate  curated famous viewpoints + negative controls -> outputs/verification_report.json
site.build             static website -> docs/ (GitHub Pages)
```

## Engine

`viewshed/horizon.py` — a numba-parallel radial sweep. Per observer and bearing it tracks the
running max elevation angle over bare terrain and over an obstacle surface (trees +15 m,
built +8 m); record increments yield exact angular (visual-magnitude) areas, so distance-band
and land-cover composition metrics are by-products of the same pass. Earth curvature and
refraction are folded into effective elevations. Throughput ≈ 2.8–5.5k observers/s on 14
cores against the 1.4 GB national DEM (full England candidate set ≈ 15 s; refinement of
8k spots × 162 positions ≈ 10 min).

Correctness anchors (tests/test_horizon_engine.py):
- flat-plain far-distance equals the physical horizon for a 1.7 m eye (√(2·R_eff·h) ≈ 5.0 km);
- escarpment edge sees far on exactly one side with the right arc;
- an enclosing ridge blocks everything;
- land-cover composition and Shannon diversity behave on synthetic worlds;
- summit woodland kills the leaf-on view while scarp-slope woodland below the lip does not.

## Design decisions

- **GB-wide DEM, England-only candidates**: views from Shropshire into Wales need Welsh
  terrain; the England mask (ONS boundary, 2-cell coastal dilation) gates candidates only.
- **Candidates, not exhaustive cells**: screening keeps the top 10% of England cells by
  multi-scale standardized TPI (500 m / 2 km / 10 km), NMS-thinned, then seeds every OSM
  viewpoint/peak, DoBIH hill (drop ≥ 30 m) and verification point so nothing famous is missed
  and results carry names.
- **Metrics before ranking**: the full metric vector is stored per candidate; scoring is a
  cheap, re-runnable convention layer on top.
- **Frozen percentiles**: refinement and any future incremental scoring reuse the same
  national distributions, so scores mean the same thing everywhere.
- **Site pages re-run the engine** for their horizon panoramas rather than persisting
  per-candidate profiles nationally (2 s for 300 pages beats 400 MB of stored arrays).
