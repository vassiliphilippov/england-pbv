# Site interactivity: drive-time estimates, Pareto ladders, page tiers, pair voting

This spec covers the presentation-layer features added on top of the scored dataset. Nothing
here feeds back into scoring; scores remain the percentile composites defined in
`metrics_specification.md`.

## Drive-time estimate

`england_pbv.site.traveltime` estimates car travel time from straight-line distance:

```
minutes = 2.61 * 1.15 * crow_km ** 0.809
```

- `2.61 * d^0.809` was fitted against 8,008 OSRM car-route durations between random pairs of
  England points (R² = 0.965, median absolute error 5.8% vs OSRM free-flow). The power law
  absorbs both circuity (road/crow distance ≈ 1.6 for short hops falling to ≈ 1.25 for long
  trips; England's published average is ≈ 1.4) and average speed rising with trip length
  (local roads → motorways), consistent with DfT speed statistics.
- `1.15` converts free-flow durations to typical driving conditions.
- The identical formula is emitted to `docs/assets/traveltime.js` at build time, so Python
  build artifacts and client-side JavaScript can never disagree.
- The postcode page refines ladder times with one request to the FOSSGIS OSRM instance
  (`routing.openstreetmap.de`, CORS-enabled, ≤ 1 request per page view) and falls back to the
  formula when offline. Estimated times are always prefixed "~"; road-routed times are not.

## Pareto ladder ("more beautiful places nearby" / postcode ladder)

Given an origin, sort candidates by (estimated) drive minutes; a candidate joins the ladder
when its view-potential score exceeds the best score of anything closer by more than a minimum
gain. The ladder is therefore ordered nearest→farthest with strictly increasing scores and
always ends at the best reachable view. When a ladder exceeds the row budget the minimum gain
is widened (never truncating the far end).

- Viewpoint pages: build-time ladder over all deduplicated candidates that have pages,
  minimum gain 0.1, at most 8 rows (`_nearby_frontier` in `site/build.py`).
- Postcode page: client-side ladder over all `ALL_POINTS`, minimum gain 0.5, at most 15 rows,
  gain widened ×1.6 until the ladder fits.

## Page tiers

| Tier | Selection | Content |
| --- | --- | --- |
| national | top 600 nationally | full: 360° panorama + 2 camera views |
| regional | top 5 per 25 km cell | full |
| lite | top 2 per 10 km cell | panorama + 1 camera view, JPEG q78 |
| micro | top 3 per 5 km cell | panorama only, downscaled to 960 px, JPEG q62, no camera views |

Tiers are cumulative (a point already paged is not re-selected); the micro tier takes the site
past 10,000 pages while staying inside GitHub Pages' 1 GB published-site budget.

## Map auto-update (index page)

The "best views in this area" list follows the map viewport. `moveend` triggers a trailing
debounce whose delay adapts to the measured update cost (VS Code-style adaptive debounce):
`delay = clamp(150 ms, 2 × EMA(update duration), 1000 ms)` with EMA α = 0.3. A move after a
quiet spell updates immediately (leading edge). The 150 ms floor sits at the threshold of
perceived instantaneity; the 1 s ceiling is the classic flow-of-thought limit.

## Pair voting (vote.html)

`verification/vote_photos.json` (built by `england_pbv.data.vote_photos`) holds free-licensed
photographs of English views with camera coordinates, spread across areas and across the full
range of scenic quality. The vote page shows random pairs (least-shown photos first, random
partners); each vote `{a, b, w, t}` is stored in `localStorage` only. Users export their votes
as JSON and contribute them via GitHub. The collected votes are intended to fit a preference
model over the objective metric components at each photo's location, and then to re-weight the
view-potential composite; until that happens votes have no effect on any score.
