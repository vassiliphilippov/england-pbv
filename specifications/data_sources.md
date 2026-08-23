# Data Sources

All endpoints verified live on 2026-08-23. Every dataset is open; attribution requirements
listed per source. Downloads land in `data/raw/` via `england_pbv.data.download`.

## Used by the pipeline (v0.1)

| Dataset | Role | Endpoint | Size | Licence |
|---|---|---|---|---|
| **OS Terrain 50** | National 50 m bare-earth DEM (all GB) | `https://api.os.uk/downloads/v1/products/Terrain50/downloads?area=GB&format=ASCII+Grid+and+GML+%28Grid%29&redirect` (307 → signed Azure URL, ~1 h expiry; always start from api.os.uk) | 162 MB zip | OGL. "Contains OS data © Crown copyright and database right 2026" |
| **ESA WorldCover 2021 v200** | 10 m land cover (composition, tree/built obstruction) | `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif`, 8 tiles for GB (N54E000 is sea → 404 by design) | 413 MB | CC-BY 4.0 ESA WorldCover |
| **OSM via Overpass** | Named viewpoints (`tourism=viewpoint`, 3084), peaks (`natural=peak`, 7545), places (city/town/village) for naming | `https://overpass-api.de/api/interpreter`, England = area `3600058447`; needs a custom User-Agent (default python UA gets HTTP 406) | ~5 MB | ODbL |
| **DoBIH** | Surveyed hill summits + prominence (candidate seeding, naming) | `https://www.hill-bagging.co.uk/dobih-downloads/hillcsv.zip` (canonical, rejects some networks) with fallback `https://www.hills-database.co.uk/hillcsv.zip` | 2.3 MB | CC-BY 4.0, attribute "Database of British and Irish Hills" |
| **ONS Countries Dec 2024 (BGC)** | England boundary mask (20 m generalised) | ONS ArcGIS FeatureServer, `Countries_December_2024_Boundaries_UK_BGC`, `CTRY24NM='England'`, GeoJSON | 4.3 MB | OGL |
| **ScenicOrNot dump** | Archived for future square-level validation (not used in scoring) | `http://scenicornot.datasciencelab.co.uk/votes.tsv` | 19.5 MB | ODbL (per current host FAQ) |
| **NE Priority Habitats Inventory** | Where moorland vegetation actually is (render palette: moor grass / bracken / heather flags on the 10 m grid) | ArcGIS FeatureServer `https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/Priority_Habitats_Inventory_England/FeatureServer/0/query`, native EPSG:27700, field `MainHabs`, 2000-feature pages, `f=geojson` | fetched per site (~3 km envelopes) | OGL, © Natural England |
| **OSM footpaths (calibration sites)** | Worn-path lines in renders (depth cue) | Overpass, `highway~path\|footway\|track\|bridleway\|steps` within 2.5 km of each calibration site (mirror: `overpass.kumi.systems`) | ~6 MB | ODbL |
| **OSM rock features (calibration sites)** | Grey crag/tor/scree material (+3 flag) | Overpass, `natural~bare_rock\|scree\|cliff\|rock\|stone\|shingle` within 3 km of each site | ~2 MB | ODbL |
| **Sentinel-2 L2A true colour (AWS Open Data)** | Real far-field ground texture (`england_satellite_10m.npy`, 10 m RGB on the BNG frame; renderer blends it in beyond 2 km) | STAC `https://earth-search.aws.element84.com/v1/search`, collection `sentinel-2-l2a`, anonymous, per-scene 8-bit `TCI.tif` COGs on `sentinel-cogs.s3.us-west-2.amazonaws.com`; least-cloudy May–Sep scene per MGRS tile (66 tiles) | ~17 GB fetched → 11.4 GB grid | Copernicus (CC BY-SA 3.0 IGO). Site must credit "Contains modified Copernicus Sentinel data [year]" |

### OS Terrain 50 archive structure (verified)

Outer zip: `data/<2-letter-100km-square>/<tile>_OST50GRID_<release-date>.zip` (2,858 nested
zips, glob the date — it changes per release). Each nested zip contains `<TILE>.asc`: ESRI
ASCII grid, 200×200 cells, `cellsize 50`, `xllcorner/yllcorner` in EPSG:27700 metres, top
row northernmost, **no NODATA line** — sea cells hold real small negative values and
fully-offshore squares are simply absent (treat as sea level).

## Phase-2 refinement sources (verified, not yet wired in)

**EA LiDAR on Defra DSP** — the old ArcGIS GPServer API is dead; the current flow:

1. *(optional)* POST bare GeoJSON polygon (WGS84) to
   `https://environment.data.gov.uk/backend/catalog/api/tiles/collections/survey/search`
   with `Content-Type: application/geo+json` → products/tiles list.
2. GET `https://environment.data.gov.uk/tiles/collections/survey/{product}/{year}/{res}/{tileId}?subscription-key=dspui`
   — tile IDs are deterministic: `{100km letters}{easting km floored to 5}{northing km floored to 5}`
   (Coombe Hill → `SP8005`). Products: `lidar_composite_dtm`, `lidar_composite_first_return_dsm`;
   year `2022`, res `1` or `2`. ~65–75 MB per 5 km tile at 1 m. Invalid combos return HTTP 500.
3. **10 m national DTM** (5.3 GB single zip): GET
   `https://environment.data.gov.uk/file-management-open/data-sets/ac3682a1-1ead-4478-8a2d-d8402ef9ead0/files/LIDAR_Composite_10m_DTM_2022.zip/download-url`
   → JSON with a ~24 h signed URL (range-resumable). England-only coverage — cannot replace the
   GB-wide grid (border viewsheds into Wales/Scotland need terrain there), but can sharpen the
   England near-field.
4. **WCS clips** (best for per-viewpoint 1 m windows):
   `https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs?service=WCS&version=2.0.1&request=GetCoverage&coverageId=13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m&SUBSET=E(min,max)&SUBSET=N(min,max)&FORMAT=image/tiff`
   (same pattern for the First Return DSM, coverageId
   `df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m`). Uncompressed
   output; keep requests ≤4000×4000 px; transient 500s — retry.

All EA endpoints are reverse-engineered from the live app (undocumented, `subscription-key=dspui`);
expect to re-verify when scripts break. Licence OGL, "© Environment Agency copyright".

**Natural England CRoW Access Layer** (USED since 2026-08-24 by `england_pbv.data.access`:
public-access flags with OSM `designation`-tagged rights of way; see access classes in that
module's docstring): ArcGIS Hub bulk GeoJSON
`https://hub.arcgis.com/api/v3/datasets/6ce15f2cd06c4536983d315694dad16b_0/downloads/data?format=geojson&spatialRefId=4326`
(42,059 polygons; OGL).

**Geofabrik England extract** (bulk OSM alternative to Overpass):
`https://download.geofabrik.de/europe/united-kingdom/england-latest.osm.pbf` (~1.6 GB, daily).

## Rejected options and why

- **UKCEH Land Cover Map**: better classes than WorldCover but non-OGL licensing for the 10 m
  raster; WorldCover is CC-BY and sufficient for composition metrics.
- **Google Street View as validation corpus**: Maps Platform terms prohibit bulk download and
  ML-validation use. Geograph is the validation corpus instead.
- **Copernicus GLO-30**: 30 m global fallback only; OS Terrain 50 + EA LiDAR dominate it in GB.
