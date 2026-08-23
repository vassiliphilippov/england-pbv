"""Build the national 50 m grids: DEM, land cover, England mask.

Run: uv run python -m england_pbv.pipeline.build_grid [--force]
"""

import argparse

from england_pbv import paths
from england_pbv.data.boundary import build_england_mask
from england_pbv.data.terrain50 import build_dem_grid
from england_pbv.data.worldcover import build_landcover_grid
from england_pbv.terrain.grid import save_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build national grids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths.ensure_dirs()

    if args.force or not paths.DEM_GRID_NPY.exists():
        print("building DEM grid from OS Terrain 50...")
        dem = build_dem_grid(paths.TERRAIN50_ZIP)
        save_grid(paths.DEM_GRID_NPY, dem)
        print(f"saved {paths.DEM_GRID_NPY} ({dem.nbytes / 1e9:.2f} GB)")
        del dem
    else:
        print("DEM grid exists, skipping")

    if args.force or not paths.LANDCOVER_GRID_NPY.exists():
        print("building land-cover grid from ESA WorldCover...")
        landcover = build_landcover_grid(paths.WORLDCOVER_DIR)
        save_grid(paths.LANDCOVER_GRID_NPY, landcover)
        print(f"saved {paths.LANDCOVER_GRID_NPY} ({landcover.nbytes / 1e9:.2f} GB)")
        del landcover
    else:
        print("land-cover grid exists, skipping")

    if args.force or not paths.ENGLAND_MASK_NPY.exists():
        print("building England mask from ONS boundary...")
        mask = build_england_mask(paths.ENGLAND_BOUNDARY_GEOJSON)
        save_grid(paths.ENGLAND_MASK_NPY, mask)
        print(f"saved {paths.ENGLAND_MASK_NPY}; England cells: {int(mask.sum())}")
    else:
        print("England mask exists, skipping")


if __name__ == "__main__":
    main()
