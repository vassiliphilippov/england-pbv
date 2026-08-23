"""Shared enums for the pipeline. Compare members, never raw values."""

from enum import Enum, IntEnum


class CandidateSource(Enum):
    SCREENING = "screening"
    OSM_VIEWPOINT = "osm_viewpoint"
    OSM_PEAK = "osm_peak"
    DOBIH_HILL = "dobih_hill"
    VERIFICATION = "verification"


class ViewKind(Enum):
    ESCARPMENT_EDGE = "escarpment-edge"
    ISOLATED_HILL_SUMMIT = "isolated-hill-summit"
    RIDGE = "ridge"
    GRITSTONE_EDGE = "gritstone-edge"
    COASTAL_CLIFF = "coastal-cliff"
    TOWER_MONUMENT = "tower/monument"
    OTHER = "other"
    NEGATIVE_CONTROL = "negative-control"


class LandCoverClass(IntEnum):
    """ESA WorldCover 2021 v200 class codes."""

    NODATA = 0
    TREE_COVER = 10
    SHRUBLAND = 20
    GRASSLAND = 30
    CROPLAND = 40
    BUILT_UP = 50
    BARE_SPARSE = 60
    SNOW_ICE = 70
    WATER = 80
    WETLAND = 90
    MANGROVES = 95
    MOSS_LICHEN = 100


LAND_COVER_LABELS: dict[LandCoverClass, str] = {
    LandCoverClass.NODATA: "unknown",
    LandCoverClass.TREE_COVER: "woodland",
    LandCoverClass.SHRUBLAND: "shrubland",
    LandCoverClass.GRASSLAND: "grassland",
    LandCoverClass.CROPLAND: "cropland",
    LandCoverClass.BUILT_UP: "built-up",
    LandCoverClass.BARE_SPARSE: "bare ground",
    LandCoverClass.SNOW_ICE: "snow/ice",
    LandCoverClass.WATER: "water",
    LandCoverClass.WETLAND: "wetland",
    LandCoverClass.MANGROVES: "mangroves",
    LandCoverClass.MOSS_LICHEN: "moss/lichen",
}
