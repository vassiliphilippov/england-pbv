"""External links and OS grid reference formatting for site pages."""

GRID_LETTERS: str = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no I


def os_grid_ref(easting: float, northing: float) -> str:
    """E.g. Coombe Hill (484930, 206690) -> 'SP 84930 06690'."""
    e, n = int(round(easting)), int(round(northing))
    e100, n100 = e // 100000, n // 100000
    first_index = (19 - n100) - (19 - n100) % 5 + (e100 + 10) // 5
    second_index = (19 - n100) * 5 % 25 + e100 % 5
    letters = GRID_LETTERS[first_index] + GRID_LETTERS[second_index]
    return f"{letters} {e % 100000:05d} {n % 100000:05d}"


def google_maps_url(lat: float, lon: float) -> str:
    return f"https://maps.google.com/?q={lat:.5f},{lon:.5f}"


def osmaps_url(lat: float, lon: float) -> str:
    return f"https://explore.osmaps.com/pin?lat={lat:.5f}&lon={lon:.5f}&zoom=15"


def peakfinder_url(lat: float, lon: float) -> str:
    return f"https://www.peakfinder.com/?lat={lat:.5f}&lng={lon:.5f}"


def geograph_square_url(easting: float, northing: float) -> str:
    ref = os_grid_ref(easting, northing).replace(" ", "")
    # 1 km square reference: letters + 2-digit easting + 2-digit northing
    return f"https://www.geograph.org.uk/gridref/{ref[:2]}{ref[2:4]}{ref[7:9]}"


def openstreetmap_url(lat: float, lon: float) -> str:
    return (
        f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=15/{lat:.5f}/{lon:.5f}"
    )
