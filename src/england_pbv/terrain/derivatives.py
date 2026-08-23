"""Multi-scale terrain derivatives on the national grid.

Box windows (scipy uniform filter, separable running sums) approximate circular
neighbourhoods; radii are converted to half-window sizes in cells. Adequate for
screening — final metrics come from the horizon sweep, not from these derivatives.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from england_pbv.constants import GRID_CELL_M


def box_mean(values: NDArray[np.float32], radius_m: float) -> NDArray[np.float32]:
    half = max(1, int(round(radius_m / GRID_CELL_M)))
    result: NDArray[np.float32] = uniform_filter(
        values,
        size=2 * half + 1,
        mode="nearest",
    )
    return result


def tpi(dem: NDArray[np.float32], radius_m: float) -> NDArray[np.float32]:
    result: NDArray[np.float32] = dem - box_mean(dem, radius_m=radius_m)
    return result
