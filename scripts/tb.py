import numpy as np
import os
import pyproj
import re


EASE_ROWS = 586
EASE_COLS = 1383


def load_flat_binary_file(fname, nrows, ncols, dtype):
    return np.fromfile(fname, dtype=dtype).reshape((nrows, ncols))


def load_tb_file(fname):
    # Data is stored as binary integers. Divide by 10 to recover float values
    return load_flat_binary_file(fname, EASE_ROWS, EASE_COLS, "<h") / 10.0


# Earth radius in km
_EASE_R_KM = 6371.228
# Nominal cell size in km
_EASE_C_KM = 25.067_525
# Scale factor for standard parallels at +/-30 degrees
_EASE_COS_PHI1 = 0.866_025_403


def ease_inverse(r, s):
    """Convert from cylindrical global EASE Grid 25 km cell cylindrical
    coordinates back to lon/lat.

    ref: https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html
    ref: ftp://sidads.colorado.edu/pub/tools/easegrid/geolocation_tools/
    """
    R = _EASE_R_KM
    C = _EASE_C_KM
    r0 = (EASE_COLS - 1) / 2.0
    s0 = (EASE_ROWS - 1) / 2.0
    if not isinstance(r, (np.ndarray, list, tuple)):
        r = [r]
    if not isinstance(r, (np.ndarray, list, tuple)):
        s = [s]
    r = np.array(r, dtype=np.float)
    s = np.array(s, dtype=np.float)
    x = r
    # In place
    x -= r0
    y = s
    y -= s0
    y = np.negative(y, out=y)
    cos30 = np.cos(np.radians(30))

    beta = (cos30 * C / R) * y
    eps = 1 + (0.5 * C / R)
    lat = np.empty_like(beta)
    cond0 = np.abs(beta) > eps
    cond1 = beta <= -1
    cond2 = beta >= 1
    lat[cond0] = np.nan
    lat[cond1] = -np.pi / 2
    lat[cond2] = np.pi / 2
    c = (~cond0) & (~(cond1 & cond2))
    lat[c] = np.arcsin(beta[c])
    lon = (C / cos30 / R) * x
    lat = np.degrees(lat, out=lat)
    lon = np.degrees(lon, out=lon)
    return lon, lat


def ease_convert(lon, lat):
    """Convert lon/lat points to EASE Grid grid-cell cylindrical coordinates.

    ref: https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html
    ref: ftp://sidads.colorado.edu/pub/tools/easegrid/geolocation_tools/
    """
    Rg = _EASE_R_KM / _EASE_C_KM
    r0 = (EASE_COLS - 1) / 2.0
    s0 = (EASE_ROWS - 1) / 2.0
    phi = np.radians(lat)
    lam = np.radians(lon)

    r = r0 + (Rg * _EASE_COS_PHI1 * lam)
    s = s0 - (Rg / _EASE_COS_PHI1 * np.sin(phi))
    return r, s


# EASE Grid 2.0 global, equal-area, 25 km nominal grid size
# https://epsg.io/3410
EASE_PROJ = pyproj.Proj(init="epsg:3410")


def ease_inverse_meters(rmeters, smeters):
    """Convert EASE Grid projection coordinates in meters to lon/lat points.

    ref: https://nsidc.org/ease/ease-grid-projection-gt
    """
    return EASE_PROJ(rmeters, smeters, inverse=True)


def ease_convert_meters(lon, lat):
    """Convert lon/lat points to global EASE projection in meters.

    ref: https://nsidc.org/ease/ease-grid-projection-gt
    """
    return EASE_PROJ(lon, lat)


# Regex for parsing EASE grid file names of the form
# 'EASE-III-MMYYYYDDDP.FFG' or 'EASE-III-MMYYYYDDDP-V2.FFG'.
#  III: satellite ID (or just SSMI for filled data)
#   MM: EASE projection grid (ML, MH, etc.)
# YYYY: year
#  DDD: day of year
#    P: A or D for ascending or descending pass
#   FF: Frequency, 19/22/37/91 GHz
#    G: polarization, V or H for vertical/horizontal
EASE_FNAME_PAT = re.compile(
    "^EASE-(SSMI|[A-Z\\d]+)-(ML|MH|NL|NH|SL|SH)(\\d{4})(\\d{3})(A|D)"
    "(?:-V2)?\\.(\\d+)(V|H)$"
)
