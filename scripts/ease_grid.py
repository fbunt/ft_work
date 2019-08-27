import numpy as np
import pyproj


# EASE Grid 1.0
# Ref: https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html  # noqa: E501
# +------+-----------------+------+---------------+----------------------------------+-----------------------------------------+  # noqa: E501
# |      |   Projection    |      |  Dimensions   |              Origin              |               Grid Extent               |  # noqa: E501
# + Grid +       /         + EPSG +-------+-------+---------+--------+--------+------+---------+----------+----------+---------+  # noqa: E501
# | Name |   Resolution    | Code |  W    |  H    |  Col    |  Row   |  Lat   | Lon  | Min Lat | Max Lat  | Min Lon  | Max Lon |  # noqa: E501
# +------+-----------------+------+-------+-------+---------+--------+--------+------+---------+----------+----------+---------+  # noqa: E501
# | ML   | Global  25.0 km | 3410 | 1383  |  586  |  691.0  | 292.5  | 0.0    | 0.0  | 86.72S  | 86.72N   | 180.00W  | 180.00E |  # noqa: E501
# | MH   | Global  12.5 km | 3410 | 2766  | 1171  | 1382.0  | 585.0  | 0.0    | 0.0  | 85.95S  | 85.95N   | 179.93W  | 180.07E |  # noqa: E501
# | NL   | N. Hem. 25.0 km | 3408 |  721  |  721  |  360.0  | 360.0  | 90.0N  | 0.0  | 0.34S   | 90.00N   | 180.00W  | 180.00E |  # noqa: E501
# | NH   | N. Hem. 12.5 km | 3408 | 1441  | 1441  |  720.0  | 720.0  | 90.0N  | 0.0  | 0.26S   | 90.00N   | 180.00W  | 180.00E |  # noqa: E501
# | SL   | S. Hem. 25.0 km | 3409 |  721  |  721  |  360.0  | 360.0  | 90.0S  | 0.0  | 90.00S  | 0.34N    | 180.00W  | 180.00E |  # noqa: E501
# | SH   | S. Hem. 12.5 km | 3409 | 1441  | 1441  |  720.0  | 720.0  | 90.0S  | 0.0  | 90.00S  | 0.26N    | 180.00W  | 180.00E |  # noqa: E501
# +------+-----------------+------+-------+-------+---------+--------+--------+------+---------+----------+----------+---------+  # noqa: E501

# Shorthand name for each grid. Used to reference the grids throughout this
# module
ML = "ML"
MH = "MH"
NL = "NL"
NH = "NH"
SL = "SL"
SH = "SH"

GRID_NAMES = frozenset((ML, MH, NL, NH, SL, SH))

# ML
EASE1_ML_ROWS = 586
EASE1_ML_COLS = 1383
# NL
EASE1_NL_ROWS = 721
EASE1_NL_COLS = 721
# SL
EASE1_SL_ROWS = 721
EASE1_SL_COLS = 721

# Shapes of each grid
EASE1_ML_SHAPE = (EASE1_ML_ROWS, EASE1_ML_COLS)
EASE1_MH_SHAPE = (2 * EASE1_ML_ROWS, 2 * EASE1_ML_COLS)
EASE1_NL_SHAPE = (EASE1_NL_ROWS, EASE1_NL_COLS)
EASE1_NH_SHAPE = (2 * EASE1_NL_ROWS, 2 * EASE1_NL_COLS)
EASE1_SL_SHAPE = (EASE1_SL_ROWS, EASE1_SL_COLS)
EASE1_SH_SHAPE = (2 * EASE1_SL_ROWS, 2 * EASE1_SL_COLS)

GRID_NAME_TO_SHAPE = {
    ML: EASE1_ML_SHAPE,
    MH: EASE1_MH_SHAPE,
    NL: EASE1_NL_SHAPE,
    NH: EASE1_NH_SHAPE,
    SL: EASE1_SL_SHAPE,
    SH: EASE1_SH_SHAPE,
}

# Projections for each grid
EASE1_ML_PROJ = pyproj.Proj(init="epsg:3410")
EASE1_MH_PROJ = EASE1_ML_PROJ
EASE1_NL_PROJ = pyproj.Proj(init="epsg:3408")
EASE1_NH_PROJ = EASE1_NL_PROJ
EASE1_SL_PROJ = pyproj.Proj(init="epsg:3409")
EASE1_SH_PROJ = EASE1_SL_PROJ

GRID_NAME_TO_PROJ = {
    ML: EASE1_ML_PROJ,
    MH: EASE1_MH_PROJ,
    NL: EASE1_NL_PROJ,
    NH: EASE1_NH_PROJ,
    SL: EASE1_SL_PROJ,
    SH: EASE1_SH_PROJ,
}

# Earth radius in km
_RE_KM = 6371.228
# Nominal cell size in km
_CELL_KM = 25.067_525
# Scale factor for standard parallels at +/-30 degrees
_COS_PHI1 = 0.866_025_403


def _to_arrays(x, y):
    """Return values as arrays, indicate if both were scalars or if mismatched.
    """
    scalars = False
    xscalar = False
    yscalar = False
    if np.ndim(x) == 0:
        xscalar = True
        x = [x]
    if np.ndim(y) == 0:
        yscalar = True
        y = [y]
    mismatch = (True in (xscalar, yscalar)) and (False in (xscalar, yscalar))
    scalars = xscalar and yscalar
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return x, y, scalars, mismatch


def _check_array_shape(x, y):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match")
    return True


def _handle_inputs(x, y, grid_name):
    if grid_name not in GRID_NAMES:
        raise ValueError("Invalid grid name")
    x, y, scalars, mismatched = _to_arrays(x, y)
    if mismatched:
        raise TypeError(
            "Inputs must either both be scalars or both be arrays."
        )
    _check_array_shape(x, y)
    return x, y, scalars


def ease1_lonlat_to_rowcol_indices(lon, lat, grid_name=ML, to_int=False):
    """Convert lon/lat points to EASE-grid cell coordinates (indices).

    Inputs can be scalars or arrays, but the type must be consistent. If
    scalars are used as input, the results will also be scalars.

    Parameters:
        lon : scalar or array-like
            A single longitude point or an array of longitude points. Must
            match the shape of `lat`.
        lat : scalar or array-like
            A single latitude point or an array of latitude points. Must
            match the shape of `lon`.
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`
        to_int : boolean
            If `True`, results will be converted to integers.

    Returns:
        (rows, cols) : tuple of scalars or `numpy.ndarray`s
            The resulting grid coordinates (indices). Scalars are returned if
            the inputs were also scalars, otherwise arrays.

    References:
        * https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html  # noqa: E501
        * https://nsidc.org/sites/nsidc.org/files/files/data/pm/DiscreteGlobalGrids.pdf  # noqa: E501
        * ftp://sidads.colorado.edu/pub/tools/easegrid/geolocation_tools/

    """
    lon, lat, scalars = _handle_inputs(lon, lat, grid_name)

    nr, nc = GRID_NAME_TO_SHAPE[grid_name]
    if grid_name[-1] == "L":
        scale = 1.0
    else:
        scale = 2.0
    Rg = scale * _RE_KM / _CELL_KM
    r0 = (nc - 1) / 2.0 * scale
    s0 = (nr - 1) / 2.0 * scale

    phi = np.radians(lat)
    lam = np.radians(lon)
    r = np.full_like(lat, np.nan)
    s = np.full_like(lon, np.nan)
    # Use in-place operations to avoid extra allocations
    if grid_name in (ML, MH):
        # Global
        # r[:] = r0 + (Rg * lam * _COS_PHI1)
        r[:] = lam
        r *= Rg * _COS_PHI1
        r += r0
        # s[:] = s0 - (Rg * np.sin(phi) / _COS_PHI1)
        s = np.sin(phi, out=s)
        s *= -Rg / _COS_PHI1
        s += s0
    elif grid_name in (NL, NH):
        # North
        rho = 2.0 * Rg * np.sin((np.pi / 4) - (phi / 4))
        # r[:] = r0 + (rho * np.sin(lam))
        r = np.sin(lam, out=r)
        r *= rho
        r += r0
        # s[:] = s0 + (rho * np.cos(lam))
        s = np.cos(lam, out=s)
        s *= rho
        s += s0
    else:
        # South
        rho = 2.0 * Rg * np.cos((np.pi / 4) - (phi / 4))
        # r[:] = r0 + (rho * np.sin(lam))
        r = np.sin(lam, out=r)
        r *= rho
        r += r0
        # s[:] = s0 - (rho * np.cos(lam))
        s = np.cos(lam, out=s)
        s *= rho
        s = np.negative(s, out=s)
        s += s0
    if to_int:
        r = r.astype(int)
        s = s.astype(int)
    if scalars:
        rows = s.min()
        cols = r.min()
    else:
        rows = s
        cols = r
    return rows, cols


def ease1_rowcol_indices_to_lonlat(rows, cols, grid_name=ML):
    """Convert row/col indices to lon/lat values.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        rows : scalar or array-like
            A single row index or array of row indices. Must match the shape of
            `cols`
        cols : scalar or array-like
            A single col index or array of col indices. Must match the shape of
            `rows`
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of scalars or `numpy.ndarray`s
            The resulting lon/lat points. Scalars are returned if
            the inputs were also scalars, otherwise arrays.

    References:
        * https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html  # noqa: E501
        * https://nsidc.org/sites/nsidc.org/files/files/data/pm/DiscreteGlobalGrids.pdf  # noqa: E501
        * ftp://sidads.colorado.edu/pub/tools/easegrid/geolocation_tools/
    """
    rows, cols, scalars = _handle_inputs(rows, cols, grid_name)

    nr, nc = GRID_NAME_TO_SHAPE[grid_name]
    if grid_name[-1] == "L":
        scale = 1.0
    else:
        scale = 2.0
    Rg = scale * _RE_KM / _CELL_KM
    r = cols
    s = rows
    r0 = (nc - 1) / 2.0 * scale
    s0 = (nr - 1) / 2.0 * scale
    x = np.empty_like(r, dtype=float)
    x[:] = r
    y = np.empty_like(s, dtype=float)
    y[:] = s
    # In place
    # x = r - r0
    x -= r0
    # y = -(s - s0)
    y -= s0
    y = np.negative(y, out=y)

    lon = np.full_like(x, np.nan)
    lat = np.full_like(y, np.nan)
    if grid_name in (ML, MH):
        # Global grid
        beta = _COS_PHI1 / Rg * y
        eps = 1 + (0.5 / Rg)
        cond0 = np.abs(beta) > eps
        cond1 = beta <= -1
        cond2 = beta >= 1
        lat[cond0] = np.nan
        lat[cond1] = -np.pi / 2
        lat[cond2] = np.pi / 2
        cond = (~cond0) & (~(cond1 & cond2))
        lat[cond] = np.arcsin(beta[cond])
        lon[:] = x / _COS_PHI1 / Rg
        lat = np.degrees(lat, out=lat)
        lon = np.degrees(lon, out=lon)
    else:
        # North/South grids
        is_north = grid_name in (NL, NH)
        sinphi1 = np.sin(np.pi / 2.0)
        cosphi1 = np.cos(np.pi / 2.0)
        if not is_north:
            sinphi1 = np.sin(-np.pi / 2.0)
            cosphi1 = np.cos(-np.pi / 2.0)
        rho = np.sqrt((x * x) + (y * y))
        lam = np.full_like(lon, np.nan)
        cond_y_eq_0 = y == 0.0
        cond_y_ne_0 = ~cond_y_eq_0
        branch1 = cond_y_eq_0 & (r <= r0)
        branch2 = cond_y_eq_0 & (r > r0)
        branch3 = cond_y_ne_0
        lam[branch1] = -np.pi / 2.0
        lam[branch2] = np.pi / 2.0
        if is_north:
            lam[branch3] = np.arctan2(x[branch3], -y[branch3])
        else:
            lam[branch3] = np.arctan2(x[branch3], y[branch3])
        gamma = rho / (2 * Rg)
        gamma[np.abs(gamma) > 1] = np.nan
        v = np.arcsin(gamma, out=gamma)
        v *= 2.0
        beta = (np.cos(v) * sinphi1) + (y * np.sin(v) * (cosphi1 / rho))
        beta[np.abs(beta) > 1] = np.nan
        phi = np.arcsin(beta)
        lat[:] = np.degrees(phi)
        lon[:] = np.degrees(lam)
        cond_rho_eq_0 = rho == 0.0
        lat[cond_rho_eq_0] = 90.0 if is_north else -90.0
        lon[cond_rho_eq_0] = 0.0
    if scalars:
        # Return scalars if the inputs were scalars
        lon = lon.min()
        lat = lat.min()
    return lon, lat


def ease1_lonlat_to_meters(lon, lat, grid_name=ML):
    """Reproject lon/lat points into the specified EASE-grid projection in
    meters.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        lon : scalar or array-like
            A single longitude point or an array of longitude points. Must
            match the shape of `lat`.
        lat : scalar or array-like
            A single latitude point or an array of latitude points. Must
            match the shape of `lon`.
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (xm, ym) : tuple of scalars or `numpy.ndarray`s
            The resulting x/y projection points in meters. Scalars are returned
            if the inputs were also scalars, otherwise arrays.

    References:
        * https://nsidc.org/ease/ease-grid-projection-gt
    """
    lon, lat, scalars = _handle_inputs(lon, lat, grid_name)

    xm, ym = GRID_NAME_TO_PROJ[grid_name](lon, lat)
    if scalars:
        xm = xm.min()
        ym = ym.min()
    return xm, ym


def ease1_meters_to_lonlat(xm, ym, grid_name=ML):
    """Convert x/y points (in meters) from the specified EASE-grid projection
    to lon/lat points.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        xm : scalar or array-like
            A single EASE-grid x coordinate in meters or an array of points.
            Must match the shape of `ym`.
        ym : scalar or array-like
            A single EASE-grid y coordinate in meters or an array of points.
            Must match the shape of `xm`.
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of scalars or `numpy.ndarray`s
            The resulting lon/lat points. Scalars are returned if the inputs
            were also scalars, otherwise arrays.

    References:
        * https://nsidc.org/ease/ease-grid-projection-gt
    """
    xm, ym, scalars = _handle_inputs(xm, ym, grid_name=ML)

    lon, lat = GRID_NAME_TO_PROJ[grid_name](xm, ym, inverse=True)
    if scalars:
        lon = lon.min()
        lat = lat.min()
    return lon, lat
