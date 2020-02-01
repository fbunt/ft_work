import numpy as np
import pyproj
import re


################
# EASE Grid V1 #
################

# Ref: https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html  # noqa: E501
# Ref: https://nsidc.org/ease/ease-grid-projection-gt
# +------+-----------------+------+---------------+----------------------------------+-----------------------------------------+  # noqa: E501
# |      |   Projection    |      |  Dimensions   |              Origin              |               Grid Extent               |  # noqa: E501
# | Grid |       /         | EPSG +-------+-------+---------+--------+--------+------+---------+----------+----------+---------+  # noqa: E501
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
V1_ML_ROWS = 586
V1_ML_COLS = 1383
# NL
V1_NL_ROWS = 721
V1_NL_COLS = 721
# SL
V1_SL_ROWS = 721
V1_SL_COLS = 721

# Shapes of each grid
V1_ML_SHAPE = (V1_ML_ROWS, V1_ML_COLS)
V1_MH_SHAPE = (2 * V1_ML_ROWS, 2 * V1_ML_COLS)
V1_NL_SHAPE = (V1_NL_ROWS, V1_NL_COLS)
V1_NH_SHAPE = (2 * V1_NL_ROWS, 2 * V1_NL_COLS)
V1_SL_SHAPE = (V1_SL_ROWS, V1_SL_COLS)
V1_SH_SHAPE = (2 * V1_SL_ROWS, 2 * V1_SL_COLS)

GRID_NAME_TO_V1_SHAPE = {
    ML: V1_ML_SHAPE,
    MH: V1_MH_SHAPE,
    NL: V1_NL_SHAPE,
    NH: V1_NH_SHAPE,
    SL: V1_SL_SHAPE,
    SH: V1_SH_SHAPE,
}
GRID_NAME_TO_V1_PROJ_CODE = {
    ML: "EPSG:3410",
    MH: "EPSG:3410",
    NL: "EPSG:3408",
    NH: "EPSG:3408",
    SL: "EPSG:3410",
    SH: "EPSG:3409",
}
# Projections for each grid
V1_ML_PROJ = pyproj.Proj(GRID_NAME_TO_V1_PROJ_CODE[ML])
V1_MH_PROJ = V1_ML_PROJ
V1_NL_PROJ = pyproj.Proj(GRID_NAME_TO_V1_PROJ_CODE[NL])
V1_NH_PROJ = V1_NL_PROJ
V1_SL_PROJ = pyproj.Proj(GRID_NAME_TO_V1_PROJ_CODE[SL])
V1_SH_PROJ = V1_SL_PROJ

GRID_NAME_TO_V1_PROJ = {
    ML: V1_ML_PROJ,
    MH: V1_MH_PROJ,
    NL: V1_NL_PROJ,
    NH: V1_NH_PROJ,
    SL: V1_SL_PROJ,
    SH: V1_SH_PROJ,
}

# Earth radius in km
_RE_KM = 6371.228
# Nominal cell size in km
_V1_CELL_KM = 25.067_525
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


def _validate_array_shape(x, y):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match")
    return True


def _validate_grid_name(name):
    if name not in GRID_NAMES:
        raise ValueError("Invalid grid name")


def _handle_inputs(x, y, grid_name):
    _validate_grid_name(grid_name)
    x, y, scalars, mismatched = _to_arrays(x, y)
    if mismatched:
        raise TypeError(
            "Inputs must either both be scalars or both be arrays."
        )
    _validate_array_shape(x, y)
    return x, y, scalars


def v1_lonlat_to_colrow_coords(lon, lat, grid_name=ML, to_int=False):
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
            If `True`, results will be rounded and converted to integers.

    Returns:
        (cols, rows) : tuple of scalars or `numpy.ndarray`s
            The resulting grid coordinates (indices). Scalars are returned if
            the inputs were also scalars, otherwise arrays.

    References:
        * https://web.archive.org/web/20190217144552/https://nsidc.org/data/ease/ease_grid.html  # noqa: E501
        * https://nsidc.org/sites/nsidc.org/files/files/data/pm/DiscreteGlobalGrids.pdf  # noqa: E501
        * ftp://sidads.colorado.edu/pub/tools/easegrid/geolocation_tools/

    """
    lon, lat, scalars = _handle_inputs(lon, lat, grid_name)

    nr, nc = GRID_NAME_TO_V1_SHAPE[grid_name]
    if grid_name[-1] == "L":
        scale = 1.0
    else:
        scale = 2.0
    Rg = scale * _RE_KM / _V1_CELL_KM
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
        rho = 2.0 * Rg * np.sin((np.pi / 4) - (phi / 2))
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
        rho = 2.0 * Rg * np.cos((np.pi / 4) - (phi / 2))
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
        r = np.round(r, out=r).astype(int)
        s = np.round(s, out=s).astype(int)
    if scalars:
        rows = s.min()
        cols = r.min()
    else:
        rows = s
        cols = r
    return cols, rows


def v1_colrow_coords_to_lonlat(cols, rows, grid_name=ML):
    """Convert row/col coords (indices) to lon/lat values.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        rows : scalar or array-like
            A single row index or array of row coords. Must match the shape of
            `cols`
        cols : scalar or array-like
            A single col index or array of col coords. Must match the shape of
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
    cols, rows, scalars = _handle_inputs(cols, rows, grid_name)

    nr, nc = GRID_NAME_TO_V1_SHAPE[grid_name]
    if grid_name[-1] == "L":
        scale = 1.0
    else:
        scale = 2.0
    Rg = scale * _RE_KM / _V1_CELL_KM
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
        beta = np.full_like(rho, np.nan)
        cond = rho != 0
        beta[cond] = (np.cos(v[cond]) * sinphi1) + (
            y[cond] * np.sin(v[cond]) * (cosphi1 / rho[cond])
        )
        # Prevent runtime warning from nan's in comparison
        # Use array view property of ndarray
        beta_not_nan = beta[~np.isnan(beta)]
        beta_not_nan[np.abs(beta_not_nan) > 1] = np.nan
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


def v1_lonlat_to_meters(lon, lat, grid_name=ML):
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

    xm, ym = GRID_NAME_TO_V1_PROJ[grid_name](lon, lat)
    if scalars:
        xm = xm.min()
        ym = ym.min()
    return xm, ym


def v1_meters_to_lonlat(xm, ym, grid_name=ML):
    """Convert x/y points (in meters) from the specified EASE-grid projection
    to lon/lat points.

    Parameters:
        xm : scalar or array-like
            A single EASE-grid x coordinate in meters or an array of points.
        ym : scalar or array-like
            A single EASE-grid y coordinate in meters or an array of points.
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of scalars or `numpy.ndarray`s
            The resulting lon/lat points.

    References:
        * https://nsidc.org/ease/ease-grid-projection-gt
    """
    _validate_grid_name(grid_name)
    return GRID_NAME_TO_V1_PROJ[grid_name](xm, ym, inverse=True)


def v1_meters_to_colrow_coords(xm, ym, grid_name="ML", to_int=False):
    """Convert EASE-grid projected coordinates in meters to EASE-grid cell
    coordinates (indices).

    Inputs can be scalars or arrays, but the type must be consistent. If
    scalars are used as input, the results will also be scalars.

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
        to_int : boolean
            If `True`, results will be rounded and converted to integers.

    Returns:
        (cols, rows) : tuple of scalars or `numpy.ndarray`s
            The resulting grid coordinates (indices). Scalars are returned if
            the inputs were also scalars, otherwise arrays.
    """
    lon, lat = v1_meters_to_lonlat(xm, ym, grid_name=grid_name)
    return v1_lonlat_to_colrow_coords(
        lon, lat, grid_name=grid_name, to_int=to_int
    )


def v1_colrow_coords_to_meters(cols, rows, grid_name):
    """Convert EASE-grid row/col coordinates (indices) to EASE-grid projected
    coordinates in meters.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        rows : scalar or array-like
            A single row index or array of row coords. Must match the shape of
            `cols`
        cols : scalar or array-like
            A single col index or array of col coords. Must match the shape of
            `rows`
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (xm, ym) : tuple of scalars or `numpy.ndarray`s
            The resulting x/y projection points in meters. Scalars are returned
            if the inputs were also scalars, otherwise arrays.
    """
    lon, lat = v1_colrow_coords_to_lonlat(cols, rows, grid_name=grid_name)
    return v1_lonlat_to_meters(lon, lat, grid_name=grid_name)


def v1_get_full_grid_coords(grid_name):
    """Return the full set of EASE grid coordinates for the specified grid.

    Parameters:
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (cols, rows) : tuple of 2D ndarrays
            The resulting grid coordinate arrays.
    """
    _validate_grid_name(grid_name)
    nr, nc = GRID_NAME_TO_V1_SHAPE[grid_name]
    cols, rows = np.meshgrid(range(nc), range(nr))
    return cols, rows


def v1_get_full_grid_lonlat(grid_name):
    """Return the full set of lon/lat points for the specified EASE grid.

    Parameters:
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of 2D ndarrays
            The resulting lon/lat arrays
    """
    cols, rows = v1_get_full_grid_coords(grid_name)
    return v1_colrow_coords_to_lonlat(cols, rows, grid_name)


_EPSG_PAT = re.compile("(?:epsg:|EPSG:)?(\\d{4})")
_EPSG_FMT = "epsg:{}"


def _get_proj(proj):
    if isinstance(proj, pyproj.Proj):
        return proj
    elif isinstance(proj, str):
        m = _EPSG_PAT.match(proj)
        if m is None:
            raise ValueError(
                f"Could not determine EPSG code from input: '{proj}'"
            )
        code = m.group(1)[0]
        return pyproj.Proj(_EPSG_FMT.format(code), preserve_unites=False)
    raise TypeError(
        f"proj type must be as tring or pyrpoj.Proj object, not {type(proj)}"
    )


def v1_meters_to_proj(xm, ym, ease_grid_name, proj):
    """Convert EASE-grid coordinates to the specified projection.

    Parameters:
        xm : scalar or array-like
            The EASE grid x coordinate(s).
        ym : scalar or array-like
            The EASE grid y coordinate(s).
        ease_grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`
        proj : str or pyproj.Proj
            Either a string specifiying the EPSG code (eg "epsg:3410", "3410")
            or a pyporj.Proj projection object.

    Returns:
        (xm2, ym2) : tuple of scalars or arrays
            The reprojected x and y coordinates.
    """
    _validate_grid_name(ease_grid_name)
    proj2 = _get_proj(proj)
    eproj = GRID_NAME_TO_V1_PROJ[ease_grid_name]
    return pyproj.transform(eproj, proj2, xm, ym)


################
# EASE GRID V2 #
################

# References:
#   1) https://nsidc.org/ease/ease-grid-projection-gt
#   2) https://www.mdpi.com/2220-9964/1/1/32
#   3) https://www.mdpi.com/2220-9964/3/3/1154/htm
#   4) ftp://sidads.colorado.edu/pub/tools/easegrid2/
#
# Notes:
#   1) ref #3 contains IMPORTANT CORRECTIONS to ref #2
#   2) The corrections specified in ref #3 above have not been applied to the
#      files in ref #4 so #4 should be used as reference but not truth.

# +------+-----------------+------+------+--------+--------------+
# |      |   Projection    |      |  Dimensions   |              |
# | GRID |       \         | EPSG +-------+-------+  Grid Cell   |
# | Name |   Resolution    | Code |  W    |  H    |     Size     |
# +------+-----------------+------+-------+-------+--------------+
# | ML   | Global  25.0 km | 6933 | 1388  |  584  |  25025.26 m  |
# | MH   | Global  12.5 km | 6933 | 2776  | 1168  |  12512.63 m  |
# | NL   | N. Hem. 25.0 km | 6931 |  720  |  720  |  25000.00 m  |
# | NH   | N. Hem. 12.5 km | 6931 | 1440  | 1440  |  12500.00 m  |
# | SL   | S. Hem. 25.0 km | 6932 |  720  |  720  |  25000.00 m  |
# | SH   | S. Hem. 12.5 km | 6932 | 1440  | 1440  |  12500.00 m  |
# +------+-----------------+------+-------+-------+--------------+

# ML
V2_ML_ROWS = 584
V2_ML_COLS = 1388
# NL
V2_NL_ROWS = 720
V2_NL_COLS = 720
# SL
V2_SL_ROWS = 720
V2_SL_COLS = 720

# Shapes of each grid
V2_ML_SHAPE = (V2_ML_ROWS, V2_ML_COLS)
V2_MH_SHAPE = (2 * V2_ML_ROWS, 2 * V2_ML_COLS)
V2_NL_SHAPE = (V2_NL_ROWS, V2_NL_COLS)
V2_NH_SHAPE = (2 * V2_NL_ROWS, 2 * V2_NL_COLS)
V2_SL_SHAPE = (V2_SL_ROWS, V2_SL_COLS)
V2_SH_SHAPE = (2 * V2_SL_ROWS, 2 * V2_SL_COLS)

GRID_NAME_TO_V2_SHAPE = {
    ML: V2_ML_SHAPE,
    MH: V2_MH_SHAPE,
    NL: V2_NL_SHAPE,
    NH: V2_NH_SHAPE,
    SL: V2_SL_SHAPE,
    SH: V2_SH_SHAPE,
}
GRID_NAME_TO_V2_MAP_SCALE = {
    ML: 25025.26,
    MH: 12512.63,
    NL: 25000.00,
    NH: 12500.00,
    SL: 25000.00,
    SH: 12500.00,
}
GRID_NAME_TO_V2_PROJ_CODE = {
    ML: "EPSG:6933",
    MH: "EPSG:6933",
    NL: "EPSG:6931",
    NH: "EPSG:6931",
    SL: "EPSG:6932",
    SH: "EPSG:6932",
}
# Projections for each grid
V2_ML_PROJ = pyproj.Proj(GRID_NAME_TO_V2_PROJ_CODE[ML])
V2_MH_PROJ = V2_ML_PROJ
V2_NL_PROJ = pyproj.Proj(GRID_NAME_TO_V2_PROJ_CODE[NL])
V2_NH_PROJ = V2_NL_PROJ
V2_SL_PROJ = pyproj.Proj(GRID_NAME_TO_V2_PROJ_CODE[SL])
V2_SH_PROJ = V2_SL_PROJ

GRID_NAME_TO_V2_PROJ = {
    ML: V2_ML_PROJ,
    MH: V2_MH_PROJ,
    NL: V2_NL_PROJ,
    NH: V2_NH_PROJ,
    SL: V2_SL_PROJ,
    SH: V2_SH_PROJ,
}


def v2_meters_to_colrow_coords(xm, ym, grid_name=ML, to_int=False):
    """Convert EASE-grid V2 projected coordinates in meters to EASE-grid V2
    cell coordinates (indices).

    Inputs can be scalars or arrays, but the type must be consistent. If
    scalars are used as input, the results will also be scalars.

    Parameters:
        xm : scalar or array-like
            A single EASE-grid V2 x coordinate in meters or an array of points.
            Must match the shape of `ym`.
        ym : scalar or array-like
            A single EASE-grid V2 y coordinate in meters or an array of points.
            Must match the shape of `xm`.
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`
        to_int : boolean
            If `True`, results will be rounded and converted to integers.

    Returns:
        (cols, rows) : tuple of scalars or `numpy.ndarray`s
            The resulting grid coordinates (indices). Scalars are returned if
            the inputs were also scalars, otherwise arrays.

    References:
        * ftp://sidads.colorado.edu/pub/tools/easegrid2/
        * https://nsidc.org/ease/ease-grid-projection-gt
        * https://www.mdpi.com/2220-9964/1/1/32
        * https://www.mdpi.com/2220-9964/3/3/1154/htm
    """
    xm, ym, scalars = _handle_inputs(xm, ym, grid_name)

    nr, nc = GRID_NAME_TO_V2_SHAPE[grid_name]
    r0 = (nc - 1) / 2.0
    s0 = (nr - 1) / 2.0
    map_scale = GRID_NAME_TO_V2_MAP_SCALE[grid_name]
    cols = r0 + (xm / map_scale)
    rows = s0 - (ym / map_scale)
    if to_int:
        cols = np.round(cols, out=cols).astype(int)
        rows = np.round(rows, out=rows).astype(int)
    if scalars:
        cols = cols.min()
        rows = rows.min()
    return cols, rows


def v2_colrow_coords_to_meters(cols, rows, grid_name=ML):
    """Convert EASE-grid V2 row/col coordinates (indices) to EASE-grid V2
    projected coordinates in meters.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        rows : scalar or array-like
            A single row index or array of row coords. Must match the shape of
            `cols`
        cols : scalar or array-like
            A single col index or array of col coords. Must match the shape of
            `rows`
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (xm, ym) : tuple of scalars or `numpy.ndarray`s
            The resulting x/y projection points in meters. Scalars are returned

    References:
        * ftp://sidads.colorado.edu/pub/tools/easegrid2/
        * https://nsidc.org/ease/ease-grid-projection-gt
        * https://www.mdpi.com/2220-9964/1/1/32
        * http           if the inputs were also scalars, otherwise arrays.
    """
    cols, rows, scalars = _handle_inputs(cols, rows, grid_name)

    nr, nc = GRID_NAME_TO_V2_SHAPE[grid_name]
    r0 = (nc - 1) / 2.0
    s0 = (nr - 1) / 2.0
    map_scale = GRID_NAME_TO_V2_MAP_SCALE[grid_name]
    xm = (cols - r0) * map_scale
    ym = (s0 - rows) * map_scale
    if scalars:
        xm = xm.min()
        ym = ym.min()
    return xm, ym


def v2_lonlat_to_colrow_coords(lon, lat, grid_name=ML, to_int=False):
    """Convert lon/lat points to EASE-grid V2 cell coordinates (indices).

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
            The EASE V2 grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`
        to_int : boolean
            If `True`, results will be rounded and converted to integers.

    Returns:
        (cols, rows) : tuple of scalars or `numpy.ndarray`s
            The resulting grid coordinates (indices). Scalars are returned if
            the inputs were also scalars, otherwise arrays.

    References:
        * ftp://sidads.colorado.edu/pub/tools/easegrid2/
        * https://nsidc.org/ease/ease-grid-projection-gt
        * https://www.mdpi.com/2220-9964/1/1/32
        * https://www.mdpi.com/2220-9964/3/3/1154/htm
    """
    lon, lat, scalars = _handle_inputs(lon, lat, grid_name)

    xm, ym = GRID_NAME_TO_V2_PROJ[grid_name](lon, lat)
    cols, rows = v2_meters_to_colrow_coords(xm, ym, grid_name, to_int=to_int)
    if scalars:
        rows = rows.min()
        cols = cols.min()
    return cols, rows


def v2_colrow_coords_to_lonlat(cols, rows, grid_name=ML):
    """Convert EASE-grid V2 row/col coords (indices) to lon/lat values.

    This function accepts scalars and arrays, but the type/shapes must match.

    Parameters:
        rows : scalar or array-like
            A single row index or array of row coords. Must match the shape of
            `cols`
        cols : scalar or array-like
            A single col index or array of col coords. Must match the shape of
            `rows`
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of scalars or `numpy.ndarray`s
            The resulting lon/lat points. Scalars are returned if
            the inputs were also scalars, otherwise arrays.

    References:
        * ftp://sidads.colorado.edu/pub/tools/easegrid2/
        * https://nsidc.org/ease/ease-grid-projection-gt
        * https://www.mdpi.com/2220-9964/1/1/32
        * https://www.mdpi.com/2220-9964/3/3/1154/htm
    """
    cols, rows, scalars = _handle_inputs(cols, rows, grid_name)

    xm, ym = v2_colrow_coords_to_meters(cols, rows, grid_name)
    lon, lat = GRID_NAME_TO_V2_PROJ[grid_name](xm, ym, inverse=True)
    if scalars:
        lon = lon.min()
        lat = lat.min()
    return lon, lat


def v2_lonlat_to_meters(lon, lat, grid_name):
    """Reproject lon/lat points into the specified EASE-grid V2 projection in
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
        * ftp://sidads.colorado.edu/pub/tools/easegrid2/
        * https://nsidc.org/ease/ease-grid-projection-gt
        * https://www.mdpi.com/2220-9964/1/1/32
        * https://www.mdpi.com/2220-9964/3/3/1154/htm
    """
    lon, lat, scalars = _handle_inputs(lon, lat, grid_name)

    xm, ym = GRID_NAME_TO_V2_PROJ[grid_name](lon, lat)
    if scalars:
        xm = xm.min()
        ym = ym.min()
    return xm, ym


def v2_meters_to_lonlat(xm, ym, grid_name=ML):
    """Convert x/y points (in meters) from the specified EASE-grid V2
    projection to lon/lat points.

    Parameters:
        xm : scalar or array-like
            A single EASE-grid V2 x coordinate in meters or an array of points.
        ym : scalar or array-like
            A single EASE-grid V2 y coordinate in meters or an array of points.
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of scalars or `numpy.ndarray`s
            The resulting lon/lat points.

    References:
        * ftp://sidads.colorado.edu/pub/tools/easegrid2/
        * https://nsidc.org/ease/ease-grid-projection-gt
        * https://www.mdpi.com/2220-9964/1/1/32
        * https://www.mdpi.com/2220-9964/3/3/1154/htm
    """
    _validate_grid_name(grid_name)
    return GRID_NAME_TO_V2_PROJ[grid_name](xm, ym, inverse=True)


def v2_get_full_grid_coords(grid_name):
    """Return the full set of EASE-grid V2 coordinates for the specified grid.

    Parameters:
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (cols, rows) : tuple of 2D ndarrays
            The resulting grid coordinate arrays.
    """
    _validate_grid_name(grid_name)
    rows, cols = np.indices(GRID_NAME_TO_V2_SHAPE[grid_name])
    return cols, rows


def v2_get_full_grid_lonlat(grid_name):
    """Return the full set of lon/lat points for the specified EASE-grid V2
    grid.

    Parameters:
        grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`

    Returns:
        (lon, lat) : tuple of 2D ndarrays
            The resulting lon/lat arrays
    """
    cols, rows = v2_get_full_grid_coords(grid_name)
    return v2_colrow_coords_to_lonlat(cols, rows, grid_name=grid_name)


def v2_meters_to_proj(xm, ym, ease_grid_name, proj):
    """Convert EASE-grid V2 coordinates to the specified projection.

    Parameters:
        xm : scalar or array-like
            The EASE grid x coordinate(s).
        ym : scalar or array-like
            The EASE grid y coordinate(s).
        ease_grid_name : str
            The EASE grid key name. Must be one of the keys found in
            `ease_grid.GRID_NAMES`
        proj : str or pyproj.Proj
            Either a string specifiying the EPSG code (eg "epsg:3410", "3410")
            or a pyporj.Proj projection object.

    Returns:
        (xm2, ym2) : tuple of scalars or arrays
            The reprojected x and y coordinates.
    """
    _validate_grid_name(ease_grid_name)
    proj2 = _get_proj(proj)
    eproj = GRID_NAME_TO_V2_PROJ[ease_grid_name]
    return pyproj.transform(eproj, proj2, xm, ym)
