from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree as KDTree
import datetime as dt
import numpy as np
import os
import rasterio as rio
import re

from utils import day_of_year_to_date
from validation_db_orm import (
    DbWMOMetDailyTempRecord,
    DbWMOMetStation,
    date_to_int,
    get_db_session,
    int_to_date,
)


class PointsGridder:
    """Take points and shift them onto a grid using nearest neighbor approach.
    """

    def __init__(self, xgrid, ygrid):
        self.tree = KDTree(np.array(list(zip(xgrid.ravel(), ygrid.ravel()))))

    def __call__(self, grid, points, values, clear=False, fill=np.nan):
        if clear:
            grid[:] = fill
        _, idx = self.tree.query(points)
        grid.ravel()[idx] = values


class NNInterpolator:
    """Nearest Neighbor interpolator that also returns distances."""

    def __init__(self, points, values):
        self.ndim = points.shape[1]
        self.tree = KDTree(points)
        self.values = values

    def __call__(self, x, y):
        xi = _ndim_coords_from_arrays((x, y), ndim=self.ndim)
        dist, idx = self.tree.query(xi)
        return dist, self.values[idx]


def percent_grid_agreement(g1, g2, n):
    return np.sum(g1 == g2) / n


def validate_grid(grid, val_grid, vpoints, vft, vgridder):
    vgridder(val_grid, vpoints, vft, clear=True)
    return percent_grid_agreement(grid, val_grid, len(vft))


class WMOValidationPointFetcher:
    def __init__(self, wmo_db):
        self._db = wmo_db
        self.stns = {
            s.station_id: s for s in wmo_db.query(DbWMOMetStation).all()
        }

    def get_points(self, datetime):
        records = (
            self._db.query(DbWMOMetDailyTempRecord)
            .filter(DbWMOMetDailyTempRecord.date_int == date_to_int(datetime))
            .all()
        )
        if not records:
            return np.zeros((0, 2)), np.zeros(0)
        lonlats = np.empty((len(records), 2))
        temps = np.empty(len(records))
        for i, r in enumerate(records):
            s = self.stns[r.station_id]
            lonlats[i] = (s.lon, s.lat)
            temps[i] = r.temperature_mean
        return lonlats, temps


def validate_ft_esdr_grids(grids, vpoint_fetcher):
    # TODO: account for am/pm/co grid types
    pass


FT_ESDR_FNAME_REGEX = re.compile(
    r"(?P<basis>AMSR|SMMR|SSMI)_(?P<freq>\d+)(?P<pol>V|H)_(?P<ampm>AM|PM)_FT_(?P<year>\d{4})_day(?P<doy>\d{3})\.tif"  # noqa: E501
)


def _parse_ft_esdr_path(path):
    fname = os.paht.basename(path)
    m = FT_ESDR_FNAME_REGEX.match(fname)
    if not m:
        raise ValueError("Could not parse FT ESDR file name for info.")
    d = day_of_year_to_date(m["year"], m["doy"])
    d = dt.datetime(
        d.year, d.month, d.day, hour=6 if m["ampm"] == "AM" else 18
    )
    return m["basis"], m["freq"], m["pol"], d


def load_ft_esdr_file(path):
    reader = rio.open(path)
    grid = reader.read(1)
    reader.close()
    _, _, _, datetime = _parse_ft_esdr_path(path)
    return datetime, grid


FROZEN = 0
THAWED = 1
WATER = -1
OTHER = -2


def grid_agreement(truth_grid, estimate_grid):
    cnt = (
        ((truth_grid == THAWED) & (estimate_grid == THAWED))
        | ((truth_grid == FROZEN) & (estimate_grid == FROZEN))
    ).sum()
    # TODO: implement total calculation


def site_agreement(truth_sites, estimate_grid):
    n = len(truth_sites)
