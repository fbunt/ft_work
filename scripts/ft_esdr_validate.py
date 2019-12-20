from collections import namedtuple
from scipy.spatial import cKDTree as KDTree
import argparse
import numpy as np
import pandas as pd
import rasterio as rio
import re
import tqdm

import ease_grid as eg
from utils import day_of_year_to_datetime, validate_file_path
from validation_db_orm import (
    DbWMOMetDailyTempMean,
    DbWMOMetStation,
    date_to_int,
    get_db_session,
    int_to_date,
)


class WMOValidationPointFetcher:
    def __init__(self, wmo_db):
        # TODO: check for correct table
        self._db = wmo_db
        print("Loading stations from db")
        self.stns = {
            s.station_id: s for s in wmo_db.query(DbWMOMetStation).all()
        }

    def fetch(self, datetime):
        # TODO: cache points
        records = (
            self._db.query(DbWMOMetDailyTempMean)
            .filter(DbWMOMetDailyTempMean.date_int == date_to_int(datetime))
            .all()
        )
        if not records:
            return None
        lonlats = np.empty((len(records), 2))
        temps = np.empty(len(records))
        for i, r in enumerate(records):
            s = self.stns[r.station_id]
            lonlats[i] = (s.lon, s.lat)
            temps[i] = r.temperature
        return lonlats, temps


TYPE_AM = "AM"
TYPE_PM = "PM"
# Composite
TYPE_CO = "CO"

FROZEN = 0
THAWED = 1
OTHER = -1

FT_ESDR_FROZEN = 0
FT_ESDR_THAWED = 1
# Frozen in AM, thawed in PM
FT_ESDR_TRANSITIONAL = 2
# Thawed in AM, frozen in PM
FT_ESDR_INV_TRANSITIONAL = 3


_EASE_LON, _EASE_LAT = eg.ease1_get_full_grid_lonlat(eg.ML)
_EPOINTS = np.array(list(zip(_EASE_LON.ravel(), _EASE_LAT.ravel())))


class PointsGridder:
    """Take points and shift them onto a grid using nearest neighbor approach.
    """

    def __init__(self, xgrid, ygrid):
        print("Generating tree")
        self.tree = KDTree(np.array(list(zip(xgrid.ravel(), ygrid.ravel()))))

    def __call__(self, grid, points, values, clear=False, fill=np.nan):
        if clear:
            grid[:] = fill
        _, idx = self.tree.query(points)
        grid.ravel()[idx] = values


def ft_model_zero_threshold(temps):
    return (temps > 273.15).astype("uint8")


COL_YEAR = "YEAR"
COL_MONTH = "MONTH"
COL_SCORE = "SCORE"
RESULT_COLS = (COL_YEAR, COL_MONTH, COL_SCORE)
SCORE_FILL = -1.0


def perform_validation(estimate_grids, point_fetcher, point_gridder):
    results = pd.DataFrame(
        [(k.year, k.month, SCORE_FILL) for k in estimate_grids],
        index=sorted(estimate_grids),
        columns=RESULT_COLS,
    )
    # Validation grid
    vgrid = np.full(eg.GRID_NAME_TO_SHAPE[eg.ML], np.nan)
    print("Validating")
    for date, egrid in tqdm.tqdm(estimate_grids.items(), ncols=80):
        vpoints, temps = point_fetcher.fetch(date)
        vft = ft_model_zero_threshold(temps)
        point_gridder(vgrid, vpoints, vft, clear=True, fill=np.nan)
        n = len(vft)
        score = (vgrid == egrid).sum() / n
        results.loc[date, COL_SCORE] = score * 100.0
    return results


def _load_ampm_ft_esdr_data(data):
    g = np.full_like(data, OTHER)
    g[data == FT_ESDR_FROZEN] = FROZEN
    g[data == FT_ESDR_THAWED] = THAWED
    return g


def _load_ampm_ft_esdr_file(fpath):
    data = rio.open(fpath).read(1)
    return _load_ampm_ft_esdr_data(data)


def _load_composite_ft_esdr_file(fpath):
    data = rio.open(fpath).read(1)
    am = _load_ampm_ft_esdr_data(data)
    pm = _load_ampm_ft_esdr_data(data)
    am[data == FT_ESDR_TRANSITIONAL] = FROZEN
    am[data == FT_ESDR_INV_TRANSITIONAL] = THAWED
    pm[data == FT_ESDR_TRANSITIONAL] = THAWED
    pm[data == FT_ESDR_INV_TRANSITIONAL] = FROZEN
    return am, pm


def _load_ft_esdr_file(fpath, type_):
    if type_ == TYPE_AM:
        return _load_ampm_ft_esdr_file(fpath), None
    if type_ == TYPE_PM:
        return None, _load_ampm_ft_esdr_file(fpath)
    # TYPE_CO
    return _load_composite_ft_esdr_file(fpath)


FT_ESDR_FNAME_REGEX = re.compile(
    r"(?P<satsys>AMSR|SMMR|SSMI)_(?P<freq>\d+)(?P<pol>V|H)_(?P<type>AM|PM|CO)_FT_(?P<year>\d{4})_day(?P<doy>\d{3})\.tif$"  # noqa: E501
)

FTESDRGrid = namedtuple("FTESDRGrid", ("dt", "type", "am_grid", "pm_grid"))


def load_ft_esdr_data_from_files(fpaths):
    nr, nc = eg.GRID_NAME_TO_SHAPE[eg.ML]
    grids = []
    for fp in tqdm.tqdm(fpaths, ncols=80):
        try:
            m = FT_ESDR_FNAME_REGEX.search(fp)
            type_ = m["type"]
            y = m["year"]
            doy = m["doy"]
            dt = day_of_year_to_datetime(y, doy)
            am_grid, pm_grid = _load_ft_esdr_file(fp, type_)
            grids.append(FTESDRGrid(dt, type_, am_grid, pm_grid))
        except (IndexError, TypeError):
            print(f"Could not parse meta data from path: {fp}")
    return grids


def perform_validation_on_ft_esdr(db, fpaths):
    for f in fpaths:
        validate_file_path(f)
    pf = WMOValidationPointFetcher(db)
    pg = PointsGridder(*eg.ease1_get_full_grid_lonlat(eg.ML))
    print("Loading files")
    data = load_ft_esdr_data_from_files(fpaths)
    dates_am = [d.dt for d in data if d.am_grid is not None]
    dates_pm = [d.dt for d in data if d.pm_grid is not None]
    grids_am = [d.am_grid for d in data if d.am_grid is not None]
    grids_pm = [d.pm_grid for d in data if d.pm_grid is not None]
    # AM
    results_am = None
    if dates_am:
        print("Processing AM")
        grids_am = {k: v for k, v in zip(dates_am, grids_am)}
        results_am = perform_validation(grids_am, pf, pg)
        print(results_am.groupby([COL_YEAR, COL_MONTH]).mean())
    # PM
    results_pm = None
    if dates_pm:
        print("Processing PM")
        grids_pm = {k: v for k, v in zip(dates_pm, grids_pm)}
        results_pm = perform_validation(grids_pm, pf, pg)
        print(results_pm.groupby([COL_YEAR, COL_MONTH]).mean())
    return results_am, results_pm


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "dbpath",
        type=validate_file_path,
        help="Path to validation database file",
    )
    p.add_argument("files", nargs="+", help="FT-ESDR files to process")
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    print("Opening database")
    db = get_db_session(args.dbpath)
    try:
        perform_validation_on_ft_esdr(db, args.files)
    finally:
        db.close()
