from collections import namedtuple
import argparse
import numpy as np
import pandas as pd
import rasterio as rio
import re

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
        self.stns = {
            s.station_id: s for s in wmo_db.query(DbWMOMetStation).all()
        }

    def get_points(self, datetime):
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


def perform_validation(grids, results, point_fetcher):
    # TODO
    pass


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
    for fp in fpaths:
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


_COL_SCORE = "score"
_SCORE_FILL = -1.0


def perform_validation_on_ft_esdr(db, fpaths):
    for f in fpaths:
        validate_file_path(f)
    pf = WMOValidationPointFetcher(db)
    data = load_ft_esdr_data_from_files(fpaths)
    dates_am = [d.dt for d in data if d.am_grid]
    dates_pm = [d.dt for d in data if d.pm_grid]
    grids_am = [d.am_grid for d in data if d.am_grid]
    grids_pm = [d.pm_grid for d in data if d.pm_grid]
    # AM
    results_am = pd.DataFrame(
        [_SCORE_FILL for d in data if d.am_grid],
        index=dates_am,
        columns=[_COL_SCORE],
    )
    perform_validation(grids_am, results_am, pf)
    # PM
    results_pm = pd.DataFrame(
        [_SCORE_FILL for d in data if d.pm_grid],
        index=dates_pm,
        columns=[_COL_SCORE],
    )
    perform_validation(grids_pm, results_pm, pf)


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
    db = get_db_session(args.dbpath)
    try:
        perform_validation_on_ft_esdr(db, args.files)
    finally:
        db.close()
