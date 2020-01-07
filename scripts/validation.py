from collections import namedtuple
from scipy.spatial import cKDTree as KDTree
import argparse
import numpy as np
import pandas as pd
import rasterio as rio
import re
import tqdm

import ease_grid as eg
from utils import (
    day_of_year_to_datetime,
    flatten_to_iterable,
    validate_file_path,
)
from validation_db_orm import (
    DbWMOMetDailyTempRecord,
    DbWMOMetStation,
    date_to_int,
    get_db_session,
    int_to_date,
)


RETRIEVAL_MEAN = "mean"
RETRIEVAL_MIN = "min"
RETRIEVAL_MAX = "max"


def _retrieve_mean(record):
    return record.temperature_mean


def _retrieve_min(record):
    return record.temperature_min


def _retrieve_max(record):
    return record.temperature_max


_RETRIEVAL_FUNCS = {
    RETRIEVAL_MEAN: _retrieve_mean,
    RETRIEVAL_MIN: _retrieve_min,
    RETRIEVAL_MAX: _retrieve_max,
}


class WMOValidationPointFetcher:
    # XXX: NOT PROCESS SAFE
    # This is because the database handle cannot be passed between processes
    # safely.
    def __init__(self, wmo_db, retrieval_type=RETRIEVAL_MEAN):
        self.retrieval_func = None
        self.set_retrieval_type(retrieval_type)
        # TODO: check for correct table
        self._db = wmo_db
        print("Loading stations from db")
        self.stns = {
            s.station_id: s for s in wmo_db.query(DbWMOMetStation).all()
        }

    def set_retrieval_type(self, retrieval_type):
        if retrieval_type not in _RETRIEVAL_FUNCS:
            raise ValueError(f"Invalid retrieval type: {retrieval_type}")
        self.retrieval_func = _RETRIEVAL_FUNCS[retrieval_type]

    def fetch(self, datetime):
        records = (
            self._db.query(DbWMOMetDailyTempRecord)
            .filter(DbWMOMetDailyTempRecord.date_int == date_to_int(datetime))
            .all()
        )
        if not records:
            return None
        lonlats = np.empty((len(records), 2))
        temps = np.empty(len(records))
        i = 0
        for r in records:
            s = self.stns[r.station_id]
            t = self.retrieval_func(r)
            lonlats[i] = (s.lon, s.lat)
            temps[i] = t
            i += t is not None
        # Trim any extra space at ends
        lonlats.resize((i, 2), refcheck=False)
        temps.resize(i, refcheck=False)
        return lonlats, temps


TYPE_AM = "AM"
TYPE_PM = "PM"
# Composite
TYPE_CO = "CO"

OTHER = -1
FROZEN = 0
THAWED = 1

FT_ESDR_FROZEN = 0
FT_ESDR_THAWED = 1
# Frozen in AM, thawed in PM
FT_ESDR_TRANSITIONAL = 2
# Thawed in AM, frozen in PM
FT_ESDR_INV_TRANSITIONAL = 3


_EASE_LON, _EASE_LAT = eg.ease1_get_full_grid_lonlat(eg.ML)
_EPOINTS = np.array(list(zip(_EASE_LON.ravel(), _EASE_LAT.ravel())))
_EASE_NH_MASK = _EASE_LAT >= 0.0
_EASE_SH_MASK = _EASE_LAT < 0.0


class PointsGridder:
    """Take points and shift them onto a grid using nearest neighbor approach.
    """

    def __init__(self, xgrid, ygrid, invalid_mask=None):
        print("Generating tree")
        self.tree = KDTree(np.array(list(zip(xgrid.ravel(), ygrid.ravel()))))
        self.imask = invalid_mask

    def __call__(self, grid, points, values, clear=False, fill=OTHER):
        if clear:
            grid[:] = fill
        _, idx = self.tree.query(points)
        grid.ravel()[idx] = values
        if self.imask is not None:
            grid[self.imask] = fill


def ft_model_zero_threshold(temps):
    return (temps > 273.15).astype("uint8")


def get_empty_data_grid(shape):
    return np.full(shape, OTHER, dtype="int8")


def get_empty_data_grid_like(a):
    return get_empty_data_grid(a.shape)


COL_YEAR = "year"
COL_MONTH = "month"
COL_SCORE = "score"
RESULT_COLS = (COL_YEAR, COL_MONTH, COL_SCORE)
SCORE_FILL = -1.0

COL_DATE = "date"
COL_PASS = "pass"
COL_REGION = "region"


def _count_valid_points(grid):
    return np.count_nonzero(grid > OTHER)


def _count_shared_valid_points(lgrid, rgrid):
    lvalid = lgrid > OTHER
    rvalid = rgrid > OTHER
    return np.count_nonzero(lvalid & rvalid)


def _validate_nh_sh_global(egrid, vgrid, vpoints, vtemps, point_gridder):
    vft = ft_model_zero_threshold(vtemps)
    point_gridder(vgrid, vpoints, vft, clear=True, fill=OTHER)
    shared_valid_mask = (egrid > OTHER) & (vgrid > OTHER)
    valid_nh = shared_valid_mask & _EASE_NH_MASK
    valid_sh = shared_valid_mask & _EASE_SH_MASK
    egrid_nh = egrid[valid_nh]
    egrid_sh = egrid[valid_sh]
    vgrid_nh = vgrid[valid_nh]
    vgrid_sh = vgrid[valid_sh]
    n_full = np.count_nonzero(shared_valid_mask)
    n_nh = np.count_nonzero(valid_nh)
    n_sh = np.count_nonzero(valid_sh)
    score_nh = np.count_nonzero(vgrid_nh == egrid_nh) / n_nh * 100.0
    score_sh = np.count_nonzero(vgrid_sh == egrid_sh) / n_sh * 100.0
    score_full = (
        np.count_nonzero(vgrid[shared_valid_mask] == egrid[shared_valid_mask])
        / n_full
        * 100.0
    )
    return score_nh, score_sh, score_full


def _validate(estimate_grids, point_fetcher, point_gridder):
    results = []
    if not estimate_grids:
        return results
    k = next(iter(estimate_grids))
    vgrid = get_empty_data_grid_like(estimate_grids[k])
    for date, egrid in tqdm.tqdm(estimate_grids.items(), ncols=80):
        vpoints, temps = point_fetcher.fetch(date)
        vft = ft_model_zero_threshold(temps)
        point_gridder(vgrid, vpoints, vft, clear=True, fill=OTHER)
        n = _count_shared_valid_points(egrid, vgrid)
        score = np.count_nonzero(vgrid == egrid) / n * 100.0
        results.append((date, score))
    return results


def _validate_with_mask(estimate_grids, point_fetcher, point_gridder, mask):
    results = []
    if not estimate_grids:
        return results
    k = next(iter(estimate_grids))
    vgrid = get_empty_data_grid(estimate_grids[k])
    for date, egrid in tqdm.tqdm(estimate_grids.items(), ncols=80):
        vpoints, temps = point_fetcher.fetch(date)
        vft = ft_model_zero_threshold(temps)
        point_gridder(vgrid, vpoints, vft, clear=True, fill=OTHER)
        egrid_masked = egrid[mask]
        vgrid_masked = vgrid[mask]
        n = _count_shared_valid_points(egrid_masked, vgrid_masked)
        score = (vgrid_masked == egrid_masked).sum() / n * 100.0
        results.append((date, score))
    return results


LABEL_NH = "NH"
LABEL_SH = "SH"
LABEL_GLOBAL = "GLOBAL"


def perform_nh_sh_global_validation(
    estimate_grids, point_fetcher, point_gridder
):
    results = []
    if not estimate_grids:
        return results
    k = next(iter(estimate_grids))
    vgrid = get_empty_data_grid_like(estimate_grids[k])
    for date, egrid in tqdm.tqdm(estimate_grids.items(), ncols=80):
        vpoints, vtemps = point_fetcher.fetch(date)
        score_nh, score_sh, score_global = _validate_nh_sh_global(
            egrid, vgrid, vpoints, vtemps, point_gridder
        )
        results.append((date, LABEL_NH, score_nh))
        results.append((date, LABEL_SH, score_sh))
        results.append((date, LABEL_GLOBAL, score_global))
    return results


def validate_global(estimate_grids, point_fetcher, point_gridder):
    # TODO: use verbose setting
    print("Global")
    data = _validate(estimate_grids, point_fetcher, point_gridder)
    return [(d, LABEL_GLOBAL, s) for d, s in data]


def validate_north_hemisphere(estimate_grids, point_fetcher, point_gridder):
    # TODO: use verbose setting
    print("North")
    data = _validate_with_mask(
        estimate_grids, point_fetcher, point_gridder, _EASE_NH_MASK
    )
    return [(d, LABEL_NH, s) for d, s in data]


def validate_south_hemisphere(estimate_grids, point_fetcher, point_gridder):
    # TODO: use verbose setting
    print("South")
    data = _validate_with_mask(
        estimate_grids, point_fetcher, point_gridder, _EASE_SH_MASK
    )
    return [(d, LABEL_SH, s) for d, s in data]


def perform_custom_regional_composite_validation(
    estimate_grids, point_fetcher, point_gridder, validation_funcs
):
    region_results = [
        func(estimate_grids, point_fetcher, point_gridder)
        for func in validation_funcs
    ]
    return flatten_to_iterable(region_results)


HEMISPHERE_AND_GLOBAL_VALIDATION_FUNCS = (
    validate_north_hemisphere,
    validate_south_hemisphere,
    validate_global,
)


LABEL_AM = "AM"
LABEL_PM = "PM"


def perform_regional_composite_validation(
    estimate_grids, point_fetcher, point_gridder, label=None
):
    results = perform_nh_sh_global_validation(
        estimate_grids, point_fetcher, point_gridder
    )
    if label is None:
        return results
    return [(d, label, reg_label, score) for d, reg_label, score in results]


def perform_am_regional_composite_validation(
    am_estimate_grids, point_fetcher, point_gridder,
):
    point_fetcher.set_retrieval_type(RETRIEVAL_MIN)
    return perform_regional_composite_validation(
        am_estimate_grids, point_fetcher, point_gridder, label=LABEL_AM
    )


def perform_pm_regional_composite_validation(
    pm_estimate_grids, point_fetcher, point_gridder,
):
    point_fetcher.set_retrieval_type(RETRIEVAL_MAX)
    return perform_regional_composite_validation(
        pm_estimate_grids, point_fetcher, point_gridder, label=LABEL_PM
    )


def perform_am_pm_regional_composite_validation(
    am_estimate_grids, pm_estimate_grids, point_fetcher, point_gridder,
):
    # AM
    print("Validating: AM")
    am = perform_am_regional_composite_validation(
        am_estimate_grids, point_fetcher, point_gridder
    )
    # PM
    print("Validating: PM")
    pm = perform_pm_regional_composite_validation(
        pm_estimate_grids, point_fetcher, point_gridder
    )
    return flatten_to_iterable((am, pm))


def output_am_pm_regional_composite_validation_stats(results_list):
    df = pd.DataFrame(
        results_list, columns=[COL_DATE, COL_PASS, COL_REGION, COL_SCORE]
    )
    year_groups = df.groupby(df.date.dt.year)
    print()
    for year, group in year_groups:
        print("-" * 72)
        print(f"YEAR: {year}")
        summary = (
            df.groupby([df.date.dt.month, COL_PASS, COL_REGION])
            .mean()
            .unstack([COL_PASS, COL_REGION])
        )
        summary.index.names = [COL_MONTH]
        print(summary)
    print("-" * 72)


def perform_default_am_pm_validation(
    am_estimate_grids, pm_estimate_grids, point_fetcher, point_gridder
):
    stats = perform_am_pm_regional_composite_validation(
        am_estimate_grids, pm_estimate_grids, point_fetcher, point_gridder,
    )
    output_am_pm_regional_composite_validation_stats(stats)


def _load_ampm_ft_esdr_data(data):
    g = get_empty_data_grid_like(data)
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


def perform_validation_on_ft_esdr(db, fpaths, water_mask_file=None):
    for f in fpaths:
        validate_file_path(f)
    if water_mask_file is not None:
        wmask = np.load(water_mask_file)
    pf = WMOValidationPointFetcher(db)
    pg = PointsGridder(
        *eg.ease1_get_full_grid_lonlat(eg.ML), invalid_mask=wmask
    )
    print("Loading files")
    data = load_ft_esdr_data_from_files(fpaths)
    dates_am = [d.dt for d in data if d.am_grid is not None]
    dates_pm = [d.dt for d in data if d.pm_grid is not None]
    grids_am = [d.am_grid for d in data if d.am_grid is not None]
    grids_pm = [d.pm_grid for d in data if d.pm_grid is not None]

    grids_am = {k: v for k, v in zip(dates_am, grids_am)}
    grids_pm = {k: v for k, v in zip(dates_pm, grids_pm)}
    perform_default_am_pm_validation(grids_am, grids_pm, pf, pg)


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "dbpath",
        type=validate_file_path,
        help="Path to validation database file",
    )
    p.add_argument("files", nargs="+", help="FT-ESDR files to process")
    p.add_argument(
        "-w",
        "--water_mask_file",
        type=validate_file_path,
        help="Path to water mask file",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    print(f"Opening database: '{args.dbpath}'")
    db = get_db_session(args.dbpath)
    try:
        perform_validation_on_ft_esdr(db, args.files, args.water_mask_file)
    finally:
        print("Closing database")
        db.close()
