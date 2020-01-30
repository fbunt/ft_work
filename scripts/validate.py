from collections import namedtuple
from scipy.spatial import cKDTree as KDTree
import argparse
import numpy as np
import os
import pandas as pd
import rasterio as rio
import re
import tqdm

import ease_grid as eg
from utils import (
    day_of_year_to_datetime,
    flatten_to_iterable,
    validate_file_path,
    validate_file_path_list,
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

    def fetch_bounded(self, datetime, bounds):
        records = (
            self._db.query(
                DbWMOMetDailyTempRecord,
                DbWMOMetStation.lon,
                DbWMOMetStation.lat,
            )
            .join(DbWMOMetDailyTempRecord.met_station)
            .filter(DbWMOMetDailyTempRecord.date_int == date_to_int(datetime))
            .filter(DbWMOMetStation.lon >= bounds[0])
            .filter(DbWMOMetStation.lon <= bounds[1])
            .filter(DbWMOMetStation.lat >= bounds[2])
            .filter(DbWMOMetStation.lat <= bounds[3])
            .all()
        )
        if not records:
            return None
        lonlats = np.empty((len(records), 2))
        temps = np.empty(len(records))
        i = 0
        for r in records:
            t = self.retrieval_func(r[0])
            lonlats[i] = (r[1], r[2])
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
        dist, idx = self.tree.query(points)
        # Indices earlier in idx list will be overwritten by duplicates
        # later in the list.
        # Push points that are farther from their nearest neighbor to the front
        # so that they will be overwritten by closer points if there is an
        # index collision.
        di = sorted(zip(dist, idx, values), reverse=True)
        idx = [i[1] for i in di]
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
            group.groupby([group.date.dt.month, COL_PASS, COL_REGION])
            .mean()
            .unstack([COL_PASS, COL_REGION])
        )
        summary.index.names = [COL_MONTH]
        print(summary)
    print("-" * 72)


def output_validation_stats_grouped_by_month(results_list, cols):
    df = pd.DataFrame(results_list, columns=cols)
    year_groups = df.groupby(df.date.dt.year)
    print()
    for year, group in year_groups:
        print("-" * 16)
        print(f"YEAR: {year}")
        summary = group.groupby([group.date.dt.month]).mean()
        summary.index.names = [COL_MONTH]
        print(summary)
    print("-" * 16)


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
    for fp in tqdm.tqdm(fpaths, ncols=80, desc="Loading files"):
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


def load_npy_files(fpaths):
    grids = []
    for fp in tqdm.tqdm(fpaths, ncols=80, desc="Loading files"):
        grids.append(np.load(fp))
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
    data = load_ft_esdr_data_from_files(fpaths)
    dates_am = [d.dt for d in data if d.am_grid is not None]
    dates_pm = [d.dt for d in data if d.pm_grid is not None]
    grids_am = [d.am_grid for d in data if d.am_grid is not None]
    grids_pm = [d.pm_grid for d in data if d.pm_grid is not None]

    grids_am = {k: v for k, v in zip(dates_am, grids_am)}
    grids_pm = {k: v for k, v in zip(dates_pm, grids_pm)}
    perform_default_am_pm_validation(grids_am, grids_pm, pf, pg)


def perform_bounded_validation(
    date_to_grid, point_fetcher, point_gridder, bounds
):
    results = []
    if not date_to_grid:
        return results
    k = next(iter(date_to_grid))
    vgrid = get_empty_data_grid_like(date_to_grid[k])
    for date, egrid in tqdm.tqdm(
        date_to_grid.items(), ncols=80, desc="Validating"
    ):
        vpoints, temps = point_fetcher.fetch_bounded(date, bounds)
        vft = ft_model_zero_threshold(temps)
        point_gridder(vgrid, vpoints, vft, clear=True, fill=OTHER)
        shared_valid_mask = (egrid > OTHER) & (vgrid > OTHER)
        n = np.count_nonzero(shared_valid_mask)
        score = np.count_nonzero(
            vgrid[shared_valid_mask] == egrid[shared_valid_mask]
        )
        score = score / n * 100.0
        results.append((date, score))
    return results


def _verify_grids_are_homogenous_shape(grids):
    if not grids:
        return
    shape = grids[0].shape
    for g in grids[1:]:
        if g.shape != shape:
            raise RuntimeError("Input grids are not homogenous in size")


def perform_custom_validation(
    db, fpaths, dates, start_row, start_col, comp_type, water_mask_file=None
):
    validate_file_path_list(fpaths)
    if len(dates) != len(fpaths):
        raise RuntimeError("Input paths size does not match input dates size")
    if len(fpaths) == 0:
        print("No data")
        return
    if water_mask_file is not None:
        validate_file_path(water_mask_file)
        wmask = np.load(water_mask_file)
    pf = WMOValidationPointFetcher(db, retrieval_type=comp_type)
    data = load_npy_files(fpaths)
    _verify_grids_are_homogenous_shape(data)

    nr, nc = data[0].shape
    xj, xi = np.meshgrid(
        range(start_col, start_col + nc), range(start_row, start_row + nr)
    )
    wmask = wmask[xi, xj]
    lon, lat = eg.ease1_get_full_grid_lonlat(eg.ML)
    lon = lon[xi, xj]
    lat = lat[xi, xj]
    bounds = [lon.min(), lon.max(), lat.min(), lat.max()]
    pg = PointsGridder(lon, lat, invalid_mask=wmask)
    date_to_grid = {d: g for d, g in zip(dates, data)}
    results = perform_bounded_validation(date_to_grid, pf, pg, bounds)
    output_validation_stats_grouped_by_month(results, [COL_DATE, COL_SCORE])


class InputFileParsingError(Exception):
    pass


def _parse_input_file(fname):
    paths = []
    dates = []
    with open(fname) as fd:
        for line in fd:
            p, dstr = line.strip().split()
            paths.append(p)
            date = np.datetime64(dstr).astype("O")
            dates.append(date)
    try:
        validate_file_path_list(paths)
    except IOError:
        raise InputFileParsingError("Could not find listed file(s)")
    return paths, dates


_COMMAND_FT_ESDR = "ft_esdr"
_COMMAND_CUSTOM = "custom"


def _run_ft_esdr(args):
    print(f"Opening database: '{args.dbpath}'")
    db = get_db_session(args.dbpath)
    try:
        perform_validation_on_ft_esdr(
            db, args.input_files, args.water_mask_file
        )
    finally:
        print("Closing database")
        db.close()


def _run_custom(args):
    paths, dates = _parse_input_file(args.path_list_file)
    print(f"Opening database: '{args.dbpath}'")
    db = get_db_session(args.dbpath)
    try:
        perform_custom_validation(
            db,
            paths,
            dates,
            args.start_row,
            args.start_col,
            args.comparison_type,
            args.water_mask_file,
        )
    finally:
        print("Closing database")
        db.close()


def main(args):
    if args.command_name == _COMMAND_FT_ESDR:
        _run_ft_esdr(args)
    elif args.command_name == _COMMAND_CUSTOM:
        _run_custom(args)


def _parser_add_db_path(p):
    p.add_argument(
        "dbpath",
        type=validate_file_path,
        help="Path to validation database file",
    )
    return p


def _validate_comparison_type(v):
    if v in (RETRIEVAL_MIN, RETRIEVAL_MAX, RETRIEVAL_MEAN):
        return v
    raise ValueError(f"Unknown comparison type: {v}")


def _parser_add_comparison_type(p):
    p.add_argument(
        "-t",
        "--comparison_type",
        action="store",
        default=RETRIEVAL_MEAN,
        type=_validate_comparison_type,
        help=(
            "The daily temperature type to compare against. Options are: "
            + f"{RETRIEVAL_MIN}, {RETRIEVAL_MAX}, {RETRIEVAL_MEAN}. Default"
            + f" is {RETRIEVAL_MEAN}."
        ),
    )
    return p


def _parser_add_files_or_path_list_file(p):
    p.add_argument(
        "files_or_path_list_file",
        nargs="+",
        type=validate_file_path,
        help=(
            "The files to be processed or, if -f is specified, a file "
            "containing the paths to the files to be processed."
        ),
    )
    p.add_argument(
        "-f",
        dest="path_file",
        action="store_true",
        help=(
            "Indicates that the input file contains the paths to files to be "
            "processed, of the form <path>[\t<date>] on each line."
        ),
    )
    return p


def _parser_add_path_list_file(p):
    p.add_argument(
        "path_list_file",
        type=validate_file_path,
        help=(
            "An input file containing lines of the form "
            "'<path>\t<date_string>'. The listed files are treated as input."
        ),
    )


def _parser_add_input_files(p):
    p.add_argument(
        "input_files",
        nargs="+",
        type=validate_file_path,
        help="The input files to process",
    )
    return p


def _parser_add_water_mask_option(p):
    p.add_argument(
        "-w",
        dest="water_mask_file",
        type=validate_file_path,
        default=None,
        help="Path to water mask file",
    )
    return p


def _build_ft_esdr_command(p):
    _parser_add_db_path(p)
    _parser_add_input_files(p)
    _parser_add_water_mask_option(p)
    return p


def _build_custom_command(p):
    _parser_add_db_path(p)
    _parser_add_comparison_type(p)
    p.add_argument(
        "start_row",
        type=int,
        help="EASE grid row of top left corner of input data",
    )
    p.add_argument(
        "start_col",
        type=int,
        help="EASE grid column of top left corner of input data",
    )
    _parser_add_path_list_file(p)
    _parser_add_water_mask_option(p)
    return p


def get_cli_parser():
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(title="commands", dest="command_name")
    p_ft_esdr = _build_ft_esdr_command(subparsers.add_parser(_COMMAND_FT_ESDR))
    p_custom = _build_custom_command(subparsers.add_parser(_COMMAND_CUSTOM))
    return p


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
