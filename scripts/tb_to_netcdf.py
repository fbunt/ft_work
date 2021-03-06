import argparse
import calendar
import datetime as dt
import glob
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import tqdm
from collections import namedtuple

import async_utils as au
import ease_grid as eg
import tb as tbmod


ROWS, COLS = eg.GRID_NAME_TO_V1_SHAPE[eg.ML]


def _parse_date(year, day_of_year):
    dt_y = np.datetime64(year, dtype="datetime64[Y]")
    date = dt_y + np.timedelta64(int(day_of_year) - 1, "D")
    d = date.astype("O")
    return dt.datetime(d.year, d.month, d.day)


def _parse_date_from_fname(fname):
    m = tbmod.EASE_FNAME_PAT.match(os.path.basename(fname))
    if not m:
        return None
    _, _, y, doy, _, _, _ = m.groups()
    return _parse_date(y, doy)


def _get_tb_meta(fname):
    m = tbmod.EASE_FNAME_PAT.match(os.path.basename(fname))
    if not m:
        return None
    # sat_id, proj, year, doy, pass_type, freq, pol
    return m.groups()


def _fill_gaps(dates, grids):
    """Fill in missing dates with zeroed data and return mask indicating
    missing days.
    """
    year = dates[0].year
    days_in_year = 365 + calendar.isleap(year)
    # filler grid for missing days
    missing_grid = np.zeros_like(grids[0])
    # mask indicating missing days
    missing_mask = np.zeros(days_in_year)

    start_day = dt.datetime(year, 1, 1)
    day_to_grid = {d: g for d, g in zip(dates, grids)}
    dates_out = [start_day + dt.timedelta(days=i) for i in range(days_in_year)]
    grids_out = []
    all_days = frozenset(dates_out)
    valid_dates = frozenset(dates)
    for i, d in enumerate(sorted(all_days)):
        if d in valid_dates:
            missing_mask[i] = 0
            grids_out.append(day_to_grid[d])
        else:
            missing_mask[i] = 1
            grids_out.append(missing_grid)
    grids_out = np.array(grids_out)
    return dates_out, grids_out, missing_mask


def load_data(files, proj):
    if not files:
        raise ValueError("files must not be empty")
    dates = [_parse_date_from_fname(f) for f in files]
    grids = [tbmod.load_tb_file(f, proj) for f in files]
    dates, grids, missing_mask = _fill_gaps(dates, grids)
    lon, lat = eg.v1_get_full_grid_lonlat(proj)
    x, y = eg.v1_lonlat_to_meters(lon, lat, proj)
    lon = lon[0]
    lat = lat[:, 0]
    x = x[0]
    y = y[:, 0]
    return dates, lon, lat, x, y, grids, missing_mask


def build_tb_netcdf(
    out_fname,
    dates,
    lon,
    lat,
    x,
    y,
    grids,
    missing_mask,
    sat_id,
    proj,
    pass_type,
    pol,
    freq,
):
    ds = nc.Dataset(out_fname, "w")
    ds.createDimension("time", None)
    ds.createDimension("lon", len(lon))
    ds.createDimension("lat", len(lat))
    # Time
    vtimes = ds.createVariable("time", "f8", ("time",))
    vtimes.calendar = "proleptic_gregorian"
    vtimes.units = f"days since {dates[0]}"
    vtimes[:] = nc.date2num(
        dates, vtimes.units, calendar="proleptic_gregorian"
    )
    # Missing day mask (1 == missing day)
    vmissing = ds.createVariable("missing_dates_mask", "u1", ("time"))
    vmissing.units = "boolean"
    vmissing[:] = missing_mask
    # lon
    vlon = ds.createVariable("lon", "f4", ("lon",))
    vlon.units = "degrees east"
    vlon[:] = lon
    # lat
    vlat = ds.createVariable("lat", "f4", ("lat",))
    vlat.units = "degrees north"
    vlat[:] = lat
    # x
    vx = ds.createVariable("x", "f8", ("lon"))
    vx.units = "meters"
    vx[:] = x
    # y
    vy = ds.createVariable("y", "f8", ("lat"))
    vy.units = "meters"
    vy[:] = y
    # tb
    vtb = ds.createVariable(
        "tb",
        "f4",
        ("time", "lat", "lon"),
        zlib=True,
        least_significant_digit=1,
        fill_value=0,
    )
    # Meta info
    vtb.units = "K"
    vtb[:] = grids
    ds.description = f"Temperature Brightness data"
    ds.history = f"Created on {dt.datetime.now()} with python's netCDF4 module"
    ds.satellite_id = sat_id
    ds.pass_type = pass_type
    ds.frequency = freq
    ds.polarization = pol
    ds.ease_grid_type = proj
    ds.projection = eg.GRID_NAME_TO_V1_PROJ_CODE[proj]
    ds.close()


def _find_all_data_files(root_dir):
    if not os.path.isdir(root_dir):
        raise IOError(f"Invalid data dir: '{root_dir}'")
    pat = os.path.join(root_dir, "**/*[HV]")
    files = glob.glob(pat, recursive=True)
    files.sort()
    return [_fname_to_meta_obj(f) for f in files]


_FileGroup = namedtuple(
    "_FileGroup",
    ("files", "year", "sat_id", "proj", "pass_type", "pol", "freq"),
)
_tb_fields = ("path", "year", "sat_id", "proj", "pass_type", "pol", "freq")
_TbFile = namedtuple("_TbFile", _tb_fields)


def _fname_to_meta_obj(path):
    sat_id, proj, y, doy, pt, f, pol = _get_tb_meta(path)
    date = _parse_date(y, doy)
    return _TbFile(path, str(date.year), sat_id, proj, pt, pol, f)


def _group_files(tbfiles):
    df = pd.DataFrame(tbfiles, columns=_tb_fields)
    # list of the form:
    # [((1987, 'F13', 'ML', 'A', 'H', '19'), DataFrame(matching rows)), ...]
    df_groups = list(df.groupby(list(_tb_fields[1:])))
    fgroups = []
    for meta, gdf in df_groups:
        paths = gdf["path"].to_list()
        paths.sort()
        fgroups.append(_FileGroup(paths, *meta))
    return fgroups


_ofname_fmt = "tb_{year}_{sat_id}_{proj}_{pass_type}_{freq}{pol}.nc"


def _get_outfile_name(fg):
    return _ofname_fmt.format(
        year=fg.year,
        sat_id=fg.sat_id,
        proj=fg.proj,
        pass_type=fg.pass_type,
        freq=fg.freq,
        pol=fg.pol,
    )


def _handle_group(fg, out_dir, overwrite):
    group_dir = os.path.join(out_dir, fg.year)
    # Use try/except to prevent race condition if executed in parallel
    try:
        os.makedirs(group_dir)
    except FileExistsError:
        # Directory was already created
        pass
    outpath = os.path.join(group_dir, _get_outfile_name(fg))
    if not overwrite and os.path.isfile(outpath):
        print(f"File already present: '{outpath}'")
        return
    build_tb_netcdf(
        outpath,
        *load_data(fg.files, fg.proj),
        fg.sat_id,
        fg.proj,
        fg.pass_type,
        fg.pol,
        fg.freq,
    )


# ref: http://www.remss.com/missions/ssmi/
_SSMI_F15_CUTOFF_DATE = dt.datetime(2006, 8, 1)


def _ssmis_f15_filter(tbfile):
    # F15 stopped producing useful data in 2006-08-01
    if tbfile.sat_id != "f15":
        return True
    if int(tbfile.year) < _SSMI_F15_CUTOFF_DATE.year:
        return True
    date = _parse_date_from_fname(tbfile.path)
    if date < _SSMI_F15_CUTOFF_DATE:
        return True
    return False


# ref: http://www.remss.com/missions/ssmi/
_SSMIS_F19_CUTOFF_DATE = dt.datetime(2016, 2, 1)


def _ssmis_f19_filter(tbfile):
    # F19 stopped producing useful data in 2016-02-01
    if tbfile.sat_id != "f19":
        return True
    if int(tbfile.year) < _SSMIS_F19_CUTOFF_DATE.year:
        return True
    date = _parse_date_from_fname(tbfile.path)
    if date < _SSMIS_F19_CUTOFF_DATE:
        return True
    return False


def _MH_filter(tbfile):
    # MH files do not have the proper size
    return tbfile.proj != eg.MH


def _filter_files(tbfiles):
    filters = [_ssmis_f15_filter, _ssmis_f19_filter, _MH_filter]
    for f in filters:
        tbfiles = filter(f, tbfiles)
    return list(tbfiles)


def batch_flat_to_netcdf(root_dir, out_dir, overwrite=False, async_=False):
    print("Finding all files")
    files = _find_all_data_files(root_dir)
    n = len(files)
    print(f"Found: {n}")
    files = _filter_files(files)
    print(f"Filtered: {n - len(files)}")
    print(f"Final count: {len(files)}")
    print("Grouping")
    fgroups = _group_files(files)
    print("Converting")
    if async_:
        jobs = [
            au.AsyncJob(_handle_group, fg, out_dir, overwrite)
            for fg in fgroups
        ]
        au.run_async_jobs(jobs, au.MULTI_PROCESS, max_workers=6, chunk_size=4)
    else:
        for fg in tqdm.tqdm(fgroups, ncols=80):
            _handle_group(fg, out_dir, overwrite)


def _validate_file_path(path):
    if not os.path.isfile(path):
        raise IOError(f"Could not locate path: '{path}'")
    return path


def _validate_dir_path(path):
    if not os.path.isdir(path):
        raise IOError(f"Could not locate path: '{path}'")
    return path


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-O", "--overwrite", action="store_true", help="Overwrite output file"
    )
    p.add_argument(
        "input_dir",
        type=_validate_dir_path,
        help="Directory to look for files in",
    )
    p.add_argument("out_dir", help="output directory path")
    p.add_argument(
        "-a",
        "--asynchronous",
        action="store_true",
        help="Perform conversions asynchronously",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    batch_flat_to_netcdf(
        args.input_dir, args.out_dir, args.overwrite, args.asynchronous
    )
