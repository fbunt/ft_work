import argparse
import datetime as dt
import glob
import netCDF4 as nc
import numpy as np
import os
import tqdm
from collections import namedtuple
from itertools import groupby

import ease_grid as eg
import tb as tbmod


ROWS, COLS = eg.GRID_NAME_TO_SHAPE[eg.ML]


class TbDataWrapper:
    def __init__(self, data_pattern):
        self._pat = data_pattern
        self._files = sorted(glob.glob(data_pattern))
        if not len(self._files):
            raise IOError(f"No files found with pattern: '{data_pattern}'")
        r, s = np.meshgrid(range(COLS), range(ROWS))
        self.lon, self.lat = eg.ease1_get_full_grid_lonlat(eg.ML)
        self.dates = []
        self.data = np.zeros((len(self._files), ROWS, COLS))
        self._load()
        self.min = self._grids.min()
        self.max = self._grids.max()

    def _load(self):
        dates = []
        print("Loading...")
        for i, f in enumerate(tqdm.tqdm(self._files, ncols=80)):
            base = os.path.basename(f)
            m = tbmod.EASE_FNAME_PAT.match(base)
            sat_id, projection, year, doy, ad_pass, freq, pol = m.groups()
            dt_year = np.datetime64(year, dtype="datetime64[Y]")
            dt = dt_year + np.timedelta64(int(doy) - 1, "D")
            grid = tbmod.load_tb_file(f)
            dates.append(dt)
            self.data[i] = grid
        self.dates = dates

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of bounds: {idx}")
        dt = self.dates[idx]
        grid = self.data[idx]
        return dt, grid


def _parse_date(fname):
    m = tbmod.EASE_FNAME_PAT.match(os.path.basename(fname))
    if not m:
        return None
    _, _, y, doy, _, _, _ = m.groups()
    dt_y = np.datetime64(y, dtype="datetime64[Y]")
    date = dt_y + np.timedelta64(int(doy) - 1, "D")
    d = date.astype("O")
    return dt.datetime(d.year, d.month, d.day)


def _get_tb_meta(fname):
    m = tbmod.EASE_FNAME_PAT.match(os.path.basename(fname))
    if not m:
        return None
    _, _, _, _, pass_type, freq, pol = m.groups()
    return pass_type, freq, pol


def tb_files_to_data(files, lon=None, lat=None):
    grids = np.empty((len(files), ROWS, COLS))
    for i, f in enumerate(files):
        grids[i] = tbmod.load_tb_file(f)
    dates = np.array([_parse_date(f) for f in files])
    if lon is None or lat is None:
        lon, lat = eg.ease1_get_full_grid_lonlat(eg.ML)
        x, y = eg.ease1_lonlat_to_meters(lon, lat)
        lon = lon[0]
        lat = lat[:, 0]
        x = x[0]
        y = y[:, 0]
    pass_type, freq, pol = _get_tb_meta(files[0])
    return dates, lon, lat, x, y, grids, pass_type, freq, pol


def build_tb_netcdf(
    out_fname, dates, lon, lat, x, y, grids, pass_type, freq, pol
):
    ds = nc.Dataset(out_fname, "w")
    dtime = ds.createDimension("time", None)
    dlon = ds.createDimension("x", len(x))
    dlat = ds.createDimension("y", len(y))
    vtimes = ds.createVariable("time", "f8", ("time",))
    vtimes.calendar = "proleptic_gregorian"
    vtimes.units = f"days since {dates[0]}"
    vtimes[:] = nc.date2num(
        dates, vtimes.units, calendar="proleptic_gregorian"
    )
    vx = ds.createVariable("x", "f8", ("x"))
    vx.units = "meters"
    vx[:] = x
    vy = ds.createVariable("y", "f8", ("y"))
    vy.units = "meters"
    vy[:] = y
    vlon = ds.createVariable("lon", "f4", ("x",))
    vlon.units = "degrees east"
    vlon[:] = lon
    vlat = ds.createVariable("lat", "f4", ("y",))
    vlat.units = "degrees north"
    vlat[:] = lat
    vtb = ds.createVariable(
        "tb",
        "f4",
        ("time", "x", "y"),
        zlib=True,
        least_significant_digit=1,
        fill_value=0,
    )
    # Meta info
    vtb.units = "K"
    vtb[:] = grids
    ds.description = f"Temperature Brightness data"
    ds.history = f"Created on {dt.datetime.now()} with python's netCDF4 module"
    ds.pass_type = pass_type
    ds.frequency = freq
    ds.polarization = pol
    ds.grid_type = "EASE Grid 1.0 - Global (ML) 25 km"
    ds.projection = "EPSG:3410"
    ds.close()


def _find_all_data_files(root_dir):
    if not os.path.ifdir(root_dir):
        raise IOError(f"Invalid data dir: '{root_dir}'")
    pat = os.path.join(root_dir, "**/*[HV]")
    files = glob.glob(pat, recursive=True)
    files.sort()
    return files


_FileGroup = namedtuple(
    "_FileGroup", ("files", "year", "pass_type", "freq", "pol")
)
_TbFile = namedtuple(
    "_TbFile", ("path", "date", "year", "pass_type", "freq", "pol")
)


def _fname_to_meta_obj(path):
    date = _parse_date(path)
    return _TbFile(path, date, date.year, *_get_tb_meta(path))


def _group_files(all_files):
    files = [_fname_to_meta_obj(f) for f in all_files]
    groups = {}
    ptfunc = lambda x: x.pass_type
    polfunc = lambda x: x.pol
    ffunc = lambda x: x.freq
    files.sort(key=lambda x: x.date)
    for y, yg in groupby(files, lambda x: x.date.year):
        yg = list(yg)
        yg.sort(key=ptfunc)
        for pt, ptg in groupby(yg, ptfunc):
            ptg = list(ptg)
            ptg.sort(key=polfunc)
            for pol, polg in groupby(ptg, polfunc):
                polg = list(polg)
                polg.sort(key=ffunc)
                for f, fg in groupby(polg, ffunc):
                    if y not in groups:
                        groups[y] = {}
                    if pt not in groups[y]:
                        groups[y][pt] = {}
                    if pol not in groups[y][pt]:
                        groups[y][pt][pol] = {}
                    groups[y][pt][pol][f] = sorted(
                        [x.path for x in fg], key=lambda x: x.path
                    )


def _validate_file_path(path):
    if not os.path.isfile(path):
        raise IOError(f"Could not locate path: '{path}'")
    return path


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-O", "--overwrite", action="store_true", help="Overwrite output file"
    )
    p.add_argument("outfile", help="output file path")
    p.add_argument(
        "files",
        nargs="+",
        type=_validate_file_path,
        help="The files to be processed",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    files = sorted(args.files)
    if os.path.isfile(args.outfile) and not args.overwrite:
        raise IOError("Output file already present")
    build_tb_netcdf(args.outfile, *tb_files_to_data(files))
