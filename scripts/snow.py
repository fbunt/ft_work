import argparse
import calendar
import glob
import gzip
import numpy as np
import os
import pyproj
import rasterio as rio
import tqdm
from itertools import zip_longest

import ease_grid as eg
from transforms import NH_VIEW_TRANS
from utils import validate_dir_path, validate_file_path


SNOW_4K_PROJ = pyproj.Proj(
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-80 +x_0=0 +y_0=0 +a=6378137 "
    "+rf=291.505347349177 +units=m +no_defs"
)
SNOW_4KM_SIZE = 6144
SNOW_MISSING_VALUE = -1


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "meta_tif", type=validate_file_path, help="tif file with meta data"
    )
    p.add_argument(
        "in_dir",
        type=validate_dir_path,
        help="Directory containing data files",
    )
    p.add_argument("out_file", type=str, help="Location of output file")
    return p


def get_value(data, r, c, n=0):
    return np.int8(np.round(data[r - n : r + n + 1, c - n : c + n + 1].mean()))


def parse_snow_cover_file(path):
    with gzip.open(path) as gfd:
        lines = gfd.read().decode("utf-8").split("\n")[30:-1]
    data = np.zeros((SNOW_4KM_SIZE, SNOW_4KM_SIZE))
    for i, line in enumerate(lines):
        line = line.strip()
        try:
            data[i] = np.fromiter(line, dtype=np.int8)
        except ValueError as e:
            print(f"\nFailed at line {i + 31}: {path}")
            raise e
    data = data[::-1]
    return data


def _parse_index(path):
    fname = os.path.basename(path)
    return int(fname[7:10]) - 1


def _chunk_iter(it, n):
    args = [iter(it)] * n
    return zip_longest(*args)


def fill_data_array(data, files, idxs, meta_tif, proj, lon, lat):
    txm, tym = proj(lon, lat)
    rc = [meta_tif.index(*pair) for pair in zip(txm.ravel(), tym.ravel())]
    # with tqdm.tqdm(total=len(files), ncols=80, desc="Loading data") as pbar:
    #     for chunk in _chunk_iter(zip(i, f)):
    for i, f in tqdm.tqdm(
        zip(idxs, files), ncols=80, total=len(files), desc="Loading data"
    ):
        snow = parse_snow_cover_file(f)
        data[i].ravel()[:] = [get_value(snow, r, c, 2) for (r, c) in rc]
        # data[i].ravel()[:] = [snow[r, c] for (r, c) in rc]


def main(args):
    with rio.open(args.meta_tif) as tif:
        proj = SNOW_4K_PROJ
        year = int(os.path.basename(os.path.dirname(args.in_dir)))
        ndays = 365 + int(calendar.isleap(year))
        files = sorted(glob.glob(os.path.join(args.in_dir, "*.asc.gz")))
        idxs = [_parse_index(f) for f in files]
        lon, lat = [
            NH_VIEW_TRANS(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)
        ]
        data = np.full((ndays, *lon.shape), SNOW_MISSING_VALUE, dtype=np.int8)
        fill_data_array(data, files, idxs, tif, proj, lon, lat)
    print(f"Saving data to disk: '{args.out_file}'")
    np.save(args.out_file, data)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
