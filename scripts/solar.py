import argparse
import datetime as dt
import numpy as np
import pysolar.solar as ps
import tqdm

import async_utils as au
import ease_grid as eg


rad_direct_vec = np.vectorize(ps.radiation.get_radiation_direct)


def get_radiation_grids(dates, lon_grid, lat_grid):
    rad = np.zeros((len(dates), *lon_grid.shape))
    for i, d in tqdm.tqdm(enumerate(dates), ncols=80, total=len(dates)):
        alt = ps.get_altitude(lat_grid, lon_grid, d)
        rad[i] = rad_direct_vec(d, alt)
    return rad


def get_radiation_grids_parallel(dates, lon_grid, lat_grid):
    rad = np.zeros((len(dates), *lon_grid.shape))
    alts = [ps.get_altitude(lat_grid, lon_grid, d) for d in dates]
    jobs = [au.AsyncJob(rad_direct_vec, d, alt) for d, alt in zip(dates, alts)]
    print("running")
    results = au.run_async_jobs(
        jobs, async_type=au.MULTI_PROCESS, chunk_size=4
    )
    for i, r in enumerate(results):
        rad[i] = r
    return rad


def get_datestimes(year, hour):
    dates = []
    cur = dt.datetime(year, 1, 1, hour, tzinfo=dt.timezone.utc)
    delta = dt.timedelta(days=1)
    while cur.year == year:
        dates.append(cur)
        cur += delta
    return dates


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "year", type=int, action="store", help="The year to create data for"
    )
    p.add_argument(
        "hour",
        type=int,
        action="store",
        help="The hour (UTC) to create data for",
    )
    p.add_argument("out_file", action="store", help="The output file path")
    return p


def main(args):
    lon, lat = eg.v1_get_full_grid_lonlat(eg.ML)
    dates = get_datestimes(args.year, args.hour)
    rads = get_radiation_grids_parallel(dates, lon, lat)
    print(f"Saving to: {args.out_file}")
    np.save(args.out_file, rads)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
