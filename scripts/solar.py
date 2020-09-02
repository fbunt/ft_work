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


def get_alts(lon, lat, dates):
    jobs = [au.AsyncJob(ps.get_altitude, lon, lat, d) for d in dates]
    results = au.run_async_jobs(jobs, au.MULTI_PROCESS, chunk_size=4)
    return results


def get_radiation_grids_parallel(dates, lon_grid, lat_grid):
    rad = np.zeros((len(dates), *lon_grid.shape))
    print("Generating alts")
    alts = [ps.get_altitude(lat_grid, lon_grid, d) for d in dates]
    print("Calculating radiation")
    jobs = [au.AsyncJob(rad_direct_vec, d, alt) for d, alt in zip(dates, alts)]
    print("running")
    results = au.run_async_jobs(
        jobs, async_type=au.MULTI_PROCESS, chunk_size=4
    )
    for i, r in enumerate(results):
        rad[i] = r
    return rad


def get_times(date, interval=20):
    delta = dt.timedelta(minutes=interval)
    cur = dt.datetime(
        date.year, date.month, date.day, hour=0, tzinfo=dt.timezone.utc
    )
    times = []
    while cur.day == date.day:
        times.append(cur)
        cur += delta
    return times


def get_alts_for_times(lon, lat, times):
    alts = [ps.get_altitude(lat, lon, t) for t in times]
    return alts


def get_cumulative_radiation_for_times(times, alts, shape):
    rad = np.zeros(shape)
    for t, a in zip(times, alts):
        rad += rad_direct_vec(t, a)
    return rad


def get_cumulative_radiation_for_date(date, lon_grid, lat_grid):
    times = get_times(date)
    alts = get_alts_for_times(lon_grid, lat_grid, times)
    return get_cumulative_radiation_for_times(times, alts, lon_grid.shape)


def get_daily_radiation_parallel(dates, lon_grid, lat_grid, num_workers=None):
    jobs = [
        au.AsyncJob(get_cumulative_radiation_for_date, d, lon_grid, lat_grid)
        for d in dates
    ]
    results = au.run_async_jobs(
        jobs,
        async_type=au.MULTI_PROCESS,
        max_workers=num_workers,
        chunk_size=1,
        progress=True,
    )
    rad = np.zeros((len(dates), *lon_grid.shape))
    for i, r in enumerate(results):
        rad[i] = r
    return rad


def get_dates_for_year(year):
    dates = []
    cur = dt.datetime(year, 1, 1)
    delta = dt.timedelta(days=1)
    while cur.year == year:
        dates.append(cur)
        cur += delta
    return dates


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
        "-w",
        "--workers",
        type=int,
        action="store",
        default=None,
        help="The number of worker processes to spawn",
    )
    p.add_argument("out_file", action="store", help="The output file path")
    return p


def main(args):
    lon, lat = eg.v1_get_full_grid_lonlat(eg.ML)
    dates = get_dates_for_year(args.year)
    rads = get_daily_radiation_parallel(dates, lon, lat, args.workers)
    print(f"Saving to: {args.out_file}")
    np.save(args.out_file, rads)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
