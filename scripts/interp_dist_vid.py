import argparse
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import tqdm
from scipy.interpolate import NearestNDInterpolator

from validation_db_orm import (
    DbWMOMetDailyTempMean,
    DbWMOMetStation,
    date_to_int,
    get_db_session,
)
import ease_grid as eg
import utils


def load_records(db, start_date, end_date, delta):
    records = []
    dates = []
    d = start_date
    while d < end_date:
        dates.append(d)
        d += delta
    print("Loading db data")
    stns = {s.station_id: s for s in db.query(DbWMOMetStation)}
    for d in tqdm.tqdm(dates, ncols=80):
        rs = (
            db.query(DbWMOMetDailyTempMean)
            .filter(DbWMOMetDailyTempMean.date_int == date_to_int(d))
            .all()
        )
        rs = [
            (
                stns[r.station_id].lon,
                stns[r.station_id].lat,
                int(r.temperature > 273.15),
            )
            for r in rs
        ]
        records.append(rs)
    return dates, records


def fill_grids(x, y, records, interp_grids, dist_grids):
    print("Interpolating")
    for i, rs in tqdm.tqdm(enumerate(records), ncols=80):
        plon = [v[0] for v in rs]
        plat = [v[1] for v in rs]
        pxm, pym = eg.ease1_lonlat_to_meters(plon, plat, eg.ML)
        points = np.array(list(zip(pxm, pym)))
        values = np.array([v[-1] for v in rs])
        ip = NearestNDInterpolator(points, values)
        igrid = ip(x, y)
        dist, _ = ip.tree.query(np.array(list(zip(x.ravel(), y.ravel()))))
        interp_grids[i] = igrid
        dist_grids[i] = dist.reshape(x.shape)


cmap = cmap = colors.ListedColormap(["skyblue", "lightcoral"])
norm = colors.BoundaryNorm([0, 1, 2], 2)


def plot(
    date, lons, lats, igrid, dgrid, interp_plot_dir, dist_plot_dir, vmin, vmax
):
    plt.figure()
    plt.contourf(lons, lats, igrid, 2, cmap=cmap, norm=norm)
    plt.title(str(date))
    name = os.path.join(interp_plot_dir, f"{date}.png")
    plt.savefig(name, dpi=250)
    plt.clf()

    plt.figure()
    plt.contourf(lons, lats, dgrid, 50, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    plt.title(str(date))
    name = os.path.join(dist_plot_dir, f"{date}.png")
    plt.savefig(name, dpi=250)
    plt.clf()


def main(args):
    db = get_db_session(args.db_path)
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    interp_plot_dir = os.path.join(out_dir, "interp")
    if not os.path.isdir(interp_plot_dir):
        os.makedirs(interp_plot_dir)
    dist_plot_dir = os.path.join(out_dir, "dist")
    if not os.path.isdir(dist_plot_dir):
        os.makedirs(dist_plot_dir)
    start_date = dt.date(2000, 7, 1)
    end_date = dt.date(2001, 7, 1)
    dt_delta = dt.timedelta(1)
    dates, records = load_records(db, start_date, end_date, dt_delta)
    lons, lats = eg.ease1_get_full_grid_lonlat(eg.ML)
    x, y = eg.ease1_lonlat_to_meters(lons, lats, eg.ML)
    interp_grids = np.zeros((len(records), *x.shape))
    dist_grids = np.zeros_like(interp_grids)
    fill_grids(x, y, records, interp_grids, dist_grids)
    dmin = dist_grids.min()
    dmax = dist_grids.max()

    for i in range(len(dates)):
        plot(
            dates[i],
            lons,
            lats,
            interp_grids[i],
            dist_grids[i],
            interp_plot_dir,
            dist_plot_dir,
            dmin,
            dmax,
        )
    db.close()


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "db_path", type=utils.validate_file_path, help="Database file path"
    )
    p.add_argument("out_dir", type=str, help="Output root dir")
    return p


if __name__ == "__main__":
    main(_get_parser().parse_args())
