import datetime as dt
import numpy as np
import os

import datahandling as dh
from prep_data import (
    dataset_to_array,
    fill_gaps,
    get_n_dates,
    get_missing_ratio,
    get_year_str,
)


def prep(
    start_date,
    tb,
    era_ft,
    era_t2m,
    out_dir,
    drop_bad_days,
    missing_cutoff=0.6,
    periodic=True,
    non_periodic_bias_val=20,
):
    out_dir = os.path.abspath(out_dir)
    n = len(tb)
    dates = np.array(get_n_dates(start_date, n))

    if drop_bad_days:
        # Filter out indices where specified ratio of Tb data is missing
        good_idxs = [
            i for i in range(n) if get_missing_ratio(tb[i]) < missing_cutoff
        ]
        bad_idxs = [
            i for i in range(n) if get_missing_ratio(tb[i]) >= missing_cutoff
        ]
    else:
        good_idxs = list(range(n))
        bad_idxs = []
    n = len(good_idxs)
    dropped_dates = dates[bad_idxs]
    dates = dates[good_idxs]
    tb = tb[good_idxs]
    era_ft = era_ft[good_idxs]
    era_t2m = era_t2m[good_idxs]
    tb = fill_gaps(
        tb, periodic=periodic, non_periodic_bias_val=non_periodic_bias_val
    )

    start_year = dates[0].year
    end_year = dates[-1].year
    year_str = get_year_str(start_year, end_year)
    np.save(f"{out_dir}/tb-D-{year_str}-eu_lowres.npy", tb)
    np.save(f"{out_dir}/era5-ft-am-{year_str}-eu_lowres.npy", era_ft)
    np.save(f"{out_dir}/era5-t2m-am-{year_str}-eu_lowres.npy", era_t2m)
    with open(f"{out_dir}/date_map-{year_str}-eu_lowres.csv", "w") as fd:
        for i, d in zip(good_idxs, dates):
            fd.write(f"{i},{d}\n")
    with open(f"{out_dir}/dropped_dates-{year_str}-eu_lowres.csv", "w") as fd:
        for d in dropped_dates:
            fd.write(f"{d}\n")


out_lon = np.load("../data/eu_lowres/lon-eu-lowres.npy")
out_lat = np.load("../data/eu_lowres/lat-eu-lowres.npy")

# Test
tb = np.load("../data/eu_lowres/tb_18-36_eu_lowres_test.npy")
print("Loading ERA5 FT")
era_ft = dataset_to_array(
    dh.TransformPipelineDataset(
        dh.ERA5BidailyDataset(
            ["../data/era5/t2m/bidaily/era5-t2m-bidaily-2013.nc"],
            "t2m",
            "AM",
            out_lon,
            out_lat,
        ),
        [dh.FTTransform()],
    )
)
era_ft = era_ft[: len(tb)]
print("Loading ERA5 t2m")
era_t2m = dataset_to_array(
    dh.ERA5BidailyDataset(
        ["../data/era5/t2m/bidaily/era5-t2m-bidaily-2013.nc"],
        "t2m",
        "AM",
        out_lon,
        out_lat,
    )
)
era_t2m = era_t2m[: len(tb)]
prep(
    dt.date(2013, 1, 1),
    tb,
    era_ft,
    era_t2m,
    "../data/eu_lowres",
    False,
    periodic=True,
    non_periodic_bias_val=20,
)
# Training
tb = np.load("../data/eu_lowres/tb_18-36_eu_lowres_train.npy")
print("Loading ERA5 FT")
era_ft = dataset_to_array(
    dh.TransformPipelineDataset(
        dh.ERA5BidailyDataset(
            [
                f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                for y in range(2014, 2020 + 1)
            ],
            "t2m",
            "AM",
            out_lon,
            out_lat,
        ),
        [dh.FTTransform()],
    )
)
era_ft = era_ft[: len(tb)]
print("Loading ERA5 t2m")
era_t2m = dataset_to_array(
    dh.ERA5BidailyDataset(
        [
            f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
            for y in range(2014, 2020 + 1)
        ],
        "t2m",
        "AM",
        out_lon,
        out_lat,
    )
)
era_t2m = era_t2m[: len(tb)]
prep(
    dt.date(2014, 1, 1),
    tb,
    era_ft,
    era_t2m,
    "../data/eu_lowres",
    False,
    periodic=False,
    non_periodic_bias_val=200,
)
