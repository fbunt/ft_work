import datetime as dt
import glob
import numpy as np
import os
import torch
import tqdm

from transforms import (
    AK_VIEW_TRANS,
    NH_VIEW_TRANS,
    N45_VIEW_TRANS,
    N45W_VIEW_TRANS,
)
import datahandling as dh
import ease_grid as eg


def get_year_str(ya, yb):
    if ya == yb:
        return str(ya)
    else:
        ya, yb = sorted([ya, yb])
        return f"{ya}-{yb}"


def build_tb_ds(path_groups, transform):
    dss = [
        dh.GridsStackDataset(
            [
                dh.NCDataset([f], "tb", transform=transform)
                for f in sorted(group)
            ]
        )
        for group in path_groups
    ]
    return torch.utils.data.ConcatDataset(dss)


def dataset_to_array(ds, dtype=float):
    n = len(ds)
    shape = (n, *ds[0].shape)
    ar = np.zeros(shape, dtype=dtype)
    for i, x in enumerate(tqdm.tqdm(ds, ncols=80, desc="DS to array")):
        ar[i] = x
    return ar


def get_n_dates(start_date, n):
    dates = []
    d = dt.date(start_date.year, start_date.month, start_date.day)
    delta = dt.timedelta(days=1)
    for i in range(n):
        dates.append(d)
        d += delta
    return dates


def get_missing_ratio(x):
    return np.isnan(x).sum() / x.size


def get_predecessor(
    x,
    i,
    missing,
    missing_func=np.isnan,
    periodic=True,
    non_periodic_bias_val=20,
):
    px = x[i].copy()
    count = np.zeros(px.shape, dtype=int)
    j = i - 1
    add_bias = False
    if j < 0 and not periodic:
        # If j has gone past the end and the dataset is not periodic, add the
        # bias on the next iteration.
        add_bias = True
    while missing.any():
        px[missing] = x[j, missing]
        count[missing] += 1 + (add_bias * non_periodic_bias_val)
        missing = missing_func(px)
        j -= 1
        # If j has gone past the end and the dataset is not periodic, add the
        # bias on the next iteration. If not, then add_bias is set to false and
        # no bias will be added.
        add_bias = (j < 0) and (not periodic)
        if j < 0:
            j = len(x) - 1
    idx = count != 0
    return px[idx], count[idx]


def get_successor(
    x,
    i,
    missing,
    missing_func=np.isnan,
    periodic=True,
    non_periodic_bias_val=20,
):
    sx = x[i].copy()
    count = np.zeros(sx.shape, dtype=int)
    j = i + 1
    add_bias = False
    if j >= len(x):
        j = 0
        if not periodic:
            # If j has gone past the end and the dataset is not periodic, add
            # the bias on the next iteration.
            add_bias = True
    while missing.any():
        sx[missing] = x[j, missing]
        count[missing] += 1 + (add_bias * non_periodic_bias_val)
        missing = missing_func(sx)
        j += 1
        # If j has gone past the end and the dataset is not periodic, add the
        # bias on the next iteration. If not, then add_bias is set to false and
        # no bias will be added.
        add_bias = (j >= len(x)) and (not periodic)
        if j >= len(x):
            j = 0
    idx = count != 0
    return sx[idx], count[idx]


def fill_gaps(
    x, missing_func=np.isnan, periodic=True, non_periodic_bias_val=20
):
    gap_filled = x.copy()
    for i in tqdm.tqdm(range(len(x)), ncols=80, desc="Gap fill"):
        gaps = missing_func(x[i])
        if not gaps.any():
            continue
        # count is how far the alg had to go to find a value
        # Get past value
        pred, pcount = get_predecessor(
            x, i, gaps, missing_func, periodic, non_periodic_bias_val
        )
        # Get future value
        succ, scount = get_successor(
            x, i, gaps, missing_func, periodic, non_periodic_bias_val
        )
        # Weighted mean
        total = pcount + scount
        # The predecessor/successor with the higher count should be weighted
        # less and the opposing weight should be 1 - w.
        pweight = 1 - (pcount / total)
        sweight = 1 - (scount / total)
        gap_filled[i][gaps] = (pweight * pred) + (sweight * succ)
    return gap_filled


def save_data(data_dict, out_dir, year_str, region):
    for fname, data in data_dict.items():
        name = fname.format(out_dir=out_dir, year_str=year_str, region=region)
        print(f"Saving to: '{name}'")
        np.save(name, data)


def is_neg_one(x):
    return x == -1


FMT_FILENAME_SNOW = "{out_dir}/snow_cover-{year_str}-{region}.npy"
FMT_FILENAME_SOLAR = "{out_dir}/solar_rad-AM-{year_str}-{region}.npy"
FMT_FILENAME_TB = "{out_dir}/tb-D-{year_str}-{region}.npy"
FMT_FILENAME_ERA_FT = "{out_dir}/era5-ft-am-{year_str}-{region}.npy"
FMT_FILENAME_ERA_T2M = "{out_dir}/era5-t2m-am-{year_str}-{region}.npy"


def prep(
    start_date,
    snow,
    solar,
    tb,
    era_ft,
    era_t2m,
    out_dir,
    region,
    drop_bad_days,
    missing_cutoff=0.6,
    periodic=True,
    non_periodic_bias_val=20,
):
    out_dir = os.path.abspath(out_dir)
    n = len(solar)
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
    snow = snow[good_idxs]
    solar = solar[good_idxs]
    tb = tb[good_idxs]
    era_ft = era_ft[good_idxs]
    era_t2m = era_t2m[good_idxs]
    tb = fill_gaps(
        tb, periodic=periodic, non_periodic_bias_val=non_periodic_bias_val
    )
    snow = np.round(
        fill_gaps(
            snow,
            missing_func=is_neg_one,
            periodic=periodic,
            non_periodic_bias_val=non_periodic_bias_val,
        )
    )

    start_year = dates[0].year
    end_year = dates[-1].year
    year_str = get_year_str(start_year, end_year)
    data_dict = {
        FMT_FILENAME_SNOW: snow,
        FMT_FILENAME_SOLAR: solar,
        FMT_FILENAME_TB: tb,
        FMT_FILENAME_ERA_FT: era_ft,
        FMT_FILENAME_ERA_T2M: era_t2m,
    }
    save_data(data_dict, out_dir, year_str, region)
    with open(f"{out_dir}/date_map-{year_str}-{region}.csv", "w") as fd:
        for i, d in zip(good_idxs, dates):
            fd.write(f"{i},{d}\n")
    with open(f"{out_dir}/dropped_dates-{year_str}-{region}.csv", "w") as fd:
        for d in dropped_dates:
            fd.write(f"{d}\n")


if __name__ == "__main__":
    AK = "ak"
    NH = "nh"
    N45 = "n45"
    N45W = "n45w"
    GL = "gl"
    reg2trans = {
        AK: AK_VIEW_TRANS,
        NH: NH_VIEW_TRANS,
        N45: N45_VIEW_TRANS,
        N45W: N45W_VIEW_TRANS,
        GL: lambda x: x,
    }
    region = N45W
    transform = reg2trans[region]

    drop_bad_days = False
    train_start_year = 2005
    train_final_year = 2014
    test_year = 2016

    out_lon, out_lat = [
        transform(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)
    ]

    base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
    out_dir = "../data/cleaned"

    # Training data
    print("Loading snow cover")
    snow = dataset_to_array(
        torch.utils.data.ConcatDataset(
            [
                dh.NpyDataset(f"../data/snow/snow_cover_{y}.npy", transform)
                for y in range(train_start_year, train_final_year + 1)
            ]
        )
    )
    print("Loading solar")
    solar = dataset_to_array(
        torch.utils.data.ConcatDataset(
            [
                dh.NpyDataset(
                    f"../data/solar/solar_rad-daily-{y}.npy", transform
                )
                for y in range(train_start_year, train_final_year + 1)
            ]
        )
    )
    path_groups = [
        glob.glob(f"../data/tb/{y}/tb_{y}_F*_ML_D*.nc")
        for y in range(train_start_year, train_final_year + 1)
    ]
    print("Loading tb")
    tb = dataset_to_array(build_tb_ds(path_groups, transform))
    print("Loading ERA")
    era_ft = dataset_to_array(
        dh.TransformPipelineDataset(
            dh.ERA5BidailyDataset(
                [
                    f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                    for y in range(train_start_year, train_final_year + 1)
                ],
                "t2m",
                "AM",
                out_lon,
                out_lat,
            ),
            [dh.FTTransform()],
        )
    )
    era_t2m = dataset_to_array(
        dh.ERA5BidailyDataset(
            [
                f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                for y in range(train_start_year, train_final_year + 1)
            ],
            "t2m",
            "AM",
            out_lon,
            out_lat,
        ),
    )
    prep(
        dt.date(train_start_year, 1, 1),
        snow,
        solar,
        tb,
        era_ft,
        era_t2m,
        out_dir,
        region,
        drop_bad_days,
    )

    # Validation data
    print("Loading snow cover")
    snow = dataset_to_array(
        dh.NpyDataset(f"../data/snow/snow_cover_{test_year}.npy", transform)
    )
    print("Loading solar")
    solar = dataset_to_array(
        dh.NpyDataset(
            f"../data/solar/solar_rad-daily-{test_year}.npy", transform
        )
    )
    print("Loading tb")
    tb = dataset_to_array(
        build_tb_ds(
            [glob.glob(f"../data/tb/{test_year}/tb_{test_year}_F17_ML_D*.nc")],
            transform,
        )
    )
    print("Loading ERA")
    era_ft = dataset_to_array(
        dh.TransformPipelineDataset(
            dh.ERA5BidailyDataset(
                [f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{test_year}.nc"],
                "t2m",
                "AM",
                out_lon,
                out_lat,
            ),
            [transform, dh.FTTransform()],
        )
    )
    era_t2m = dataset_to_array(
        dh.TransformPipelineDataset(
            dh.ERA5BidailyDataset(
                [f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{test_year}.nc"],
                "t2m",
                "AM",
                out_lon,
                out_lat,
            ),
            [transform],
        )
    )
    prep(
        dt.date(test_year, 1, 1),
        snow,
        solar,
        tb,
        era_ft,
        era_t2m,
        out_dir,
        region,
        drop_bad_days,
    )
