import datetime as dt
import glob
import numpy as np
import os
import torch
import tqdm

from transforms import (
    AK,
    GL,
    N20,
    N45,
    N45W,
    NH,
    REGION_TO_TRANS,
)
import datahandling as dh
import ease_grid as eg
from validate import RETRIEVAL_MIN, RETRIEVAL_MAX


def get_year_str(ya, yb):
    if ya == yb:
        return str(ya)
    else:
        ya, yb = sorted([ya, yb])
        return f"{ya}-{yb}"


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


def save_data(data_dict, out_dir, year_str, region, pass_, am_pm):
    for fname, data in data_dict.items():
        name = fname.format(
            out_dir=out_dir,
            year_str=year_str,
            region=region,
            pass_=pass_,
            am_pm=am_pm,
        )
        print(f"Saving to: '{name}'")
        np.save(name, data)


def is_neg_one(x):
    return x == -1


FMT_FILENAME_SNOW = "{out_dir}/snow_cover-{year_str}-{region}.npy"
FMT_FILENAME_SOLAR = "{out_dir}/solar_rad-{am_pm}-{year_str}-{region}.npy"
FMT_FILENAME_TB = "{out_dir}/tb-{pass_}-{year_str}-{region}.npy"
FMT_FILENAME_ERA_FT = "{out_dir}/era5-ft-{am_pm}-{year_str}-{region}.npy"
FMT_FILENAME_ERA_T2M = "{out_dir}/era5-t2m-{am_pm}-{year_str}-{region}.npy"
FMT_FILENAME_FT_LABEL = "{out_dir}/ft_label-{am_pm}-{year_str}-{region}.npy"
FMT_FILENAME_AWS_MASK = "{out_dir}/aws_mask-{am_pm}-{year_str}-{region}.npy"

SNOW_KEY = "snow"
SOLAR_KEY = "solar"
TB_KEY = "tb"
ERA_FT_KEY = "era_ft"
ERA_T2M_KEY = "era_t2m"


def prep(
    start_date,
    data,
    out_dir,
    region,
    land_mask,
    lon_grid,
    lat_grid,
    am_pm,
    db_path,
    drop_bad_days,
    prep_tb,
    missing_cutoff=0.6,
    periodic=True,
    non_periodic_bias_val=20,
):
    out_dir = os.path.abspath(out_dir)
    tb = data[TB_KEY]
    n = len(tb)
    dates = np.array(get_n_dates(start_date, n))
    pass_ = "D" if am_pm == "AM" else "A"
    retrievel = RETRIEVAL_MIN if am_pm == "AM" else RETRIEVAL_MAX

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

    aws_data = dh.get_aws_data(
        dates, db_path, land_mask, lon_grid, lat_grid, retrievel
    )
    if SNOW_KEY in data:
        snow = data[SNOW_KEY][good_idxs]
        snow = np.round(
            fill_gaps(
                snow,
                missing_func=is_neg_one,
                periodic=periodic,
                non_periodic_bias_val=non_periodic_bias_val,
            )
        )
    if SOLAR_KEY in data:
        solar = data[SOLAR_KEY][good_idxs]
    tb = tb[good_idxs]
    era_ft = data[ERA_FT_KEY][good_idxs]
    ft_label = era_ft.copy()
    aws_mask = np.zeros((n, *lat_grid.shape), dtype=bool)
    for i in tqdm.tqdm(range(n), ncols=80, desc="FT Label"):
        ifzn, ithw = aws_data[i]
        aws_mask[i].ravel()[ifzn] = True
        aws_mask[i].ravel()[ithw] = True
        ft_label[i, 0].ravel()[ifzn] = 1
        ft_label[i, 1].ravel()[ithw] = 1
    if ERA_T2M_KEY in data:
        era_t2m = data[ERA_T2M_KEY][good_idxs]
    if prep_tb:
        tb = fill_gaps(
            tb, periodic=periodic, non_periodic_bias_val=non_periodic_bias_val
        )

    start_year = dates[0].year
    end_year = dates[-1].year
    year_str = get_year_str(start_year, end_year)
    out_dict = {
        FMT_FILENAME_ERA_FT: era_ft,
        FMT_FILENAME_FT_LABEL: ft_label,
        FMT_FILENAME_AWS_MASK: aws_mask,
    }
    if prep_tb:
        out_dict[FMT_FILENAME_TB] = tb
    if SNOW_KEY in data:
        out_dict[FMT_FILENAME_SNOW] = snow
    if SOLAR_KEY in data:
        out_dict[FMT_FILENAME_SOLAR] = solar
    if ERA_T2M_KEY in data:
        out_dict[FMT_FILENAME_ERA_T2M] = era_t2m
    save_data(out_dict, out_dir, year_str, region, pass_, am_pm)
    dh.persist_data_object(
        aws_data,
        os.path.join(out_dir, f"aws_data-{am_pm}-{year_str}-{region}.pkl"),
        overwrite=True,
    )
    with open(f"{out_dir}/date_map-{year_str}-{region}.csv", "w") as fd:
        for i, d in zip(good_idxs, dates):
            fd.write(f"{i},{d}\n")
    with open(f"{out_dir}/dropped_dates-{year_str}-{region}.csv", "w") as fd:
        for d in dropped_dates:
            fd.write(f"{d}\n")


if __name__ == "__main__":
    region = NH
    transform = REGION_TO_TRANS[region]

    # NOTE: prep_tb is handled differently. tb data is still loaded but no gap
    # filling is done and the result is not saved.
    prep_tb = True
    prep_snow = False
    prep_solar = False
    prep_era_t2m = False
    am_pm = "AM"
    pass_ = "D" if am_pm == "AM" else "A"

    drop_bad_days = False
    train_start_year = 2005
    train_final_year = 2014
    test_year = 2016

    out_lon, out_lat = [
        transform(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)
    ]

    base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
    water_mask = transform(base_water_mask)
    land_mask = ~water_mask
    db_path = "../data/dbs/wmo_gsod.db"
    out_dir = "../data/cleaned"

    # Training data
    data = {}
    if prep_snow:
        print("Loading snow cover")
        snow = dh.dataset_to_array(
            torch.utils.data.ConcatDataset(
                [
                    dh.NpyDataset(
                        f"../data/snow/snow_cover_{y}.npy", transform
                    )
                    for y in range(train_start_year, train_final_year + 1)
                ]
            )
        )
        data[SNOW_KEY] = snow
    if prep_solar:
        print("Loading solar")
        solar = dh.dataset_to_array(
            torch.utils.data.ConcatDataset(
                [
                    dh.NpyDataset(
                        f"../data/solar/solar_rad-daily-{y}.npy", transform
                    )
                    for y in range(train_start_year, train_final_year + 1)
                ]
            )
        )
        data[SOLAR_KEY] = solar
    path_groups = [
        glob.glob(f"../data/tb/{y}/tb_{y}_F*_ML_{pass_}*.nc")
        for y in range(train_start_year, train_final_year + 1)
    ]
    print("Loading tb")
    tb = dh.dataset_to_array(dh.build_tb_ds(path_groups, transform))
    data[TB_KEY] = tb
    print("Loading ERA")
    era_ft = dh.dataset_to_array(
        dh.TransformPipelineDataset(
            dh.ERA5BidailyDataset(
                [
                    f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                    for y in range(train_start_year, train_final_year + 1)
                ],
                "t2m",
                am_pm,
                out_lon,
                out_lat,
            ),
            [dh.FTTransform()],
        )
    )
    data[ERA_FT_KEY] = era_ft
    if prep_era_t2m:
        era_t2m = dh.dataset_to_array(
            dh.ERA5BidailyDataset(
                [
                    f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                    for y in range(train_start_year, train_final_year + 1)
                ],
                "t2m",
                am_pm,
                out_lon,
                out_lat,
            ),
        )
        data[ERA_T2M_KEY] = era_t2m
    prep(
        dt.date(train_start_year, 1, 1),
        data,
        out_dir,
        region,
        land_mask,
        out_lon,
        out_lat,
        am_pm,
        db_path,
        drop_bad_days,
        prep_tb,
    )

    # Validation data
    data = {}
    if prep_snow:
        print("Loading snow cover")
        snow = dh.dataset_to_array(
            dh.NpyDataset(
                f"../data/snow/snow_cover_{test_year}.npy", transform
            )
        )
        data[SNOW_KEY] = snow
    if prep_solar:
        print("Loading solar")
        solar = dh.dataset_to_array(
            dh.NpyDataset(
                f"../data/solar/solar_rad-daily-{test_year}.npy", transform
            )
        )
        data[SOLAR_KEY] = solar
    print("Loading tb")
    tb = dh.dataset_to_array(
        dh.build_tb_ds(
            [
                glob.glob(
                    f"../data/tb/{test_year}/tb_{test_year}_F17_ML_{pass_}*.nc"
                )
            ],
            transform,
        )
    )
    data[TB_KEY] = tb
    print("Loading ERA")
    era_ft = dh.dataset_to_array(
        dh.TransformPipelineDataset(
            dh.ERA5BidailyDataset(
                [f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{test_year}.nc"],
                "t2m",
                am_pm,
                out_lon,
                out_lat,
            ),
            [dh.FTTransform()],
        )
    )
    data[ERA_FT_KEY] = era_ft
    if prep_era_t2m:
        era_t2m = dh.dataset_to_array(
            dh.ERA5BidailyDataset(
                [f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{test_year}.nc"],
                "t2m",
                am_pm,
                out_lon,
                out_lat,
            )
        )
        data[ERA_T2M_KEY] = era_t2m
    prep(
        dt.date(test_year, 1, 1),
        data,
        out_dir,
        region,
        land_mask,
        out_lon,
        out_lat,
        am_pm,
        db_path,
        drop_bad_days,
        prep_tb,
    )
