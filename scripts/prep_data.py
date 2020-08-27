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
import dataloading as dl


def build_tb_ds(path_groups, transform):
    dss = [
        dl.GridsStackDataset(
            [
                dl.NCDataset([f], "tb", transform=transform)
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


def get_predecessor(x, i, missing):
    px = x[i].copy()
    count = np.zeros(px.shape, dtype=int)
    j = i - 1
    while missing.any():
        px[missing] = x[j, missing]
        count[missing] += 1
        missing = np.isnan(px)
        j -= 1
    idx = count != 0
    return px[idx], count[idx]


def get_successor(x, i, missing):
    sx = x[i].copy()
    count = np.zeros(sx.shape, dtype=int)
    j = i + 1
    if j >= len(x):
        j = 0
    while missing.any():
        sx[missing] = x[j, missing]
        count[missing] += 1
        missing = np.isnan(sx)
        j += 1
        if j >= len(x):
            j = 0
    idx = count != 0
    return sx[idx], count[idx]


def fill_gaps(x, missing_func=np.isnan):
    gap_filled = x.copy()
    for i in tqdm.tqdm(range(len(x)), ncols=80, desc="Gap fill"):
        gaps = missing_func(x[i])
        if not gaps.any():
            continue
        # count is how far the alg had to go to find a value
        # Get past value
        pred, pcount = get_predecessor(x, i, gaps)
        # Get future value
        succ, scount = get_successor(x, i, gaps)
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


FMT_FILENAME_SNOW = "{out_dir}/snow_cover-{year_str}-{region}.npy"
FMT_FILENAME_SOLAR = "{out_dir}/solar_rad-AM-{year_str}-{region}.npy"
FMT_FILENAME_TB = "{out_dir}/tb-D-{year_str}-{region}.npy"
FMT_FILENAME_ERA = "{out_dir}/era5-t2m-am-{year_str}-{region}.npy"
FMT_FILENAME_TB_VALID = "{out_dir}/tb_valid_mask-D-{year_str}-{region}.npy"


def prep(
    start_date,
    snow,
    solar,
    tb,
    era,
    out_dir,
    region,
    drop_bad_days,
    missing_cutoff=0.6,
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
    era = era[good_idxs]
    tb = fill_gaps(tb)
    snow = np.round(fill_gaps(snow, missing_func=lambda x: x == -1))
    nanmask = np.isnan(tb)
    # Only need one of the identical masks
    nanmask = nanmask[:, 0]
    vmask = ~nanmask

    start_year = dates[0].year
    end_year = dates[-1].year
    year_str = ""
    if start_year == end_year:
        year_str = str(start_year)
    else:
        year_str = f"{start_year}-{end_year}"
    data_dict = {
        FMT_FILENAME_SNOW: snow,
        FMT_FILENAME_SOLAR: solar,
        FMT_FILENAME_TB: tb,
        FMT_FILENAME_ERA: era,
        FMT_FILENAME_TB_VALID: vmask,
    }
    save_data(data_dict, out_dir, year_str, region)
    with open(f"{out_dir}/date_map-{year_str}-{region}.csv", "w") as fd:
        for i, d in zip(good_idxs, dates):
            fd.write(f"{i},{d}\n")
    with open(f"{out_dir}/dropped_dates-{year_str}-{region}.csv", "w") as fd:
        for d in dropped_dates:
            fd.write(f"{d}\n")


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

base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
out_dir = "../data/cleaned"

# Training data
print("Loading snow cover")
snow = dataset_to_array(
    torch.utils.data.ConcatDataset(
        [
            dl.NpyDataset("../data/snow/snow_cover_2007.npy", transform),
            dl.NpyDataset("../data/snow/snow_cover_2008.npy", transform),
            dl.NpyDataset("../data/snow/snow_cover_2009.npy", transform),
            dl.NpyDataset("../data/snow/snow_cover_2010.npy", transform),
        ]
    )
)
print("Loading solar")
solar = dataset_to_array(
    torch.utils.data.ConcatDataset(
        [
            dl.NpyDataset("../data/solar/solar_rad-daily-2007.npy", transform),
            dl.NpyDataset("../data/solar/solar_rad-daily-2008.npy", transform),
            dl.NpyDataset("../data/solar/solar_rad-daily-2009.npy", transform),
            dl.NpyDataset("../data/solar/solar_rad-daily-2010.npy", transform),
        ]
    )
)
path_groups = [
    glob.glob("../data/tb/2007/tb_2007_F17_ML_D*.nc"),
    glob.glob("../data/tb/2008/tb_2008_F17_ML_D*.nc"),
    glob.glob("../data/tb/2009/tb_2009_F17_ML_D*.nc"),
    glob.glob("../data/tb/2010/tb_2010_F17_ML_D*.nc"),
]
print("Loading tb")
tb = dataset_to_array(build_tb_ds(path_groups, transform))
print("Loading ERA")
era = dataset_to_array(
    dl.ERA5BidailyDataset(
        [
            "../data/era5/t2m/bidaily/era5-t2m-bidaily-2007.nc",
            "../data/era5/t2m/bidaily/era5-t2m-bidaily-2008.nc",
            "../data/era5/t2m/bidaily/era5-t2m-bidaily-2009.nc",
            "../data/era5/t2m/bidaily/era5-t2m-bidaily-2010.nc",
        ],
        "t2m",
        "AM",
        other_mask=None,
        transform=transform,
    )
)
prep(dt.date(2007, 1, 1), snow, solar, tb, era, out_dir, region, drop_bad_days)


# Validation data
print("Loading snow cover")
snow = dataset_to_array(
    dl.NpyDataset("../data/snow/snow_cover_2015.npy", transform)
)
print("Loading solar")
solar = dataset_to_array(
    dl.NpyDataset("../data/solar/solar_rad-daily-2015.npy", transform)
)
print("Loading tb")
tb = dataset_to_array(
    build_tb_ds([glob.glob("../data/tb/2015/tb_2015_F17_ML_D*.nc")], transform)
)
print("Loading ERA")
era = dataset_to_array(
    dl.ERA5BidailyDataset(
        ["../data/era5/t2m/bidaily/era5-t2m-bidaily-2015.nc"],
        "t2m",
        "AM",
        other_mask=None,
        transform=transform,
    )
)
prep(dt.date(2015, 1, 1), snow, solar, tb, era, out_dir, region, drop_bad_days)
