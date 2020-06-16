import datetime as dt
import glob
import numpy as np
import os
import torch
import tqdm

from transforms import AK_VIEW_TRANS, NH_VIEW_TRANS
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


def dataset_to_array(ds):
    n = len(ds)
    shape = (n, *ds[0].shape)
    ar = np.zeros(shape)
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


def fill_gaps(x, nanmask):
    cleaned = x.copy()
    j = 0
    while True:
        for i in tqdm.tqdm(range(len(x)), ncols=80, desc=f"Gap Fill {j + 1}"):
            m = nanmask[i]
            if m.any():
                left = i - 1
                right = i + 1
                if right == len(x):
                    right = 0
                cleaned[i][m] = np.nanmean([x[left][m], x[right][m]], axis=0)
        nanmask = np.isnan(cleaned)
        if not nanmask.any():
            break
        x = cleaned.copy()
        j += 1
    return cleaned


def prep(start_date, solar, tb, era, out_dir, region):
    out_dir = os.path.abspath(out_dir)
    n = len(solar)
    dates = np.array(get_n_dates(start_date, n))
    # Filter out indices where all Tb data is missing
    good_idxs = [i for i in range(n) if not np.isnan(tb[i]).all()]
    bad_idxs = [i for i in range(n) if np.isnan(tb[i]).all()]
    n = len(good_idxs)
    dropped_dates = dates[bad_idxs]
    dates = dates[good_idxs]
    solar = solar[good_idxs]
    tb = tb[good_idxs]
    era = era[good_idxs]
    nanmask = np.isnan(tb)
    tb = fill_gaps(tb, nanmask)
    # Only need one of the identical masks
    nanmask = nanmask[:, 0]
    vmask = nanmask

    start_year = dates[0].year
    end_year = dates[-1].year
    year_str = ""
    if start_year == end_year:
        year_str = str(start_year)
    else:
        year_str = f"{start_year}-{end_year}"
    np.save(f"{out_dir}/solar_rad-{year_str}-AM-{region}.npy", solar)
    np.save(f"{out_dir}/tb-{year_str}-D-{region}.npy", tb)
    np.save(f"{out_dir}/era5-t2m-am-{year_str}-{region}.npy", era)
    np.save(f"{out_dir}/tb_valid_mask-{year_str}-D-{region}.npy", vmask)
    with open(f"{out_dir}/date_map-{year_str}.csv", "w") as fd:
        for i, d in zip(good_idxs, dates):
            fd.write(f"{i},{d}\n")
    with open(f"{out_dir}/dropped_dates-{year_str}.csv", "w") as fd:
        for d in dropped_dates:
            fd.write(f"{d}\n")


AK = "ak"
NH = "nh"
reg2trans = {AK: AK_VIEW_TRANS, NH: NH_VIEW_TRANS}
region = AK
transform = reg2trans[region]

base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
out_dir = "../data/cleaned"

# Training data
print("Loading solar")
solar = dataset_to_array(
    torch.utils.data.ConcatDataset(
        [
            dl.NpyDataset(f"../data/solar/solar_rad-daily-2007-{region}.npy"),
            dl.NpyDataset(f"../data/solar/solar_rad-daily-2008-{region}.npy"),
            dl.NpyDataset(f"../data/solar/solar_rad-daily-2009-{region}.npy"),
            dl.NpyDataset(f"../data/solar/solar_rad-daily-2010-{region}.npy"),
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
        other_mask=base_water_mask,
        transform=transform,
    )
)
prep(dt.date(2007, 1, 1), solar, tb, era, out_dir, region)


# Validation data
print("Loading solar")
solar = dataset_to_array(
    dl.NpyDataset("../data/solar/solar_rad-daily-2015-{region}.npy")
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
        other_mask=base_water_mask,
        transform=transform,
    )
)
prep(dt.date(2015, 1, 1), solar, tb, era, out_dir)
