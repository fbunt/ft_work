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


def prep_data(start_date, solar_ds, tb_ds, era_ds, out_dir, region):
    out_dir = os.path.abspath(out_dir)
    solar = []
    tb = []
    era = []
    vmask = []
    d = dt.date(start_date.year, start_date.month, start_date.day)
    delta = dt.timedelta(days=1)
    dates = []
    dropped_dates = []
    idx = []
    it = tqdm.tqdm(
        enumerate(zip(solar_ds, tb_ds, era_ds)), ncols=80, total=len(era_ds)
    )
    for i, (sri, tbi, erai) in it:
        tbi = tbi.numpy()
        nan = np.isnan(tbi)
        if not nan.all():
            for chan, na in zip(tbi, nan):
                chan[na] = np.nanmean(chan)
            solar.append(sri)
            tb.append(tbi)
            era.append(erai)
            m = (~nan)[0]
            vmask.append(m)
            dates.append(d)
            idx.append(i)
        else:
            it.write(f"Dropping: {d}")
            dropped_dates.append(d)
        d += delta
    solar = np.stack(solar, axis=0)
    tb = np.stack(tb, axis=0)
    era = np.stack(era, axis=0)
    vmask = np.stack(vmask, axis=0)

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
        for i, d in zip(idx, dates):
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
solar_ds = torch.utils.data.ConcatDataset(
    [
        dl.NpyDataset(f"../data/solar/solar_rad-daily-2007-{region}.npy"),
        dl.NpyDataset(f"../data/solar/solar_rad-daily-2008-{region}.npy"),
        dl.NpyDataset(f"../data/solar/solar_rad-daily-2009-{region}.npy"),
        dl.NpyDataset(f"../data/solar/solar_rad-daily-2010-{region}.npy"),
    ]
)
path_groups = [
    glob.glob("../data/tb/2007/tb_2007_F17_ML_D*.nc"),
    glob.glob("../data/tb/2008/tb_2008_F17_ML_D*.nc"),
    glob.glob("../data/tb/2009/tb_2009_F17_ML_D*.nc"),
    glob.glob("../data/tb/2010/tb_2010_F17_ML_D*.nc"),
]
tb_ds = build_tb_ds(path_groups, transform)
era_ds = dl.ERA5BidailyDataset(
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
prep_data(dt.date(2007, 1, 1), solar_ds, tb_ds, era_ds, out_dir, region)


# Validation data
solar_ds = dl.NpyDataset("../data/solar/solar_rad-daily-2015-{region}.npy")
tb_ds = build_tb_ds(
    [glob.glob("../data/tb/2015/tb_2015_F17_ML_D*.nc")], transform
)
era_ds = dl.ERA5BidailyDataset(
    ["../data/era5/t2m/bidaily/era5-t2m-bidaily-2015.nc"],
    "t2m",
    "AM",
    other_mask=base_water_mask,
    transform=transform,
)
prep_data(dt.date(2015, 1, 1), solar_ds, tb_ds, era_ds, out_dir)
