import datetime as dt
import glob
import numpy as np
import torch
import tqdm

import dataloading as dl


transform = dl.ViewCopyTransform(15, 62, 12, 191)
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")

# Training data
# Solar
solar_ds = torch.utils.data.ConcatDataset(
    [
        dl.NpyDataset("../data/solar/solar_rad-2007-ak.npy"),
        dl.NpyDataset("../data/solar/solar_rad-2008-ak.npy"),
        dl.NpyDataset("../data/solar/solar_rad-2009-ak.npy"),
        dl.NpyDataset("../data/solar/solar_rad-2010-ak.npy"),
    ]
)
# TB
tb_files = sorted(glob.glob("../data/tb/2007/tb_2007_F17_ML_D*.nc"))
tds07 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
tb_files = sorted(glob.glob("../data/tb/2008/tb_2008_F17_ML_D*.nc"))
tds08 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
tb_files = sorted(glob.glob("../data/tb/2009/tb_2009_F17_ML_D*.nc"))
tds09 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
tb_files = sorted(glob.glob("../data/tb/2010/tb_2010_F17_ML_D*.nc"))
tds10 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
tb_ds = torch.utils.data.ConcatDataset([tds07, tds08, tds09, tds10])

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

solar = []
tb = []
era = []
vmask = []
d = dt.date(2007, 1, 1)
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

np.save("../data/cleaned/solar_rad-2007-2010-AM-ak.npy", solar)
np.save("../data/cleaned/tb-2007-2010-D-ak.npy", tb)
np.save("../data/cleaned/era5-t2m-am-2007-2010-ak.npy", era)
np.save("../data/cleaned/tb_valid_mask-2007-2010-D-ak.npy", vmask)
with open("../data/cleaned/date_map-2007-2010.csv", "w") as fd:
    for i, d in zip(idx, dates):
        fd.write(f"{i},{d}\n")
with open("../data/cleaned/dropped_dates-2007-2010.csv", "w") as fd:
    for d in dropped_dates:
        fd.write(f"{d}\n")
tb = None
era = None
tb_ds = None
era_ds = None


# Validation data
solar_ds = dl.NpyDataset("../data/solar/solar_rad-2015-ak.npy")
tb_files = sorted(glob.glob("../data/tb/2015/tb_2015_F17_ML_D*.nc"))
tds15 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
era_ds = dl.ERA5BidailyDataset(
    ["../data/era5/t2m/bidaily/era5-t2m-bidaily-2015.nc"],
    "t2m",
    "AM",
    other_mask=base_water_mask,
    transform=transform,
)

solar = []
tb = []
era = []
vmask = []
d = dt.date(2015, 1, 1)
delta = dt.timedelta(days=1)
dates = []
dropped_dates = []
idx = []
it = tqdm.tqdm(
    enumerate(zip(solar_ds, tds15, era_ds)), ncols=80, total=len(era_ds)
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

np.save("../data/cleaned/solar_rad-2015-AM-ak.npy", solar)
np.save("../data/cleaned/tb-2015-D-ak.npy", tb)
np.save("../data/cleaned/era5-t2m-am-2015-ak.npy", era)
np.save("../data/cleaned/tb_valid_mask-2015-D-ak.npy", vmask)
with open("../data/cleaned/date_map-2015.csv", "w") as fd:
    for i, d in zip(idx, dates):
        fd.write(f"{i},{d}\n")
with open("../data/cleaned/dropped_dates-2015.csv", "w") as fd:
    for d in dropped_dates:
        fd.write(f"{d}\n")
