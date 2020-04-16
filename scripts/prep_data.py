import datetime as dt
import glob
import numpy as np
import torch

import dataloading as dl


transform = dl.ViewCopyTransform(15, 62, 12, 191)
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")

# Training data
tb_files = sorted(glob.glob("../data/tb/tb_2008_F17_ML_D*.nc"))
tds08 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
tb_files = sorted(glob.glob("../data/tb/tb_2009_F17_ML_D*.nc"))
tds09 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
tb_ds = torch.utils.data.ConcatDataset([tds08, tds09])

era_ds = dl.ERA5BidailyDataset(
    [
        "../data/era5/t2m/bidaily/era5-t2m-bidaily-2008.nc",
        "../data/era5/t2m/bidaily/era5-t2m-bidaily-2009.nc",
    ],
    "t2m",
    "AM",
    other_mask=base_water_mask,
    transform=transform,
)

tb = []
era = []
d = dt.date(2008, 1, 1)
delta = dt.timedelta(days=1)
dates = []
dropped_dates = []
idx = []
for i, (tbi, erai) in enumerate(zip(tb_ds, era_ds)):
    if not np.isnan(tbi).all():
        tbi = tbi.numpy()
        tbi[np.isnan(tbi)] = 0
        tb.append(tbi)
        era.append(erai)
        dates.append(d)
        idx.append(i)
    else:
        print(f"Dropping: {d}")
        dropped_dates.append(d)
    d += delta
tb = np.stack(tb, axis=0)
era = np.stack(era, axis=0)

np.save("../data/cleaned/tb-2008-2009-D-ak.npy", tb)
np.save("../data/cleaned/era5-t2m-am-2008-2009-ak.npy", era)
with open("../data/cleaned/date_map-2008-2009.csv", "w") as fd:
    for i, d in zip(idx, dates):
        fd.write(f"{i},{d}\n")
with open("../data/cleaned/dropped_dates-2008-2009.csv", "w") as fd:
    for d in dropped_dates:
        fd.write(f"{d}\n")


# Validation data
tb_files = sorted(glob.glob("../data/tb/tb_2015_F17_ML_D*.nc"))
tds15 = dl.GridsStackDataset(
    [dl.NCDataset([f], "tb", transform=transform) for f in tb_files]
)
era_ds = dl.ERA5BidailyDataset(
    [
        "../data/era5/t2m/bidaily/era5-t2m-bidaily-2015.nc",
    ],
    "t2m",
    "AM",
    other_mask=base_water_mask,
    transform=transform,
)

tb = []
era = []
d = dt.date(2015, 1, 1)
delta = dt.timedelta(days=1)
dates = []
dropped_dates = []
idx = []
for i, (tbi, erai) in enumerate(zip(tds15, era_ds)):
    if not np.isnan(tbi).all():
        tbi = tbi.numpy()
        tbi[np.isnan(tbi)] = 0
        tb.append(tbi)
        era.append(erai)
        dates.append(d)
        idx.append(i)
    else:
        print(f"Dropping: {d}")
        dropped_dates.append(d)
    d += delta
tb = np.stack(tb, axis=0)
era = np.stack(era, axis=0)

np.save("../data/cleaned/tb-2015-D-ak.npy", tb)
np.save("../data/cleaned/era5-t2m-am-2015-ak.npy", era)
with open("../data/cleaned/date_map-2015.csv", "w") as fd:
    for i, d in zip(idx, dates):
        fd.write(f"{i},{d}\n")
with open("../data/cleaned/dropped_dates-2015.csv", "w") as fd:
    for d in dropped_dates:
        fd.write(f"{d}\n")
