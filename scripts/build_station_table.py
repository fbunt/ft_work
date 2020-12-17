import datetime as dt
import numpy as np
import pandas as pd
import tqdm

import datahandling as dh
import ease_grid as eg
from validate import RETRIEVAL_MIN
from validation_db_orm import (
    date_to_int,
    get_db_session,
    DbWMOMetDailyTempRecord,
    DbWMOMetStation,
)
from transforms import N45_VIEW_TRANS as transform


pred = np.load("../runs/gscc/2020-11-02-19:51:31.769613-gold/pred.npy")
land_mask = ~transform(np.load("../data/masks/ft_esdr_water_mask.npy"))

dates, _, _ = dh.read_accuracies_file(
    "../runs/gscc/2020-11-02-19:51:31.769613-gold/acc.csv"
)
d2i = {d: i for i, d in enumerate(dates)}
assert len(dates) == len(pred)
d2p = {d: p.ravel() for d, p in zip(dates, pred)}
lon, lat = [transform(x) for x in eg.v1_get_full_grid_lonlat(eg.ML)]

df = dh.get_aws_full_data_for_dates(
    dates,
    "../data/dbs/wmo_gsod.db",
    land_mask,
    lon,
    lat,
    RETRIEVAL_MIN,
)
df = df[df.date != dt.date(2015, 1, 1)]
vres = {
    sid: np.zeros(len(dates), dtype=int) - 1 for sid in sorted(df.sid.unique())
}
ftres = {
    sid: np.zeros(len(dates), dtype=int) - 1 for sid in sorted(df.sid.unique())
}
for sid, group in tqdm.tqdm(df.groupby(by=["sid"]), ncols=80, desc="Val data"):
    for d, dgroup in group.groupby(by=["date"]):
        v = d2p[d][dgroup.flat_grid_idx.iloc[0]] == dgroup.ft.iloc[0]
        vres[sid][d2i[d]] = v
        ftres[sid][d2i[d]] = dgroup.ft.iloc[0]
val_acc = {
    sid: vres[sid][vres[sid] >= 0].sum() / (vres[sid] >= 0).sum()
    for sid in vres
}
vdata = [[sid, val_acc[sid]] + vres[sid].tolist() for sid in vres]
vcs = ["id", "val_acc"] + [f"v{i}" for i in range(len(dates))]
vdf = pd.DataFrame(vdata, columns=vcs)
fcs = ["id"] + [f"ft{i}" for i in range(len(dates))]
ftdata = [[sid] + ftres[sid].tolist() for sid in sorted(ftres.keys())]
ftdf = pd.DataFrame(ftdata, columns=fcs)

bounds = [lon.min(), lon.max(), lat.min(), lat.max()]
db = get_db_session("../data/dbs/wmo_gsod.db")
# queries = {}
# for d in tqdm.tqdm(dates, ncols=80):
#     records = (
#         db.query(DbWMOMetStation.id, DbWMOMetDailyTempRecord.temperature_min)
#         .join(DbWMOMetDailyTempRecord.met_station)
#         .filter(DbWMOMetDailyTempRecord.date_int == date_to_int(d))
#         .filter(DbWMOMetStation.lon >= bounds[0])
#         .filter(DbWMOMetStation.lon <= bounds[1])
#         .filter(DbWMOMetStation.lat >= bounds[2])
#         .filter(DbWMOMetStation.lat <= bounds[3])
#         .order_by(DbWMOMetStation.id)
#         .all()
#     )
#     queries[d] = records
# ftres = {
#     sid: np.full(len(dates), -1, dtype=int) for sid in sorted(vdf.id.unique())
# }
# for d, i in d2i.items():
#     for sid, temp in queries[d]:
#         if sid in ftres and temp is not None:
#             ftres[sid][i] = temp > 273.15
# ftdata = [[sid] + ftres[sid].tolist() for sid in sorted(ftres.keys())]
# fcs = ["id"] + [f"ft{i}" for i in range(len(dates))]
# ftdf = pd.DataFrame(ftdata, columns=fcs)

records = (
    db.query(DbWMOMetStation)
    .join(DbWMOMetDailyTempRecord)
    .filter(DbWMOMetDailyTempRecord.date_int > 20150101)
    .filter(DbWMOMetDailyTempRecord.date_int < 20160101)
    .filter(DbWMOMetStation.lon >= bounds[0])
    .filter(DbWMOMetStation.lon <= bounds[1])
    .filter(DbWMOMetStation.lat >= bounds[2])
    .filter(DbWMOMetStation.lat <= bounds[3])
    .order_by(DbWMOMetStation.id)
    .all()
)
db.close()
meta = {"id": [], "lon": [], "lat": [], "name": [], "elevation": []}
for s in records:
    meta["id"].append(s.id)
    meta["lon"].append(s.lon)
    meta["lat"].append(s.lat)
    meta["name"].append(s.name)
    meta["elevation"].append(s.elevation)
meta = pd.DataFrame(meta, index=range(len(records)))

stns = pd.merge(meta, vdf, how="inner", on="id")
stns = pd.merge(stns, ftdf, how="inner", on="id")
stns.to_csv(
    "../../progress_presentation_fall_2020/data/stations/stations_2015.csv"
)
