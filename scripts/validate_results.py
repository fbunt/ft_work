import numpy as np
import pandas as pd
from itertools import repeat

import ease_grid as eg
import datetime as dt
from transforms import NH_VIEW_TRANS as trans
from utils import date_range
from validate import validate_against_aws_db
from validation_db_orm import get_db_session


def load_total_val(path):
    return pd.read_csv(
        path, index_col=0, parse_dates=True, header=[0, 1, 2, 3]
    )


if __name__ == "__main__":
    db = get_db_session("../data/dbs/wmo_gsod.db")
    dates = date_range(dt.date(1988, 1, 2), dt.date(2019, 1, 1))
    lon, lat = [trans(x) for x in eg.v1_get_full_grid_lonlat(eg.ML)]
    land = np.load("../data/masks/ft_esdr_land_mask_nh.npy")
    water = ~land
    non_cc_mask = np.load(
        "../data/masks/ft_esdr_non_cold_constrained_mask-nh.npy"
    )
    invalid = non_cc_mask | water
    cc_mask = ~invalid
    inv_cc_mask = land & ~cc_mask

    pred = np.load("../data/nh_am_pred.npy")
    am_cc = validate_against_aws_db(pred, db, dates, lon, lat, cc_mask, "AM")
    am_cc.to_csv("../data/pred_val_nh_am_cc.csv")
    am_full = validate_against_aws_db(pred, db, dates, lon, lat, land, "AM")
    am_full.to_csv("../data/pred_val_nh_am_tot.csv")
    am_non_cc = validate_against_aws_db(
        pred, db, dates, lon, lat, inv_cc_mask, "AM"
    )
    am_non_cc.to_csv("../data/pred_val_nh_am_non_cc.csv")

    pred = np.load("../data/nh_pm_pred.npy")
    pm_cc = validate_against_aws_db(pred, db, dates, lon, lat, cc_mask, "PM")
    pm_cc.to_csv("../data/pred_val_nh_pm_cc.csv")
    pm_full = validate_against_aws_db(pred, db, dates, lon, lat, land, "PM")
    pm_full.to_csv("../data/pred_val_nh_pm_tot.csv")
    pm_non_cc = validate_against_aws_db(
        pred, db, dates, lon, lat, inv_cc_mask, "PM"
    )
    pm_non_cc.to_csv("../data/pred_val_nh_pm_non_cc.csv")
    pred = 0

    names = ["time", "subset", "dataset"]
    for k, n in zip(["AM", "cc", "pred"], names):
        am_cc = pd.concat({k: am_cc}, names=[n], axis=1)
    for k, n in zip(["AM", "non_cc", "pred"], names):
        am_non_cc = pd.concat({k: am_non_cc}, names=[n], axis=1)
    for k, n in zip(["AM", "full", "pred"], names):
        am_full = pd.concat({k: am_full}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "pred"], names):
        pm_cc = pd.concat({k: pm_cc}, names=[n], axis=1)
    for k, n in zip(["AM", "non_cc", "pred"], names):
        pm_non_cc = pd.concat({k: pm_non_cc}, names=[n], axis=1)
    for k, n in zip(["PM", "full", "pred"], names):
        pm_full = pd.concat({k: pm_full}, names=[n], axis=1)
    df = am_full.merge(pm_full, left_index=True, right_index=True)
    df = df.merge(am_cc, left_index=True, right_index=True)
    df = df.merge(pm_cc, left_index=True, right_index=True)

    pred = np.load("../data/ft_esdr/ft_esdr_nh_am_1988-2018.npy")[1:]
    am_esdr = validate_against_aws_db(pred, db, dates, lon, lat, cc_mask, "AM")
    am_esdr.to_csv("../data/ftesdr_val_nh_am.csv")

    pred = np.load("../data/ft_esdr/ft_esdr_nh_pm_1988-2018.npy")[1:]
    pm_esdr = validate_against_aws_db(pred, db, dates, lon, lat, cc_mask, "PM")
    pm_esdr.to_csv("../data/ftesdr_val_nh_pm.csv")
    pred = 0
    for k, n in zip(["AM", "cc", "esdr"], names):
        am_esdr = pd.concat({k: am_esdr}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "esdr"], names):
        pm_esdr = pd.concat({k: pm_esdr}, names=[n], axis=1)
    df = df.merge(am_esdr, left_index=True, right_index=True)
    df = df.merge(pm_esdr, left_index=True, right_index=True)
    df.to_csv("../data/total_validation_pred_esdr_1988-2018.csv")
