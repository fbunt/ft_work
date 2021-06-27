import multiprocessing as mp
import numpy as np
import pandas as pd
import tqdm
from itertools import repeat

import ease_grid as eg
import datetime as dt
from transforms import REGION_TO_TRANS, NH, GL, SH
from utils import date_range
from validate import validate_against_aws_db
from validation_db_orm import get_db_session


def load_total_val(path):
    return pd.read_csv(
        path, index_col=0, parse_dates=True, header=[0, 1, 2, 3]
    )


CC_MASK = 0
LAND_MASK = 1
INV_CC_MASK = 2


def worker(args):
    (
        pred_path,
        am_pm,
        region,
        mask_code,
        out_path,
    ) = args
    db = get_db_session("../data/dbs/wmo_gsod.db")
    dates = date_range(dt.date(1988, 1, 2), dt.date(2019, 1, 1))
    trans = REGION_TO_TRANS[region]
    lon, lat = [trans(x) for x in eg.v1_get_full_grid_lonlat(eg.ML)]
    land = trans(np.load("../data/masks/ft_esdr_land_mask.npy"))
    water = ~land
    non_cc_mask = trans(
        np.load("../data/masks/ft_esdr_non_cold_constrained_mask.npy")
    )
    invalid = non_cc_mask | water
    cc_mask = ~invalid
    inv_cc_mask = land & ~cc_mask
    mask = None
    if mask_code == CC_MASK:
        mask = cc_mask
    elif mask_code == LAND_MASK:
        mask = land
    else:
        mask = inv_cc_mask

    pred = trans(np.load(pred_path))
    df = validate_against_aws_db(
        pred, db, dates, lon, lat, mask, am_pm, progress=False
    )
    df.to_csv(out_path)


AM, PM = "AM", "PM"


def process_parallel():
    arg_list = [
        # AM NH
        ("../data/gl_am_pred.npy", AM, NH, CC_MASK, "../data/val_new/pred_val_nh_am_cc.csv"),
        ("../data/gl_am_pred.npy", AM, NH, LAND_MASK, "../data/val_new/pred_val_nh_am_full.csv"),
        # PM NH
        ("../data/gl_pm_pred.npy", PM, NH, CC_MASK, "../data/val_new/pred_val_nh_pm_cc.csv"),
        ("../data/gl_pm_pred.npy", PM, NH, LAND_MASK, "../data/val_new/pred_val_nh_pm_full.csv"),
        # AM SH
        ("../data/gl_am_pred.npy", AM, SH, CC_MASK, "../data/val_new/pred_val_sh_am_cc.csv"),
        ("../data/gl_am_pred.npy", AM, SH, LAND_MASK, "../data/val_new/pred_val_sh_am_full.csv"),
        # PM SH
        ("../data/gl_pm_pred.npy", PM, SH, CC_MASK, "../data/val_new/pred_val_sh_pm_cc.csv"),
        ("../data/gl_pm_pred.npy", PM, SH, LAND_MASK, "../data/val_new/pred_val_sh_pm_full.csv"),
        # AM NH FTESDR
        ("../data/ft_esdr/ft_esdr_gl_am_1988-2018.npy", AM, NH, CC_MASK, "../data/val_new/ftesdr_val_nh_am.csv"),
        # PM NH FTESDR
        ("../data/ft_esdr/ft_esdr_gl_pm_1988-2018.npy", PM, NH, CC_MASK, "../data/val_new/ftesdr_val_nh_pm.csv"),
        # AM SH FTESDR
        ("../data/ft_esdr/ft_esdr_gl_am_1988-2018.npy", AM, SH, CC_MASK, "../data/val_new/ftesdr_val_sh_am.csv"),
        # PM SH FTESDR
        ("../data/ft_esdr/ft_esdr_gl_pm_1988-2018.npy", PM, SH, CC_MASK, "../data/val_new/ftesdr_val_sh_pm.csv"),
    ]

    with mp.Pool(processes=len(arg_list)) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(worker, arg_list), ncols=80, total=len(arg_list)
        ):
            pass
    am_nh_cc = pd.read_csv("../data/val_new/pred_val_nh_am_cc.csv", index_col=0, parse_dates=True)
    am_nh_fu = pd.read_csv("../data/val_new/pred_val_nh_am_full.csv", index_col=0, parse_dates=True)
    pm_nh_cc = pd.read_csv("../data/val_new/pred_val_nh_pm_cc.csv", index_col=0, parse_dates=True)
    pm_nh_fu = pd.read_csv("../data/val_new/pred_val_nh_pm_full.csv", index_col=0, parse_dates=True)
    am_sh_cc = pd.read_csv("../data/val_new/pred_val_sh_am_cc.csv", index_col=0, parse_dates=True)
    am_sh_fu = pd.read_csv("../data/val_new/pred_val_sh_am_full.csv", index_col=0, parse_dates=True)
    pm_sh_cc = pd.read_csv("../data/val_new/pred_val_sh_pm_cc.csv", index_col=0, parse_dates=True)
    pm_sh_fu = pd.read_csv("../data/val_new/pred_val_sh_pm_full.csv", index_col=0, parse_dates=True)
    am_gl_cc = am_nh_cc + am_sh_cc
    pm_gl_cc = pm_nh_cc + pm_sh_cc
    am_gl_fu = am_nh_fu + am_sh_fu
    pm_gl_fu = pm_nh_fu + pm_sh_fu
    # ftesdr
    am_nh_ft = pd.read_csv("../data/val_new/ftesdr_val_nh_am.csv", index_col=0, parse_dates=True)
    pm_nh_ft = pd.read_csv("../data/val_new/ftesdr_val_nh_pm.csv", index_col=0, parse_dates=True)
    am_sh_ft = pd.read_csv("../data/val_new/ftesdr_val_sh_am.csv", index_col=0, parse_dates=True)
    pm_sh_ft = pd.read_csv("../data/val_new/ftesdr_val_sh_pm.csv", index_col=0, parse_dates=True)
    am_gl_ft = am_nh_ft + am_sh_ft
    pm_gl_ft = pm_nh_ft + pm_sh_ft


    names = ["time", "region", "subset", "dataset"]
    for k, n in zip(["AM", "cc", "nh", "pred"], names):
        am_nh_cc = pd.concat({k: am_nh_cc}, names=[n], axis=1)
    for k, n in zip(["AM", "full", "nh", "pred"], names):
        am_nh_fu = pd.concat({k: am_nh_fu}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "nh", "pred"], names):
        pm_nh_cc = pd.concat({k: pm_nh_cc}, names=[n], axis=1)
    for k, n in zip(["PM", "full", "nh", "pred"], names):
        pm_nh_fu = pd.concat({k: pm_nh_fu}, names=[n], axis=1)
    for k, n in zip(["AM", "cc", "sh", "pred"], names):
        am_sh_cc = pd.concat({k: am_sh_cc}, names=[n], axis=1)
    for k, n in zip(["AM", "full", "sh", "pred"], names):
        am_sh_fu = pd.concat({k: am_sh_fu}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "sh", "pred"], names):
        pm_sh_cc = pd.concat({k: pm_sh_cc}, names=[n], axis=1)
    for k, n in zip(["PM", "full", "sh", "pred"], names):
        pm_sh_fu = pd.concat({k: pm_sh_fu}, names=[n], axis=1)

    for k, n in zip(["AM", "cc", "gl", "pred"], names):
        am_gl_cc = pd.concat({k: am_gl_cc}, names=[n], axis=1)
    for k, n in zip(["AM", "full", "gl", "pred"], names):
        pm_gl_fu = pd.concat({k: pm_gl_fu}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "gl", "pred"], names):
        pm_gl_cc = pd.concat({k: pm_gl_cc}, names=[n], axis=1)
    for k, n in zip(["PM", "full", "gl", "pred"], names):
        am_gl_fu = pd.concat({k: am_gl_fu}, names=[n], axis=1)

    for k, n in zip(["AM", "cc", "nh", "esdr"], names):
        am_nh_ft = pd.concat({k: am_nh_ft}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "nh", "esdr"], names):
        pm_gl_ft = pd.concat({k: pm_nh_ft}, names=[n], axis=1)
    for k, n in zip(["AM", "cc", "sh", "esdr"], names):
        am_sh_ft = pd.concat({k: am_sh_ft}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "sh", "esdr"], names):
        pm_gl_ft = pd.concat({k: pm_nh_ft}, names=[n], axis=1)
    for k, n in zip(["AM", "cc", "gl", "esdr"], names):
        am_gl_ft = pd.concat({k: am_gl_ft}, names=[n], axis=1)
    for k, n in zip(["PM", "cc", "gl", "esdr"], names):
        pm_gl_ft = pd.concat({k: pm_gl_ft}, names=[n], axis=1)

    df = am_gl_fu
    df = df.merge(pm_gl_fu, left_index=True, right_index=True)
    df = df.merge(am_gl_cc, left_index=True, right_index=True)
    df = df.merge(pm_gl_cc, left_index=True, right_index=True)
    df = df.merge(am_nh_fu, left_index=True, right_index=True)
    df = df.merge(pm_nh_fu, left_index=True, right_index=True)
    df = df.merge(am_nh_cc, left_index=True, right_index=True)
    df = df.merge(pm_nh_cc, left_index=True, right_index=True)
    df = df.merge(am_sh_fu, left_index=True, right_index=True)
    df = df.merge(pm_sh_fu, left_index=True, right_index=True)
    df = df.merge(am_sh_cc, left_index=True, right_index=True)
    df = df.merge(pm_sh_cc, left_index=True, right_index=True)

    df = df.merge(am_nh_ft, left_index=True, right_index=True)
    df = df.merge(pm_nh_ft, left_index=True, right_index=True)
    df = df.merge(am_sh_ft, left_index=True, right_index=True)
    df = df.merge(pm_sh_ft, left_index=True, right_index=True)
    df = df.merge(am_gl_ft, left_index=True, right_index=True)
    df = df.merge(pm_gl_ft, left_index=True, right_index=True)
    df.to_csv("../data/val_new/total_validation_pred_esdr_1988-2018.csv")


if __name__ == "__main__":
    process_parallel()
