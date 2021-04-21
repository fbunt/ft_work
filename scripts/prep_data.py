import argparse
import datetime as dt
import glob
import numpy as np
import os
import re
import torch
import tqdm

from transforms import (
    REGION_CODES,
    REGION_TO_TRANS,
)
import datahandling as dh
import ease_grid as eg
from utils import validate_dir_path, validate_file_path
from validate import RETRIEVAL_MIN, RETRIEVAL_MAX


def get_dates(start_date, end_date):
    d = dt.timedelta(days=1)
    dates = [start_date]
    while dates[-1] < end_date:
        dates.append(dates[-1] + d)
    return dates


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


def get_missing_ratio(x):
    return np.isnan(x).sum() / x.size


def get_predecessor(
    x,
    i,
    missing,
    missing_func=np.isnan,
    periodic=True,
):
    px = x[i].copy()
    count = np.zeros(px.shape, dtype=int)
    skip_loop = False
    j = i - 1
    if j < 0:
        if periodic:
            j = len(x) - 1
        else:
            missing = missing_func(px)
            px[missing] = 0
            count[missing] = -1
            skip_loop = True
    if not skip_loop:
        while missing.any():
            px[missing] = x[j, missing]
            count[missing] += 1
            missing = missing_func(px)
            j -= 1
            if j < 0:
                if periodic:
                    j = len(x) - 1
                else:
                    px[missing] = 0
                    count[missing] = -1
                    break
    idx = count != 0
    return px[idx], count[idx]


def get_successor(
    x,
    i,
    missing,
    missing_func=np.isnan,
    periodic=True,
):
    sx = x[i].copy()
    count = np.zeros(sx.shape, dtype=int)
    skip_loop = False
    j = i + 1
    if j >= len(x):
        if periodic:
            j = 0
        else:
            missing = missing_func(sx)
            sx[missing] = 0
            count[missing] = -1
            skip_loop = True
    if not skip_loop:
        while missing.any():
            sx[missing] = x[j, missing]
            count[missing] += 1
            missing = missing_func(sx)
            j += 1
            if j >= len(x):
                if periodic:
                    j = 0
                else:
                    sx[missing] = 0
                    count[missing] = -1
                    break
    idx = count != 0
    return sx[idx], count[idx]


def fill_gaps(x, missing_func=np.isnan, periodic=True):
    gap_filled = x.copy()
    for i in tqdm.tqdm(range(len(x)), ncols=80, desc="Gap fill"):
        gaps = missing_func(x[i])
        if not gaps.any():
            continue
        # count is how far the alg had to go to find a value
        # Get past value
        pred, pcount = get_predecessor(x, i, gaps, missing_func, periodic)
        # Areas where get_predecessor ran into the edge and periodic
        # edge-handling was turned off
        pedge = pcount == -1
        # Get future value
        succ, scount = get_successor(x, i, gaps, missing_func, periodic)
        # Areas where get_successor ran into the edge and periodic
        # edge-handling was turned off
        sedge = scount == -1
        pedge_any = pedge.any()
        sedge_any = sedge.any()
        assert not (
            pedge_any and sedge_any
        ).any(), "Ran into edge in forward and backword search"
        # [i, gaps] returns a copy instead of a view
        gap_copy = gap_filled[i, gaps]
        # For parts where the edge was hit and periodic handling was turned
        # off, just fill with opposite search's results. Then trim.
        if pedge_any:
            gap_copy[pedge] = succ[pedge]
        if sedge_any:
            gap_copy[sedge] = pred[sedge]
        remaining = ~(pedge | sedge)
        pcount = pcount[remaining]
        pred = pred[remaining]
        scount = scount[remaining]
        succ = succ[remaining]
        # Weighted mean
        total = pcount + scount
        # The predecessor/successor with the higher count should be weighted
        # less and the opposing weight should be 1 - w.
        pweight = 1 - (pcount / total)
        sweight = 1 - (scount / total)
        gap_copy[remaining] = (pweight * pred) + (sweight * succ)
        gap_filled[i, gaps] = gap_copy
    return gap_filled


def save_data(data_dict, out_dir, dates_str, region, pass_, am_pm):
    for fname, data in data_dict.items():
        name = fname.format(
            out_dir=out_dir,
            dates_str=dates_str,
            region=region,
            pass_=pass_,
            am_pm=am_pm,
        )
        print(f"Saving to: '{name}'")
        np.save(name, data)


def is_neg_one(x):
    return x == -1


FMT_FILENAME_SNOW = "{out_dir}/snow_cover-{dates_str}-{region}.npy"
FMT_FILENAME_SOLAR = "{out_dir}/solar_rad-{am_pm}-{dates_str}-{region}.npy"
FMT_FILENAME_TB = "{out_dir}/tb-{pass_}-{dates_str}-{region}.npy"
FMT_FILENAME_ERA_FT = "{out_dir}/era5-ft-{am_pm}-{dates_str}-{region}.npy"
FMT_FILENAME_ERA_T2M = "{out_dir}/era5-t2m-{am_pm}-{dates_str}-{region}.npy"
FMT_FILENAME_FT_LABEL = "{out_dir}/ft_label-{am_pm}-{dates_str}-{region}.npy"
FMT_FILENAME_AWS_MASK = "{out_dir}/aws_mask-{am_pm}-{dates_str}-{region}.npy"

SNOW_KEY = "snow"
SOLAR_KEY = "solar"
TB_KEY = "tb"
ERA_FT_KEY = "era_ft"
ERA_T2M_KEY = "era_t2m"


def prep(
    start_date,
    end_date,
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
):
    dates = np.array(get_dates(start_date, end_date))
    n = len(dates)
    out_dir = os.path.abspath(out_dir)
    pass_ = "D" if am_pm == "AM" else "A"
    retrievel = RETRIEVAL_MIN if am_pm == "AM" else RETRIEVAL_MAX

    tb = data[TB_KEY]
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
            fill_gaps(snow, missing_func=is_neg_one, periodic=periodic)
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
        tb = fill_gaps(tb, periodic=periodic)

    ss = start_date.year if is_first_day_of_year(start_date) else start_date
    es = end_date.year if is_last_day_of_year(end_date) else end_date
    dates_str = f"{ss}_{es}"
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
    save_data(out_dict, out_dir, dates_str, region, pass_, am_pm)
    dh.persist_data_object(
        aws_data,
        os.path.join(out_dir, f"aws_data-{am_pm}-{dates_str}-{region}.pkl"),
        overwrite=True,
    )
    with open(f"{out_dir}/date_map-{dates_str}-{region}.csv", "w") as fd:
        for i, d in zip(good_idxs, dates):
            fd.write(f"{i},{d}\n")
    with open(f"{out_dir}/dropped_dates-{dates_str}-{region}.csv", "w") as fd:
        for d in dropped_dates:
            fd.write(f"{d}\n")


def is_first_day_of_year(date):
    return date.month == 1 and date.day == 1


def is_last_day_of_year(date):
    return date.month == 12 and date.day == 31


def trim_datasets_to_dates(dss, start_date, end_date):
    single = False
    if isinstance(dss, torch.utils.data.Dataset):
        # if input is single dataset rather than a list of them, wrap in list
        dss = [dss]
        single = True
    start_yday = start_date.timetuple().tm_yday
    end_yday = end_date.timetuple().tm_yday
    if start_date.year == end_date.year:
        idxs = list(range(start_yday - 1, end_yday))
        dss = [torch.utils.data.Subset(dss[0], idxs)]
    else:
        if len(dss) == 1:
            # Datasets already joined into one
            diff = end_date - start_date
            # Number of days total
            n = diff.days + 1
            idxs = list(range(start_yday - 1, start_yday - 1 + n))
            dss = [torch.utils.data.Subset(dss[0], idxs)]
        else:
            if not is_first_day_of_year(start_date):
                idxs = list(range(start_yday - 1, len(dss[0])))
                dss[0] = torch.utils.data.Subset(dss[0], idxs)
            if not is_last_day_of_year(end_date):
                idxs = list(range(0, end_yday))
                dss[-1] = torch.utils.data.Subset(dss[-1], idxs)
    if not single:
        return dss
    else:
        return dss[0]


def validate_region(reg):
    if reg in REGION_CODES:
        return reg
    else:
        raise ValueError(f"Unknown region code: '{reg}'")


def _parse_date_arg(s, start=True):
    _YEAR_PATTERN = re.compile(r"^\d{4}$")
    if _YEAR_PATTERN.match(s) is not None:
        if start:
            s = f"{s}-01-01"
        else:
            s = f"{s}-12-31"
    # Parse dates of the form YYYY-MM-DD
    return dt.date.fromisoformat(s)


DEFAULT_DB = "../data/dbs/wmo_gsod.db"


def build_training_command_parser(p):
    p.add_argument(
        "-d",
        "--db_path",
        type=validate_file_path,
        default=DEFAULT_DB,
        help=f"Path to AWS database file. Default: '{DEFAULT_DB}'",
    )
    p.add_argument(
        "-b",
        "--drop_bad_days",
        action="store_true",
        help="Drop days with large amounts of missing data",
    )
    p.add_argument(
        "-t",
        "--prep_tb",
        action="store_true",
        help=(
            "Prepare Tb data. This is handled differently. tb data is still"
            " loaded but no gap filling is done and the result is not saved."
        ),
    )
    p.add_argument(
        "-s",
        "--prep_solar",
        action="store_true",
        help="Prepare total daily solar data.",
    )
    p.add_argument(
        "-n",
        "--prep_snow",
        action="store_true",
        help="Prepare snow cover data.",
    )
    p.add_argument(
        "-e",
        "--prep_era_t2m",
        action="store_true",
        help="Prepare ERA4 t2m data.",
    )
    ropts = (("{}, " * (len(REGION_CODES) - 1)) + "{}").format(
        *sorted(REGION_CODES)
    )
    p.add_argument(
        "region",
        type=validate_region,
        help=f"The region to pull data from. Options are: {ropts}",
    )
    p.add_argument("am_pm", type=str, help="AM or PM data")
    p.add_argument(
        "start_date",
        type=_parse_date_arg,
        help="Data start date or year. If year, then the entire year is used.",
    )
    p.add_argument(
        "end_date",
        type=lambda x: _parse_date_arg(x, False),
        help="Data end date or year. If year, then the entire year is used.",
    )
    p.add_argument("dest", type=validate_dir_path, help="Output directory")
    return p


def build_tbfill_command_parser(p):
    p.add_argument(
        "-r",
        "--region",
        type=validate_region,
        help="The region to process. Default is NH.",
    )
    p.add_argument("am_pm", type=str, help="AM or PM")
    p.add_argument("start_year", type=int, help="First year of data")
    p.add_argument("end_year", type=int, help="Final year of data")
    return p


def get_parser():
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(dest="subcom_name")
    ptrain = build_training_command_parser(subparsers.add_parser("training"))
    ptbfill = build_tbfill_command_parser(subparsers.add_parser("tbfill"))
    return p


def prep_data(
    region,
    start_date,
    end_date,
    am_pm,
    dest="../data/cleaned",
    db_path="../data/dbs/wmo_gsod.db",
    prep_tb=True,
    prep_solar=False,
    prep_snow=False,
    prep_era_t2m=False,
    drop_bad_days=False,
):
    if isinstance(start_date, int):
        start_date = _parse_date_arg(str(start_date))
    if isinstance(end_date, int):
        end_date = _parse_date_arg(str(end_date), False)
    assert start_date < end_date, "Start date must come before end date"

    transform = REGION_TO_TRANS[region]
    pass_ = "D" if am_pm == "AM" else "A"
    out_lon, out_lat = [
        transform(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)
    ]
    base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
    water_mask = transform(base_water_mask)
    land_mask = ~water_mask

    data = {}
    # Snow
    if prep_snow:
        print("Loading snow cover")
        snow = dh.dataset_to_array(
            torch.utils.data.ConcatDataset(
                trim_datasets_to_dates(
                    [
                        dh.NpyDataset(
                            f"../data/snow/snow_cover_{y}.npy", transform
                        )
                        for y in range(start_date.year, end_date.year + 1)
                    ],
                    start_date,
                    end_date,
                )
            )
        )
        data[SNOW_KEY] = snow
    # Solar
    if prep_solar:
        print("Loading solar")
        solar = dh.dataset_to_array(
            torch.utils.data.ConcatDataset(
                trim_datasets_to_dates(
                    [
                        dh.NpyDataset(
                            f"../data/solar/solar_rad-daily-{y}.npy", transform
                        )
                        for y in range(start_date.year, end_date.year + 1)
                    ],
                    start_date,
                    end_date,
                )
            )
        )
        data[SOLAR_KEY] = solar
    # Tb
    # TODO: caching
    path_groups = [
        glob.glob(f"../data/tb/{y}/tb_{y}_F*_ML_{pass_}*.nc")
        for y in range(start_date.year, end_date.year + 1)
    ]
    print("Loading tb")
    tb = dh.dataset_to_array(
        trim_datasets_to_dates(
            build_tb_ds(path_groups, transform), start_date, end_date
        )
    )
    data[TB_KEY] = tb
    # ERA5 FT
    # TODO: caching
    print("Loading ERA")
    era_ft = dh.dataset_to_array(
        trim_datasets_to_dates(
            dh.TransformPipelineDataset(
                dh.ERA5BidailyDataset(
                    [
                        f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                        for y in range(start_date.year, end_date.year + 1)
                    ],
                    "t2m",
                    am_pm,
                    out_lon,
                    out_lat,
                ),
                [dh.FTTransform()],
            ),
            start_date,
            end_date,
        )
    )
    data[ERA_FT_KEY] = era_ft
    # ERA5 t2m
    # TODO: caching
    if prep_era_t2m:
        era_t2m = dh.dataset_to_array(
            trim_datasets_to_dates(
                dh.ERA5BidailyDataset(
                    [
                        f"../data/era5/t2m/bidaily/era5-t2m-bidaily-{y}.nc"
                        for y in range(start_date.year, end_date.year + 1)
                    ],
                    "t2m",
                    am_pm,
                    out_lon,
                    out_lat,
                ),
                start_date,
                end_date,
            )
        )
        data[ERA_T2M_KEY] = era_t2m
    sizes = set(len(d) for d in data.values())
    assert (
        len(sizes) == 1
    ), "All data must be the same length in the time dimension"
    prep(
        start_date,
        end_date,
        data,
        dest,
        region,
        land_mask,
        out_lon,
        out_lat,
        am_pm,
        db_path,
        drop_bad_days,
        prep_tb,
    )


def load_tb_year(year, am_pm, transform):
    pass_ = "D" if am_pm == "AM" else "A"
    year = year
    trans = transform
    files = glob.glob(f"../data/tb/{year}/tb_{year}_F*_ML_{pass_}*.nc")
    print(f"loading {year}")
    return dh.dataset_to_array(build_tb_ds([files], trans))


def tbfill(region, am_pm, start_year, end_year):
    assert start_year <= end_year
    transform = REGION_TO_TRANS[region]
    ycur = load_tb_year(start_year - 1, am_pm, transform)
    ynext = load_tb_year(start_year, am_pm, transform)
    for y in range(start_year, end_year + 1):
        yprev = ycur
        ycur = ynext
        nprev = yprev.shape[0] // 6
        ncur = ycur.shape[0]
        if y < end_year:
            ynext = load_tb_year(y + 1, am_pm, transform)
            nnext = ynext.shape[0] // 6
            n = nprev + ncur + nnext
            unfilled = np.zeros((n, *ycur.shape[1:]))
            unfilled[:nprev] = yprev[-nprev:]
            unfilled[nprev : nprev + ncur] = ycur
            unfilled[-nnext:] = ynext[:nnext]
        else:
            n = nprev + ncur
            unfilled = np.zeros((n, *ycur.shape[1:]))
            unfilled[:nprev] = yprev[-nprev:]
            unfilled[nprev:] = ycur
        print(f"Filling {y}")
        filled = fill_gaps(unfilled, periodic=False)[nprev : nprev + ncur]
        np.save(f"../data/tb/{y}/tb_{y}_{am_pm}_{region}_filled.npy", filled)
        unfilled = None
        filled = None


def main(args):
    args = vars(args)
    com = args.pop("subcom_name")
    if com == "training":
        prep_data(**args)
    elif com == "tbfill":
        tbfill(**args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
