import argparse
import numpy as np
import os

import ease_grid as eg
from datahandling import get_aws_data, load_dates, persist_data_object
from transforms import GL, REGION_CODES, REGION_TO_TRANS
from utils import validate_file_path, validate_dir_path
from validate import RETRIEVAL_MIN, RETRIEVAL_MAX


def _validate_region(reg):
    if reg in REGION_CODES:
        return reg
    raise ValueError(
        f"Invalid region code: {reg}\nValid options are: {REGION_CODES}"
    )


def get_year_str(ya, yb):
    if ya == yb:
        return str(ya)
    else:
        ya, yb = sorted([ya, yb])
        return f"{ya}-{yb}"


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-r",
        "--region",
        action="store",
        default=GL,
        type=_validate_region,
        help="Region to use when querying the database. Default is GL.",
    )
    p.add_argument(
        "start_year",
        type=int,
        help="Start year for queryies. Can be the same as end_year.",
    )
    p.add_argument(
        "end_year",
        type=int,
        help="End year for queryies, inclusive. Can be the same as end_year.",
    )
    p.add_argument("out_dir", type=validate_dir_path, help="Output location")
    return p


def main(args):
    if args.start_year > args.end_year:
        raise ValueError("Start year must be less than or equal to end_year")
    year_str = get_year_str(args.start_year, args.end_year)
    transform = REGION_TO_TRANS[args.region]
    dates = load_dates(
        f"../data/cleaned/date_map-{year_str}-{args.region}.csv"
    )
    # TODO: add option for AM/PM
    db_path = "../data/dbs/wmo_gsod.db"
    land_mask = ~transform(np.load("../data/masks/ft_esdr_water_mask.npy"))
    # TODO: add option for AM/PM
    ret_type = RETRIEVAL_MIN
    lon, lat = [transform(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)]
    aws_data = get_aws_data(
        dates,
        db_path,
        land_mask,
        lon,
        lat,
        ret_type,
    )
    # TODO: add option for AM/PM
    out_file = os.path.join(
        args.out_dir, f"aws_data-AM-{year_str}-{args.region}.pkl"
    )
    print(f"Saving data to '{out_file}'")
    persist_data_object(aws_data, out_file)


if __name__ == "__main__":
    main(get_cli_parser().parse_args())
