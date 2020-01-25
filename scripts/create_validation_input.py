import argparse
import datetime as dt
import numpy
import os
import re

from utils import validate_file_path


FILE_REGEX = re.compile("\w(\\d+)\.npy")


def get_dates(start_year, flist, offset=0):
    start = dt.datetime(start_year, 1, 1)
    dates = []
    for f in flist:
        doy = int(FILE_REGEX.search(f).groups()[0]) + offset
        dates.append(start + dt.timedelta(days=doy))
    return dates


def create(fd, flist, start_year, offset):
    dates = get_dates(start_year, flist, offset)
    pairs = sorted(zip(flist, dates), key=lambda x: x[1])
    for p in pairs:
        path, date = p
        fd.write(f"{path}\t{date.isoformat()}\n")


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument("start_year", type=int, help="The year of the files")
    p.add_argument("out_file", type=str, help="File to write output to")
    p.add_argument(
        "input_files", nargs="+", type=validate_file_path, help="Input files"
    )
    p.add_argument(
        "-o", "--offset", type=int, default=0, help="Day of year offset"
    )
    return p


def main(args):
    flist = args.input_files
    start_year = args.start_year
    offset = args.offset
    if os.path.isfile(args.out_file):
        print("Output file already exists. Exiting")
        return
    with open(args.out_file, "w") as fd:
        create(fd, flist, start_year, offset)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
