import datetime as dt
import itertools
import numpy as np
import os


def day_of_year_to_date(year, day_of_year):
    dt_y = np.datetime64(year, dtype="datetime64[Y]")
    date = dt_y + np.timedelta64(int(day_of_year) - 1, "D")
    return date.astype("O")


def day_of_year_to_datetime(year_str, day_of_year_str):
    d = day_of_year_to_date(year_str, day_of_year_str)
    return dt.datetime(d.year, d.month, d.day)


def validate_file_path(path):
    if os.path.isfile(path):
        return path
    raise IOError(f"Could not find file: '{path}'")


def validate_dir_path(path):
    if os.path.isdir(path):
        return path
    raise IOError(f"Could not find dir: '{path}'")


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def flatten_to_iterable(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))
