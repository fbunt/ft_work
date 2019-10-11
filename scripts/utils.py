import datetime as dt
import numpy as np


def day_of_year_to_datetime(year, day_of_year):
    dt_y = np.datetime64(year, dtype="datetime64[Y]")
    date = dt_y + np.timedelta64(int(day_of_year) - 1, "D")
    d = date.astype("O")
    return dt.datetime(d.year, d.month, d.day)


def day_of_year_to_date(year, day_of_year):
    dt_y = np.datetime64(year, dtype="datetime64[Y]")
    date = dt_y + np.timedelta64(int(day_of_year) - 1, "D")
    return date.astype("O")
