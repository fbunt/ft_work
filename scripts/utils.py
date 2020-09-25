from matplotlib.colors import ListedColormap
import datetime as dt
import itertools
import numpy as np
import os

FT_CMAP = ListedColormap(
    [
        # Light blue: frozen
        "lightskyblue",
        # Olive: thawed
        "olive",
        # Blue: other/water
        (0, 0, 1),
    ]
)


def day_of_year_to_date(year, day_of_year):
    dt_y = np.datetime64(year, dtype="datetime64[Y]")
    date = dt_y + np.timedelta64(int(day_of_year) - 1, "D")
    return date.astype("O")


def day_of_year_to_datetime(year_str, day_of_year_str):
    d = day_of_year_to_date(year_str, day_of_year_str)
    return dt.datetime(d.year, d.month, d.day)


def datetime64_year(ndt):
    return ndt.astype("datetime64[Y]").astype(int) + 1970


def datetime64_month(ndt):
    return ndt.astype("datetime64[M]").astype(int) % 12 + 1


def datetime64_day(ndt):
    return (ndt.astype("datetime64[D]") - ndt.astype("datetime64[M]")).astype(
        int
    ) + 1


def datetime64_hour(ndt):
    return ndt.astype("datetime64[h]").astype(int) % 24


def datetime64_minute(ndt):
    return (ndt.astype("datetime64[m]") - ndt.astype("datetime64[h]")).astype(
        int
    )


def datetime64_second(ndt):
    return (ndt.astype("datetime64[s]") - ndt.astype("datetime64[m]")).astype(
        int
    )


def datetime64_to_date(ndt):
    return dt.date(
        datetime64_year(ndt), datetime64_month(ndt), datetime64_day(ndt)
    )


def datetime64_to_datetime(ndt, tzone=dt.timezone.utc):
    return dt.datetime(
        datetime64_year(ndt),
        datetime64_month(ndt),
        datetime64_day(ndt),
        datetime64_hour(ndt),
        datetime64_minute(ndt),
        datetime64_second(ndt),
        tzinfo=tzone,
    )


def validate_file_path(path):
    if os.path.isfile(path):
        return path
    raise IOError(f"Could not find file: '{path}'")


def validate_file_path_list(paths):
    for p in paths:
        validate_file_path(p)
    return paths


def validate_dir_path(path):
    if os.path.isdir(path):
        return path
    raise IOError(f"Could not find dir: '{path}'")


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def flatten_to_iterable(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def write_accuracies_file(dates, era_acc, aws_acc, path):
    with open(path, "w") as fd:
        for d, ae, aa in zip(dates, era_acc, aws_acc):
            fd.write(f"{d},{ae},{aa}\n")


def read_accuracies_file(path):
    with open(path) as fd:
        dates = []
        era = []
        aws = []
        for line in fd:
            values = line.split(",")
            dates.append(dt.date.fromisoformat(values[0]))
            era.append(float(values[1]))
            aws.append(float(values[2]))
        era = np.array(era)
        aws = np.array(aws)
        return dates, era, aws
