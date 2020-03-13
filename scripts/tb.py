from collections import namedtuple
import numpy as np
import os
import re

import ease_grid as eg


def load_flat_binary_file(fname, nrows, ncols, dtype):
    return np.fromfile(fname, dtype=dtype).reshape((nrows, ncols))


def load_tb_file(fname, proj):
    nr, nc = eg.GRID_NAME_TO_V1_SHAPE[proj]
    # Data is stored as binary integers. Divide by 10 to recover float values
    return load_flat_binary_file(fname, nr, nc, "<h") / 10.0


# Regex for parsing EASE grid file names of the form
# 'EASE-III-MMYYYYDDDP.FFG' or 'EASE-III-MMYYYYDDDP-V2.FFG'.
#  III: satellite ID (or just SSMI for filled data)
#   MM: EASE projection grid (ML, MH, etc.)
# YYYY: year
#  DDD: day of year
#    P: A or D for ascending or descending pass
#   FF: Frequency, 19/22/37/91 GHz
#    G: polarization, V or H for vertical/horizontal
#
# Groups: dataset_or_sat_id, proj, year, day_of_year, pass_type, freq, pol
EASE_FNAME_PAT = re.compile(
    "^EASE-(SSMI|[A-Z\\d]+)-(ML|MH|NL|NH|SL|SH)(\\d{4})(\\d{3})(A|D)"
    "(?:-V2)?\\.(\\d+)(V|H)$"
)


KEY_YEAR = "year"
KEY_SAT_NAME = "sat_name"
KEY_GRID_CODE = "grid_code"
KEY_SAT_PASS = "sat_pass"
KEY_FREQ = "freq"
KEY_POL = "pol"
KEY_FREQ_POL = "freq_pol"
KEY_PATH = "path"

KEY_19H = "19H"
KEY_19V = "19V"
KEY_22V = "22V"
KEY_37H = "37H"
KEY_37V = "37V"
KEY_85H = "85H"
KEY_85V = "85V"
KEY_91H = "91H"
KEY_91V = "91V"

FP_KEYS = [
    KEY_19H,
    KEY_19V,
    KEY_22V,
    KEY_37H,
    KEY_37V,
    KEY_85H,
    KEY_85V,
    KEY_91H,
    KEY_91V,
]

SAT_ASCENDING = "A"
SAT_DESCENDING = "D"

# Regex for parsing netCDF SMMR/SSMI/SSMIS Tb files.
# Fields:
#   year: retrieval year
#   sat_name: satellite name (eg F08, SMMR)
#   grid_code: EASE grid code
#   sat_pass: ascending or descending pass code (A or D)
#   freq: Tb frequency (eg 19, 37, etc GHz)
#   pol: retrieval polarization (H or V)
NC_TB_FNAME_PAT = re.compile(
    f"tb_(?P<{KEY_YEAR}>\\d{{4}})_(?P<{KEY_SAT_NAME}>[A-Z]+\\d+)"
    f"_(?P<{KEY_GRID_CODE}>ML|MH|NL|NH|SL|SH)"
    f"_(?P<{KEY_SAT_PASS}>A|D)_(?P<{KEY_FREQ}>\\d+)(?P<{KEY_POL}>H|V)\\.nc$"
)


TbFileInfo = namedtuple(
    "TbFileInfo",
    (
        KEY_YEAR,
        KEY_SAT_NAME,
        KEY_GRID_CODE,
        KEY_SAT_PASS,
        KEY_FREQ,
        KEY_POL,
        KEY_PATH,
    ),
)
TbFileInfoMerged = namedtuple(
    "TbFileInfoMerged",
    (
        KEY_YEAR,
        KEY_SAT_NAME,
        KEY_GRID_CODE,
        KEY_SAT_PASS,
        KEY_FREQ_POL,
        KEY_PATH,
    ),
)


def parse_nc_tb_fname(path, merge_freq_pol=True):
    name = os.path.basename(path)
    m = NC_TB_FNAME_PAT.match(name)
    if m is None:
        return m
    if merge_freq_pol:
        return TbFileInfoMerged(
            int(m.group(KEY_YEAR)),
            m.group(KEY_SAT_NAME),
            m.group(KEY_GRID_CODE),
            m.group(KEY_SAT_PASS),
            m.group(KEY_FREQ) + m.group(KEY_POL),
            path,
        )
    else:
        return TbFileInfo(
            int(m.group(KEY_YEAR)),
            m.group(KEY_SAT_NAME),
            m.group(KEY_GRID_CODE),
            m.group(KEY_SAT_PASS),
            m.group(KEY_FREQ),
            m.group(KEY_POL),
            path,
        )
