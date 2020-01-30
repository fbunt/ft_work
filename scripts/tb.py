import numpy as np
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
