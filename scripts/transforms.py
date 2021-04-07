import numpy as np

from datahandling import ViewCopyTransform


_EASE_LAT = np.load("../data/coords/lat_eg_ml.npy")


def gl_view(x):
    return x


def get_transform_from_min_lat(lat_value):
    mask = _EASE_LAT >= lat_value
    r, c = _EASE_LAT[mask].reshape(-1, _EASE_LAT.shape[1]).shape
    trans = ViewCopyTransform(0, r - 1, 0, c - 1)
    return trans


# Returns as is
GL_VIEW_TRANS = gl_view
# Copies a small window around Alaska
AK_VIEW_TRANS = ViewCopyTransform(15, 62, 12, 191)
# Copies the northern hemisphere
NH_VIEW_TRANS = get_transform_from_min_lat(0)
# Copies everything North of 20 N
N20_VIEW_TRANS = get_transform_from_min_lat(20)
# Copies everything at and above 45 N
N45_VIEW_TRANS = get_transform_from_min_lat(45)
# Copies everything at and above N 45 and everything west of 0
N45W_VIEW_TRANS = ViewCopyTransform(0, 84, 0, 691)


# Region codes
AK = "ak"
GL = "gl"
N20 = "n20"
N45 = "n45"
N45W = "n45w"
NH = "nh"
REGION_CODES = frozenset((AK, GL, N20, N45, N45W, NH))
REGION_TO_TRANS = {
    AK: AK_VIEW_TRANS,
    GL: GL_VIEW_TRANS,
    N20: N20_VIEW_TRANS,
    N45: N45_VIEW_TRANS,
    N45W: N45W_VIEW_TRANS,
    NH: NH_VIEW_TRANS,
}
