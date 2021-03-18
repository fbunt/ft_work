from datahandling import ViewCopyTransform


def gl_view(x):
    return x


# Returns as is
GL_VIEW_TRANS = gl_view
# Copies a small window around Alaska
AK_VIEW_TRANS = ViewCopyTransform(15, 62, 12, 191)
# Copies the northern hemisphere
NH_VIEW_TRANS = ViewCopyTransform(0, 292, 0, 1382)
# Copies everything at and above N 45
N45_VIEW_TRANS = ViewCopyTransform(0, 84, 0, 1382)
# Copies everything at and above N 45 and everything west of 0
N45W_VIEW_TRANS = ViewCopyTransform(0, 84, 0, 691)


# Region codes
AK = "ak"
GL = "gl"
N45 = "n45"
N45W = "n45w"
NH = "nh"
REGION_CODES = frozenset((AK, N45, N45W, NH))
REGION_TO_TRANS = {
    AK: AK_VIEW_TRANS,
    GL: GL_VIEW_TRANS,
    N45: N45_VIEW_TRANS,
    N45W: N45W_VIEW_TRANS,
    NH: NH_VIEW_TRANS,
}
