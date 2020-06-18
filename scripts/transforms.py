from dataloading import ViewCopyTransform


# Copies a small window around Alaska
AK_VIEW_TRANS = ViewCopyTransform(15, 62, 12, 191)
# Copies the northern hemisphere
NH_VIEW_TRANS = ViewCopyTransform(0, 292, 0, 1383)
# Copies everything at and above N 45
N45_VIEW_TRANS = ViewCopyTransform(0, 84, 0, 1383)
# Copies everything at and above N 45 and everything west of 0
N45W_VIEW_TRANS = ViewCopyTransform(0, 84, 0, 691)
