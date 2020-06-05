from dataloading import ViewCopyTransform


# Copies a small window around Alaska
AK_VIEW_TRANS = ViewCopyTransform(15, 62, 12, 191)
# Copies the northern hemisphere
NH_VIEW_TRANS = ViewCopyTransform(0, 292, 0, 1383)
