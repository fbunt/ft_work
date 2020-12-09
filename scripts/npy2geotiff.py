import argparse
import numpy as np
import rasterio as rio
from rasterio.transform import Affine

import ease_grid as eg
import utils
from transforms import REGION_TO_TRANS


def validate_region(reg):
    if reg in REGION_TO_TRANS:
        return reg
    raise KeyError("Region '{reg}' is not a valid region code")


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-m",
        "--mask",
        type=utils.validate_file_path,
        action="store",
        default=None,
        help="Mask to use with missing_value",
    )
    p.add_argument(
        "-v",
        "--missing_value",
        type=int,
        action="store",
        default=-1,
        help="Missing value",
    )
    p.add_argument(
        "-t",
        "--type",
        type=str,
        action="store",
        default="float64",
        help="Data type for final tiff",
    )
    p.add_argument(
        "region",
        type=validate_region,
        help="Region code for the data extent",
    )
    p.add_argument("in_file", type=utils.validate_file_path, help="Input file")
    p.add_argument("out_file", type=str, help="Output location")
    return p


class UnsupportedNumberDimsError(Exception):
    pass


def main(args):
    data = np.load(args.in_file)
    if len(data.shape) > 3 or len(data.shape) < 2:
        raise UnsupportedNumberDimsError(
            f"Number of dims must be 2 or 3. Got {len(data.shape)}."
        )
    elif len(data.shape) == 3:
        nbands = data.shape[0]
    else:
        nbands = 1
        # Add extra axis for iteration purposes
        data = np.expand_dims(data, axis=0)
    trans = REGION_TO_TRANS[args.region]
    if args.mask is not None:
        mask = trans(np.load(args.mask).astype(bool))
        data[..., mask] = args.missing_value
    crs = eg.GRID_NAME_TO_V1_PROJ[eg.ML]
    x, y = [
        trans(xi)
        for xi in eg.v1_lonlat_to_meters(*eg.v1_get_full_grid_lonlat(eg.ML))
    ]
    x = x[0]
    y = y[:, 0]
    xres = (x[-1] - x[0]) / len(x)
    yres = (y[-1] - y[0]) / len(y)
    t = Affine.translation(x[0], y[0]) * Affine.scale(xres, yres)
    ds = rio.open(
        args.out_file,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=nbands,
        dtype=args.type,
        crs=crs.srs,
        transform=t,
        compress="lzw",
        nodata=args.missing_value,
    )
    for i, band in enumerate(data):
        ds.write(band, i + 1)
    ds.close()


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
