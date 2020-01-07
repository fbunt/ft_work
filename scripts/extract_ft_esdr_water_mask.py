import argparse
import numpy as np
import os
import rasterio as rio
import sys

from utils import validate_file_path


WATER = 254


def extract_water_mask(ft_grid):
    mask = np.zeros_like(ft_grid, dtype='int8')
    mask[ft_grid == WATER] = 1
    return mask


def write_mask(mask, outpath):
    np.save(outpath, mask)


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=validate_file_path, help="Input FT_ESDR file")
    p.add_argument("output", type=str, help="Output file path")
    p.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="Overwrite output file location, if it exists",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    out_exists = os.path.isfile(args.output)
    if out_exists and not args.overwrite:
        print("Output location already exists. Use -O to overwrite")
        sys.exit(0)

    with rio.open(args.input) as fd:
        mask = extract_water_mask(fd.read(1))
    write_mask(mask, args.output)
