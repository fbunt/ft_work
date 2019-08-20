import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import tqdm

import tb as tbmod


class TbDataWrapper:
    def __init__(self, data_pattern):
        self._pat = data_pattern
        self._files = sorted(glob.glob(data_pattern))
        print(self._files)
        if not len(self._files):
            raise IOError(f"No files found with pattern: '{data_pattern}'")
        r, s = np.meshgrid(range(tbmod.EASE_COLS), range(tbmod.EASE_ROWS))
        self.lon, self.lat = tbmod.ease_inverse(r, s)
        self._dates = []
        self._grids = None
        self._load()
        self.min = self._grids.min()
        self.max = self._grids.max()

    def _load(self):
        dates = []
        grids = np.zeros((len(self), tbmod.EASE_ROWS, tbmod.EASE_COLS))
        print("Loading...")
        for i, f in enumerate(tqdm.tqdm(self._files, ncols=80)):
            base = os.path.basename(f)
            m = tbmod.EASE_FNAME_PAT.match(base)
            sat_id, projection, year, doy, ad_pass, freq, pol = m.groups()
            dt_year = np.datetime64(year, dtype="datetime64[Y]")
            dt = dt_year + np.timedelta64(int(doy) - 1, "D")
            grid = tbmod.load_tb_file(f)
            dates.append(dt)
            grids[i] = grid
        self._dates = dates
        self._grids = grids

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of bounds: {idx}")
        dt = self._dates[idx]
        grid = self._grids[idx]
        return dt, grid


IMG_NAME_FMT = "tb_{}.png"


def create_tb_images(data_pattern, out_dir, nlevels=50):
    if not os.path.isdir(out_dir):
        raise IOError(f"Could not find output directory: {out_dir}")
    data = TbDataWrapper(data_pattern)
    fmt = os.path.join(out_dir, IMG_NAME_FMT)
    lon = data.lon
    lat = data.lat
    print("Plotting")
    for i in tqdm.tqdm(range(len(data)), ncols=80):
        dt, grid = data[i]
        grid[grid == 0] = np.nan
        fig = plt.figure()
        plt.contourf(
            lon, lat, grid, levels=nlevels, vmin=data.min, vmax=data.max
        )
        # plt.colorbar()
        plt.title(str(dt))
        fname = fmt.format(dt)
        plt.savefig(fname, dpi=200)
        plt.close(fig)


def create_video_from_imgs(img_dir, out_path, fps, overwrite=False):
    if os.path.isfile(out_path) and not overwrite:
        raise IOError("Output file already exists. Use overwrite option")
    if fps <= 0:
        raise ValueError("fps must be a positive value")
    img_glob = os.path.join(img_dir, "*.png")
    os.system(
        "ffmpeg -y"
        f" -framerate {fps}"
        f" -pattern_type glob -i '{img_glob}'"
        f" -c:v libx264 -crf 6 {out_path}"
    )


def _validate_dir(path):
    if os.path.isdir(path):
        return path
    raise IOError(f"Invalid directory path: '{path}'")


def _validate_pos_int(i):
    if i > 0:
        return i
    raise ValueError(f"Value must be greater than zero: {i}.")


_DEFAULT_PLOT_DIR = ".plot_tmp"


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("data_pattern", type=str, help="Data globbing pattern")
    p.add_argument("out_file", type=str, help="Output video file name")
    p.add_argument(
        "-n",
        "--nlevels",
        type=_validate_pos_int,
        default=50,
        help="Number of contours in plots",
    )
    p.add_argument(
        "-f",
        "--fps",
        type=_validate_pos_int,
        default=2,
        help="FPS value for video",
    )
    p.add_argument(
        "-p",
        "--plot_dir",
        type=str,
        default=_DEFAULT_PLOT_DIR,
        help="Directory to store plots in",
    )
    p.add_argument(
        "-O", "--overwrite", action="store_true", help="Overwrite video file"
    )
    return p


def _setup(args):
    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            "Output destination already exists. Use -O to overwrite."
        )
    if not os.path.isdir(args.plot_dir):
        os.makedirs(args.plot_dir)


def _tear_down(args):
    if args.plot_dir == _DEFAULT_PLOT_DIR and os.path.isdir(args.plot_dir):
        shutil.rmtree(args.plot_dir)


def main(args):
    try:
        _setup(args)
        create_tb_images(args.data_pattern, args.plot_dir, args.nlevels)
        create_video_from_imgs(
            args.plot_dir, args.out_file, args.fps, args.overwrite
        )
    except Exception as e:
        raise e
    finally:
        _tear_down(args)


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args)
