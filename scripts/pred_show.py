from matplotlib.animation import FuncAnimation
import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import FT_CMAP, validate_file_path


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--interval",
        type=int,
        default=200,
        help="The number of ms between frames, default=200",
    )
    p.add_argument(
        "npy_file", type=validate_file_path, help="The file to pull data from"
    )
    return p


def update_plot(i, *args):
    data, plot, ax = args
    grid = data[i]
    plot.set_data(grid)
    ax.set_title(f"Day: {i + 1}")
    return plot


def show_data(file, interval):
    data = np.load(file)
    fig, ax = plt.subplots()

    grid = np.zeros_like(data[0]) + np.nan
    img = ax.imshow(
        grid, vmin=data.min(), vmax=data.max(), animated=True, cmap=FT_CMAP
    )
    ax.set_title("0")

    ani = FuncAnimation(
        fig,
        update_plot,
        range(len(data)),
        fargs=(data, img, ax),
        interval=interval,
    )
    plt.show()


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    show_data(args.npy_file, args.interval)
