import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import namedtuple

from utils import validate_dir_path


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=validate_dir_path, help="Data directory")
    return p


Dataset = namedtuple(
    "Dataset", ("name", "path", "freqs", "dates", "era_acc", "aws_acc")
)


NAMES = [
    "19",
    "22",
    "37",
    "19-22",
    "19-37",
    "22-37",
    "19-22-37",
]
NAME_2_FREQS = {
    "19": [19],
    "19-22": [19, 22],
    "19-22-37": [19, 22, 37],
    "19-37": [19, 37],
    "22": [22],
    "22-37": [22, 37],
    "37": [37],
}


def process_data_dir(d):
    name = os.path.basename(d)
    data = pd.read_csv(
        os.path.join(d, "acc.csv"),
        names=["dates", "era", "aws"],
        parse_dates=True,
        index_col=0,
    )
    data_np = data.to_numpy()
    dates = data.index.to_pydatetime()
    era = data_np[:, 0]
    aws = data_np[:, 1]
    return Dataset(name, d, NAME_2_FREQS[name], dates, era, aws)


def autolabel(ax, bars):
    """Attach a text label above each bar in *bars*, displaying its height.

    Adapted from matplotlib example.
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            "{:.1f}".format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            # 3 points vertical offset
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def bar_chart(datasets):
    labels = NAMES
    era = [datasets[n].era_acc.mean() for n in labels]
    aws = [datasets[n].aws_acc.mean() for n in labels]
    width = 0.35
    margin = 0.5
    label_locs = np.arange(len(labels))
    fig, ax = plt.subplots()
    era_bars = ax.bar(label_locs - (width / 2), era, width, label="ERA")
    aws_bars = ax.bar(label_locs + (width / 2), aws, width, label="AWS")
    autolabel(ax, era_bars)
    autolabel(ax, aws_bars)
    ymin = min(min(era), min(aws)) - margin
    ymax = max(max(era), max(aws)) + margin
    plt.ylim(ymin, ymax)
    ax.set_title("Band Accuracy Comparison")
    ax.set_ylabel("Mean % Accuracy")
    ax.set_xlabel("Band Frequencies (GHz)")
    ax.set_xticks(label_locs)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def main(args):
    dirs = [
        d
        for d in glob.glob(os.path.join(args.dir, "*"))
        if os.path.basename(d) in NAME_2_FREQS
    ]
    datasets = {os.path.basename(d): process_data_dir(d) for d in dirs}
    bar_chart(datasets)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
