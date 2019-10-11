from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio as rio


def plot(grid):
    colors_labels = {
        "tab:blue": "Frozen",
        "tab:red": "Thawed",
        "tab:green": "Assumed Thawed",
        "tab:purple": "Water",
    }
    cmap = colors.ListedColormap(list(colors_labels.keys()))
    norm = colors.BoundaryNorm([0, 1, 253, 254, 255], 4)
    fig, ax = plt.subplots()
    plt.imshow(grid, cmap=cmap, norm=norm)
    patches = [Patch(color=c, label=l) for c, l in colors_labels.items()]
    ax.legend(handles=patches, facecolor="white")
    plt.show()


if __name__ == "__main__":
    import sys

    path = sys.argv[-1]
    plot(rio.open(path).read(1))
