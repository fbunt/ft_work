from collections import namedtuple
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import os
import shutil
import stat
import torch
import torch.nn as nn
import tqdm

from dataloading import (
    ComposedDataset,
    GridsStackDataset,
    IndexEchoDataset,
    NpyDataset,
    RepeatDataset,
    SingleValueGridDataset,
)
from model import (
    LABEL_FROZEN,
    LABEL_OTHER,
    LABEL_THAWED,
    UNet,
    local_variation_loss,
)
from transforms import (
    AK_VIEW_TRANS,
    N45_VIEW_TRANS,
    N45W_VIEW_TRANS,
    NH_VIEW_TRANS,
)
from validate import (
    RETRIEVAL_MIN,
    WMOValidationPointFetcher,
    WMOValidator,
    ft_model_zero_threshold,
    get_nearest_flat_idxs_and_values,
)
from validation_db_orm import get_db_session
import ease_grid as eg


def load_dates(path):
    dates = []
    with open(path) as fd:
        for line in fd:
            i, ds = line.strip().split(",")
            dates.append(dt.date.fromisoformat(ds))
    return dates


def write_results(
    root,
    model,
    input_val_ds,
    era_val_ds,
    val_mask_ds,
    val_dates,
    config,
    device,
    land_mask,
    db,
    view_trans,
):
    os.makedirs(root, exist_ok=True)
    pred_plots = os.path.join(root, "pred_plots")
    if os.path.isdir(pred_plots):
        shutil.rmtree(pred_plots)
    os.makedirs(pred_plots)

    land_mask = land_mask.numpy()
    mpath = os.path.join(root, "model.pt")
    print(f"Saving model: '{mpath}'")
    torch.save(model.state_dict(), mpath)
    print("Generating predictions")
    pred = []
    for i, v in enumerate(tqdm.tqdm(input_val_ds, ncols=80)):
        pred.append(
            torch.softmax(
                model(v.unsqueeze(0).to(device, dtype=torch.float)).detach(), 1
            )
            .cpu()
            .squeeze()
            .numpy()
            .argmax(0)
        )
    pred = np.array(pred)
    ppath = os.path.join(root, "pred.npy")
    print(f"Saving predictions: '{ppath}'")
    np.save(ppath, pred)

    # Validate against ERA5
    print("Validating against ERA5")
    if config.val_use_valid_mask:
        masked_pred = [
            p[land_mask & vmask] for p, vmask in zip(pred, val_mask_ds)
        ]
        era = [
            v.argmax(0)[land_mask & vmask]
            for v, vmask in zip(era_val_ds, val_mask_ds)
        ]
    else:
        masked_pred = [p[land_mask] for p in pred]
        era = [v.argmax(0)[land_mask] for v in era_val_ds]
    era_acc = np.array(
        [(p == e).sum() / p.size for p, e in zip(masked_pred, era)]
    )
    era_acc *= 100
    # Validate against AWS DB
    pf = WMOValidationPointFetcher(db, RETRIEVAL_MIN)
    elon, elat = [view_trans(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)]
    aws_val = WMOValidator(pf)
    if config.val_use_valid_mask:
        mask = (land_mask & vmask for vmask in val_mask_ds)
    else:
        mask = land_mask
    aws_acc = aws_val.validate_bounded(
        pred,
        val_dates,
        elon,
        elat,
        mask,
        show_progress=True,
        variable_mask=config.val_use_valid_mask,
    )
    aws_acc *= 100
    acc_file = os.path.join(root, "acc.csv")
    with open(acc_file, "w") as fd:
        for d, ae, aa in zip(val_dates, era_acc, aws_acc):
            fd.write(f"{d},{ae},{aa}\n")
    # ERA
    plt.figure()
    plt.plot(val_dates, era_acc, lw=1, label="ERA5")
    plt.ylim(0, 100)
    plt.title(f"ERA Accuracy: {era_acc.mean():.3}%")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root, "acc_era_plot.png"), dpi=300)
    # AWS
    plt.figure()
    plt.plot(val_dates, aws_acc, lw=1, label="AWS")
    plt.ylim(0, 100)
    plt.title(f"AWS Accuracy: {aws_acc.mean():.3}%")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root, "acc_aws_plot.png"), dpi=300)
    # Both
    plt.figure()
    plt.plot(val_dates, era_acc, lw=1, label="ERA5")
    plt.plot(val_dates, aws_acc, lw=1, label="AWS")
    plt.ylim(0, 100)
    plt.legend(loc=0)
    plt.title(
        f"Mean Accuracy: ERA: {era_acc.mean():.3}% AWS: {aws_acc.mean():.3}"
    )
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root, "acc_plot.png"), dpi=300)
    plt.close()
    # Save prediction plots
    print(f"Creating prediction plots: '{pred_plots}'")
    pfmt = os.path.join(pred_plots, "{:03}.png")
    cmap = ListedColormap(
        [
            # Light blue: frozen
            (0.5294117647058824, 0.807843137254902, 0.9803921568627451, 0.5),
            # Red: thawed
            "olive",
            # Blue: other/water
            (0, 0, 1),
        ]
    )
    for i, p in enumerate(tqdm.tqdm(pred, ncols=80)):
        plt.figure()
        plt.imshow(p, cmap=cmap, vmin=LABEL_FROZEN, vmax=LABEL_OTHER)
        plt.title(f"Day: {i + 1}")
        plt.tight_layout(pad=2)
        plt.savefig(pfmt.format(i + 1), dpi=400)
        plt.close()


def aws_loss_func(batch_pred_logits, batch_idxs, batch_labels, config, device):
    loss = 0.0
    for pred, flat_idxs, labels in zip(
        batch_pred_logits, batch_idxs, batch_labels
    ):
        if not sum(flat_idxs.size()):
            continue
        # Add batch dim to left and flatten the (H, W) dims
        pred = pred.view(1, config.n_classes, -1)
        # Index in with indices corresponding to AWS stations
        pred = pred[..., flat_idxs]
        labels = labels.unsqueeze(0).to(device)
        loss += torch.nn.functional.cross_entropy(pred, labels)
    return loss


def get_aws_data(
    dates_path, masks_path, db_path, land_mask, transform, ret_type, config
):
    train_dates = load_dates(dates_path)
    mask_ds = NpyDataset(masks_path)
    db = get_db_session(db_path)
    aws_pf = WMOValidationPointFetcher(db, retrieval_type=ret_type)
    lon, lat = [transform(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)]
    tree = KDTree(np.array(list(zip(lon.ravel(), lat.ravel()))))
    geo_bounds = [
        lon.min(),
        lon.max(),
        lat.min(),
        lat.max(),
    ]
    valid_flat_idxs = []
    aws_labels = []
    use_valid_mask = config.aws_use_valid_mask
    for d, mask in tqdm.tqdm(
        zip(train_dates, mask_ds),
        ncols=80,
        total=len(train_dates),
        desc="Loading AWS",
    ):
        vpoints, vtemps = aws_pf.fetch_bounded(d, geo_bounds)
        vft = ft_model_zero_threshold(vtemps).astype(int)
        if use_valid_mask:
            mask = mask & land_mask
        else:
            mask = land_mask
        # The set of valid indices
        valid_idxs = set(np.nonzero(mask.ravel())[0])
        idxs, vft = get_nearest_flat_idxs_and_values(
            tree, vpoints, vft, valid_idxs
        )
        valid_flat_idxs.append(torch.tensor(idxs).long())
        aws_labels.append(torch.tensor(vft))
    db.close()
    return list(zip(valid_flat_idxs, aws_labels))


def normalize(x):
    if len(x.shape) < 4:
        return (x - x.mean()) / x.std()
    else:
        x = x.copy()
        for i in range(x.shape[1]):
            sub = x[:, i]
            x[:, i] = (sub - sub.mean()) / sub.std()
        return x


def build_day_of_year_ds(dates_path, shape, config):
    dates = load_dates(dates_path)
    doys = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    if config.normalize:
        doys = normalize(doys)
    ds = SingleValueGridDataset(doys, shape)
    return ds


def build_input_dataset(
    config,
    tb_path,
    dem,
    land_mask,
    latitude_grid,
    date_map_path,
    date_ds_shape,
    solar_path,
):
    datasets = []
    tb_ds = np.load(tb_path)
    if config.normalize:
        tb_ds = normalize(tb_ds)
    tb_ds = NpyDataset(tb_ds)
    reduced_indices = list(range(1, len(tb_ds)))
    # Land channel
    ds = RepeatDataset(land_mask, len(tb_ds))
    if config.use_prior_day:
        ds = Subset(ds, reduced_indices)
    datasets.append(ds)
    # DEM channel
    if config.use_dem:
        dem_channel = torch.tensor(dem).float()
        if config.normalize:
            dem_channel = normalize(dem_channel)
        ds = RepeatDataset(dem_channel, len(tb_ds))
        if config.use_prior_day:
            ds = Subset(ds, reduced_indices)
        datasets.append(ds)
    # Latitude channel
    if config.use_latitude:
        if config.normalize:
            latitude_grid = normalize(latitude_grid)
        ds = RepeatDataset(latitude_grid, len(tb_ds))
        if config.use_prior_day:
            ds = Subset(ds, reduced_indices)
        datasets.append(ds)
    # Day of year channel
    if config.use_day_of_year:
        ds = build_day_of_year_ds(date_map_path, date_ds_shape, config)
        if config.use_prior_day:
            ds = Subset(ds, reduced_indices)
        datasets.append(ds)
    # Solar radiation channel
    if config.use_solar:
        ds = np.load(solar_path)
        if config.normalize:
            ds = normalize(ds)
        ds = NpyDataset(ds)
        if config.use_prior_day:
            ds = Subset(ds, reduced_indices)
        datasets.append(ds)
    # Prior day Tb channels
    if config.use_prior_day:
        ds = Subset(tb_ds, list(range(0, len(tb_ds) - 1)))
        datasets.append(ds)
    # Tb channels
    if config.use_prior_day:
        tb_ds = Subset(tb_ds, reduced_indices)
    datasets.append(tb_ds)
    return GridsStackDataset(datasets)


def combine_loss(era_loss, aws_loss, land_loss, lv_loss, config):
    loss = 0
    if not config.use_relative_weights:
        loss += era_loss * config.era_weight
        loss += aws_loss * config.aws_loss_weight
        loss += land_loss * config.land_reg_weight
        loss += lv_loss * config.lv_reg_weight
    else:
        losses = [era_loss, aws_loss, land_loss]
        loss_items = [l.item() for l in losses]
        fractions = [
            config.era_rel_weight,
            config.aws_rel_weight,
            config.land_rel_weight,
        ]
        total = sum(loss_items)
        # Calculate the weight that forces the loss to contribute the specified
        # fraction to the total loss
        for li, f, lo in zip(loss_items, fractions, losses):
            weight = f * total / li
            loss += weight * lo
        # Tack on smaller loss values afterword
        loss += lv_loss * config.lv_reg_weight
    return loss


def _validate_relative_weights(*weights):
    assert sum(weights) == 1.0, "Relative weights must sum to 1.0"


Config = namedtuple(
    "Config",
    (
        "in_chan",
        "n_classes",
        "depth",
        "base_filters",
        "epochs",
        "batch_size",
        "batch_shuffle",
        "drop_last",
        "learning_rate",
        "lr_gamma",
        "aws_use_valid_mask",
        "val_use_valid_mask",
        "optimizer",
        "normalize",
        "use_dem",
        "use_latitude",
        "use_day_of_year",
        "use_solar",
        "use_prior_day",
        "region",
        "l2_reg_weight",
        "era_weight",
        "aws_loss_weight",
        "land_reg_weight",
        "lv_reg_weight",
        "use_relative_weights",
        "era_rel_weight",
        "aws_rel_weight",
        "land_rel_weight",
    ),
)


# Region codes
AK = "ak"
N45 = "n45"
N45W = "n45w"
NH = "nh"

region_to_trans = {
    AK: AK_VIEW_TRANS,
    N45: N45_VIEW_TRANS,
    N45W: N45W_VIEW_TRANS,
    NH: NH_VIEW_TRANS,
}

config = Config(
    # Base channels:
    #  * land mask: 1
    #  * tb: 5
    in_chan=6,
    n_classes=3,
    depth=4,
    base_filters=64,
    epochs=50,
    batch_size=10,
    batch_shuffle=False,
    drop_last=False,
    learning_rate=5e-4,
    lr_gamma=0.89,
    aws_use_valid_mask=False,
    val_use_valid_mask=False,
    optimizer=torch.optim.Adam,
    normalize=False,
    # 1 channel
    use_dem=False,
    # 1 channel
    use_latitude=False,
    # 1 channel
    use_day_of_year=False,
    # 1 channel
    use_solar=False,
    # 5 channels
    use_prior_day=False,
    region=AK,
    l2_reg_weight=1e-2,
    era_weight=1e0,
    aws_loss_weight=5e-2,
    land_reg_weight=1e-4,
    lv_reg_weight=5e-2,
    use_relative_weights=False,
    era_rel_weight=0.70,
    aws_rel_weight=0.05,
    land_rel_weight=0.25,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.use_relative_weights:
    _validate_relative_weights(
        config.era_rel_weight, config.aws_rel_weight, config.land_rel_weight
    )

transform = region_to_trans[config.region]
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))
land_mask = ~water_mask
land_mask_np = land_mask.numpy()
land_channel = torch.tensor(land_mask_np).float()
dem_channel = torch.tensor(transform(np.load("../data/z/dem.npy"))).float()
elon, elat = eg.v1_get_full_grid_lonlat(eg.ML)
lat_channel = torch.tensor(transform(elat)).float()
data_grid_shape = land_mask_np.shape

# AWS
aws_data = get_aws_data(
    f"../data/cleaned/date_map-2007-2010-{config.region}.csv",
    f"../data/cleaned/tb_valid_mask-D-2007-2010-{config.region}.npy",
    "../data/dbs/wmo_gsod.db",
    land_mask_np,
    transform,
    RETRIEVAL_MIN,
    config,
)
# Input dataset creation
input_ds = build_input_dataset(
    config,
    f"../data/cleaned/tb-D-2007-2010-{config.region}.npy",
    transform(np.load("../data/z/dem.npy")),
    land_channel,
    lat_channel,
    f"../data/cleaned/date_map-2007-2010-{config.region}.csv",
    data_grid_shape,
    f"../data/cleaned/solar_rad-AM-2007-2010-{config.region}.npy",
)
# Validation dataset
era_ds = NpyDataset(
    f"../data/cleaned/era5-t2m-am-2007-2010-{config.region}.npy"
)
if config.use_prior_day:
    era_ds = Subset(era_ds, list(range(1, len(input_ds) + 1)))
idx_ds = IndexEchoDataset(len(input_ds))
ds = ComposedDataset([idx_ds, input_ds, era_ds])
dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=config.batch_size,
    shuffle=config.batch_shuffle,
    drop_last=config.drop_last,
)

model = UNet(
    config.in_chan,
    config.n_classes,
    depth=config.depth,
    base_filter_bank_size=config.base_filters,
)
model.to(device)
opt = config.optimizer(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.l2_reg_weight,
)
sched = torch.optim.lr_scheduler.StepLR(opt, 1, config.lr_gamma)

# Create run dir and fill with info
stamp = str(dt.datetime.now()).replace(" ", "-")
run_dir = f"../runs/{stamp}"
os.makedirs(run_dir, exist_ok=True)
# Dump configuration info
with open(os.path.join(run_dir, "config"), "w") as fd:
    fd.write(f"{config}\n")
log_dir = os.path.join(run_dir, "logs")
writer = SummaryWriter(log_dir)
show_log_sh = os.path.join(run_dir, "show_log.sh")
# Create script to view logs
with open(show_log_sh, "w") as fd:
    fd.write("#!/usr/bin/env bash\n")
    fd.write(f"tensorboard --logdir {os.path.abspath(log_dir)}\n")
    fd.flush()
st = os.stat(show_log_sh)
os.chmod(show_log_sh, st.st_mode | stat.S_IXUSR)

criterion = nn.CrossEntropyLoss()
iters = 0
for epoch in range(config.epochs):
    writer.add_scalar(
        "learning_rate", next(iter(opt.param_groups))["lr"], epoch
    )
    it = tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Epoch: {epoch + 1}/{config.epochs}",
    )
    for i, (ds_idxs, input_data, label) in it:
        step = (epoch * len(dataloader)) + i
        input_data = input_data.to(device, dtype=torch.float)
        # Compress 1-hot encoding to single channel
        label = label.argmax(dim=1).to(device)
        # valid_mask = valid_mask.to(device)

        model.train()
        model.zero_grad()
        log_class_prob = model(input_data)
        class_prob = torch.softmax(log_class_prob, 1)

        #
        # ERA
        #
        era_loss = criterion(log_class_prob, label)
        writer.add_scalar("CE Loss", era_loss.item(), step)
        #
        # AWS loss
        #
        batch_aws_data = [aws_data[j] for j in ds_idxs]
        batch_aws_flat_idxs = [v[0] for v in batch_aws_data]
        batch_aws_labels = [v[1] for v in batch_aws_data]
        aws_loss = aws_loss_func(
            log_class_prob,
            batch_aws_flat_idxs,
            batch_aws_labels,
            config,
            device,
        )
        writer.add_scalar("AWS Loss", aws_loss.item(), step)
        #
        # Land/Water
        #
        # Minimize the probabilities of FT classes in water regions
        land_loss = class_prob[:, LABEL_FROZEN, water_mask].sum()
        land_loss += class_prob[:, LABEL_THAWED, water_mask].sum()
        # Minimize the probability of OTHER class in land regions
        land_loss += class_prob[:, LABEL_OTHER, land_mask].sum()
        writer.add_scalar("Land Loss", land_loss.item(), step)
        #
        # Local variation
        #
        # Minimize high frequency variation
        lv_loss = local_variation_loss(class_prob)
        writer.add_scalar("LV Loss", lv_loss.item(), step)
        loss = combine_loss(era_loss, aws_loss, land_loss, lv_loss, config)

        loss.backward()
        opt.step()

        writer.add_scalar("training_loss", loss.item(), step)
        iters += 1
    sched.step()
writer.close()
# Free up data for GC
input_ds = None
era_ds = None
idx_ds = None
ds = None
dataloader = None

# Validation
input_ds = build_input_dataset(
    config,
    f"../data/cleaned/tb-D-2015-{config.region}.npy",
    transform(np.load("../data/z/dem.npy")),
    land_channel,
    lat_channel,
    f"../data/cleaned/date_map-2015-{config.region}.csv",
    data_grid_shape,
    f"../data/cleaned/solar_rad-AM-2015-{config.region}.npy",
)
reduced_indices = list(range(1, len(input_ds) + 1))
era_ds = NpyDataset(f"../data/cleaned/era5-t2m-am-2015-{config.region}.npy")
val_mask_ds = NpyDataset(
    f"../data/cleaned/tb_valid_mask-D-2015-{config.region}.npy"
)
val_dates = load_dates(f"../data/cleaned/date_map-2015-{config.region}.csv")
if config.use_prior_day:
    era_ds = Subset(era_ds, reduced_indices)
    val_mask_ds = Subset(val_mask_ds, reduced_indices)
    val_dates = Subset(val_dates, reduced_indices)
write_results(
    run_dir,
    model,
    input_ds,
    era_ds,
    val_mask_ds,
    val_dates,
    config,
    device,
    land_mask,
    get_db_session("../data/dbs/wmo_gsod.db"),
    transform,
)
