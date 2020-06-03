from collections import namedtuple
from scipy.spatial import cKDTree as KDTree
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
from transforms import AK_VIEW_TRANS
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
    masked_pred = [p[land_mask & vmask] for p, vmask in zip(pred, val_mask_ds)]
    era = [
        v.argmax(0)[land_mask & vmask]
        for v, vmask in zip(era_val_ds, val_mask_ds)
    ]
    era_acc = np.array(
        [(p == e).sum() / p.size for p, e in zip(masked_pred, era)]
    )
    era_acc *= 100
    # Validate against AWS DB
    pf = WMOValidationPointFetcher(db, RETRIEVAL_MIN)
    elon, elat = [view_trans(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)]
    aws_val = WMOValidator(pf)
    mask_iter = (land_mask & vmask for vmask in val_mask_ds)
    aws_acc = aws_val.validate_bounded(
        pred,
        val_dates,
        elon,
        elat,
        mask_iter,
        show_progress=True,
        variable_mask=True,
    )
    aws_acc *= 100
    acc_file = os.path.join(root, "acc.csv")
    with open(acc_file, "w") as fd:
        for d, ae, aa in zip(val_dates, era_acc, aws_acc):
            fd.write(f"{d},{ae},{aa}\n")
    # Save prediction plots
    print(f"Creating prediction plots: '{pred_plots}'")
    pfmt = os.path.join(pred_plots, "{:03}.png")
    for i, p in enumerate(tqdm.tqdm(pred, ncols=80)):
        plt.figure()
        plt.imshow(p)
        plt.title(f"Day: {i + 1}")
        plt.savefig(pfmt.format(i + 1), dpi=200)
        plt.close()
    # ERA
    plt.figure()
    plt.plot(val_dates, era_acc, lw=1, label="ERA5")
    plt.title(f"ERA Accuracy: {era_acc.mean():.3}%")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root, "acc_era_plot.png"), dpi=300)
    # AWS
    plt.figure()
    plt.plot(val_dates, aws_acc, lw=1, label="AWS")
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


def aws_loss_func(batch_pred_logits, batch_idxs, batch_labels, config, device):
    loss = 0
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
    dates_path, masks_path, db_path, land_mask, transform, ret_type
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
    for d, mask in tqdm.tqdm(
        zip(train_dates, mask_ds),
        ncols=80,
        total=len(train_dates),
        desc="Loading AWS",
    ):
        vpoints, vtemps = aws_pf.fetch_bounded(d, geo_bounds)
        vft = ft_model_zero_threshold(vtemps).astype(int)
        mask = mask & land_mask
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
        for i in range(x.shape[1]):
            sub = x[:, i]
            x[:, i] = (sub - sub.mean()) / sub.std()
        return x


def build_day_of_year_ds(dates_path, shape):
    dates = load_dates(dates_path)
    doys = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    doys = normalize(doys)
    ds = SingleValueGridDataset(doys, shape)
    return ds


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
        "l2_reg_weight",
        "lv_reg_weight",
        "land_reg_weight",
        "use_aws",
        "aws_loss_weight",
        "optimizer",
    ),
)


config = Config(
    in_chan=10,
    n_classes=3,
    depth=4,
    base_filters=64,
    epochs=50,
    batch_size=18,
    batch_shuffle=False,
    drop_last=False,
    learning_rate=0.0005,
    lr_gamma=0.89,
    l2_reg_weight=0.01,
    lv_reg_weight=0.05,
    land_reg_weight=0.0001,
    use_aws=False,
    aws_loss_weight=1e-4,
    optimizer=torch.optim.Adam,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = AK_VIEW_TRANS
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))
land_mask = ~water_mask
land_mask_np = land_mask.numpy()
land_channel = torch.tensor(normalize(land_mask.numpy())).float()
dem_channel = torch.tensor(
    transform(normalize(np.load("../data/z/dem.npy")))
).float()
elon, elat = eg.v1_get_full_grid_lonlat(eg.ML)
lat_channel = torch.tensor(transform(normalize(elat))).float()
doy_ds = build_day_of_year_ds(
    "../data/train/date_map-2007-2010.csv", land_mask_np.shape
)

if config.use_aws:
    aws_data = get_aws_data(
        "../data/train/date_map-2007-2010.csv",
        "../data/train/tb_valid_mask-2007-2010-D-ak.npy",
        "../data/dbs/wmo_gsod.db",
        land_mask_np,
        transform,
        RETRIEVAL_MIN,
    )

solar_ds = NpyDataset(
    normalize(np.load("../data/train/solar_rad-2007-2010-AM-ak.npy"))
)
tb_ds = NpyDataset(normalize(np.load("../data/train/tb-2007-2010-D-ak.npy")))
# Tack on land mask as first channel
tb_ds = GridsStackDataset(
    [
        RepeatDataset(land_channel, len(tb_ds)),
        RepeatDataset(dem_channel, len(tb_ds)),
        RepeatDataset(lat_channel, len(tb_ds)),
        doy_ds,
        solar_ds,
        tb_ds,
    ]
)
era_ds = NpyDataset("../data/train/era5-t2m-am-2007-2010-ak.npy")
idx_ds = IndexEchoDataset(len(tb_ds))
ds = ComposedDataset([idx_ds, tb_ds, era_ds])
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

        loss = criterion(log_class_prob, label)
        writer.add_scalar("CE Loss", loss.item(), step)
        # Minimize high frequency variation
        lv_loss = config.lv_reg_weight * local_variation_loss(class_prob)
        writer.add_scalar("LV Loss", lv_loss.item(), step)
        loss += lv_loss
        # Minimize the probabilities of FT classes in water regions
        land_loss = class_prob[:, LABEL_FROZEN, water_mask].sum()
        land_loss += class_prob[:, LABEL_THAWED, water_mask].sum()
        # Minimize the probability of OTHER class in land regions
        land_loss += class_prob[:, LABEL_OTHER, land_mask].sum()
        land_loss *= config.land_reg_weight
        writer.add_scalar("Land Loss", land_loss.item(), step)
        loss += land_loss
        # AWS loss
        if config.use_aws:
            batch_aws_data = [aws_data[j] for j in ds_idxs]
            batch_aws_flat_idxs = [v[0] for v in batch_aws_data]
            batch_aws_labels = [v[1] for v in batch_aws_data]
            aws_loss = config.aws_loss_weight * aws_loss_func(
                log_class_prob,
                batch_aws_flat_idxs,
                batch_aws_labels,
                config,
                device,
            )
            writer.add_scalar("AWS Loss", aws_loss.item(), step)
            loss += aws_loss

        loss.backward()
        opt.step()

        writer.add_scalar("training_loss", loss.item(), step)
        iters += 1
    sched.step()
writer.close()

val_doy_ds = build_day_of_year_ds(
    "../data/val/date_map-2015.csv", land_mask_np.shape
)
solar_val_ds = NpyDataset(
    normalize(np.load("../data/val/solar_rad-2015-AM-ak.npy"))
)
input_val_ds = NpyDataset(normalize(np.load("../data/val/tb-2015-D-ak.npy")))
input_val_ds = GridsStackDataset(
    [
        RepeatDataset(land_channel, len(input_val_ds)),
        RepeatDataset(dem_channel, len(input_val_ds)),
        RepeatDataset(lat_channel, len(input_val_ds)),
        val_doy_ds,
        solar_val_ds,
        input_val_ds,
    ]
)
era_val_ds = NpyDataset("../data/val/era5-t2m-am-2015-ak.npy")
val_mask_ds = NpyDataset("../data/val/tb_valid_mask-2015-D-ak.npy")
val_dates = load_dates("../data/val/date_map-2015.csv")
write_results(
    run_dir,
    model,
    input_val_ds,
    era_val_ds,
    val_mask_ds,
    val_dates,
    config,
    device,
    land_mask,
    get_db_session("../data/dbs/wmo_gsod.db"),
    transform,
)
