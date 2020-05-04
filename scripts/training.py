from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import stat
import torch
import torch.nn as nn
import tqdm

from dataloading import (
    ComposedDataset,
    NCTbDataset,
    NpyDataset,
    RepeatDataset,
    GridsStackDataset,
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
    validate_grid_against_truth_bulk,
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

    pfmt = os.path.join(pred_plots, "{:03}.png")
    print(f"Creating prediction plots: '{pred_plots}'")
    for i, p in enumerate(tqdm.tqdm(pred, ncols=80)):
        plt.figure()
        plt.imshow(p)
        plt.title(f"Day: {i + 1}")
        plt.savefig(pfmt.format(i + 1), dpi=200)
        plt.close()
    print("Validating against ERA5")
    era = np.stack([v.argmax(0)[land_mask] for v in era_val_ds], 0)
    era_acc = validate_grid_against_truth_bulk(pred[..., land_mask], era)
    era_acc *= 100
    pf = WMOValidationPointFetcher(db, RETRIEVAL_MIN)
    elon, elat = [view_trans(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)]
    aws_val = WMOValidator(pf)
    aws_acc = aws_val.validate_bounded(
        val_dates, elon, elat, pred, land_mask, True
    )
    aws_acc *= 100
    plt.figure()
    plt.plot(val_dates, era_acc, lw=1, label="ERA5")
    plt.plot(val_dates, aws_acc, lw=1, label="AWS")
    plt.legend(loc=0)
    plt.title(
        f"Mean Accuracy: ERA: {era_acc.mean():.3}% AWS: {aws_acc.mean():.3}"
    )
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(root, "acc_plot.png"), dpi=300)
    plt.close()


Config = namedtuple(
    "Config",
    (
        "in_chan",
        "n_classes",
        "depth",
        "base_filters",
        "epochs",
        "batch_size",
        "learning_rate",
        "lr_gamma",
        "l2_reg_weight",
        "lv_reg_weight",
        "land_reg_weight",
        "optimizer",
    ),
)


config = Config(
    in_chan=6,
    n_classes=3,
    depth=4,
    base_filters=64,
    epochs=50,
    batch_size=10,
    learning_rate=0.0005,
    lr_gamma=0.89,
    l2_reg_weight=0.01,
    lv_reg_weight=0.05,
    land_reg_weight=0.0001,
    optimizer=torch.optim.Adam,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = AK_VIEW_TRANS
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))
land_mask = ~water_mask
land_channel = land_mask.float()

root_data_dir = "../data/train/"
tb_ds = NpyDataset("../data/train/tb-2008-2009-D-ak.npy")
# Tack on land mask as first channel
tb_ds = GridsStackDataset([RepeatDataset(land_channel, len(tb_ds)), tb_ds])
era_ds = NpyDataset("../data/train/era5-t2m-am-2008-2009-ak.npy")
ds = ComposedDataset([tb_ds, era_ds])
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=config.batch_size, shuffle=True, drop_last=True
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
    for i, (input_data, label) in it:
        step = (epoch * len(dataloader)) + i
        input_data = input_data.to(device, dtype=torch.float)
        # Compress 1-hot encoding to single channel
        label = label.argmax(dim=1).to(device)

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
        loss.backward()
        opt.step()

        writer.add_scalar("training_loss", loss.item(), step)
        iters += 1
    sched.step()
writer.close()

input_val_ds = NpyDataset("../data/val/tb-2015-D-ak.npy")
input_val_ds = GridsStackDataset(
    [RepeatDataset(land_channel, len(input_val_ds)), input_val_ds]
)
era_val_ds = NpyDataset("../data/val/era5-t2m-am-2015-ak.npy")
val_dates = load_dates("../data/val/date_map-2015.csv")
write_results(
    run_dir,
    model,
    input_val_ds,
    era_val_ds,
    val_dates,
    config,
    device,
    land_mask,
    get_db_session("../data/dbs/wmo_gsod.db"),
    transform,
)
