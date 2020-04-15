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
    ViewCopyTransform,
)
from model import (
    LABEL_FROZEN,
    LABEL_OTHER,
    LABEL_THAWED,
    UNet,
    local_variation_loss,
)


def write_results(root, model, val_dl, device):
    os.makedirs(root, exist_ok=True)
    pred_plots = os.path.join(root, "pred_plots")
    if os.path.isdir(pred_plots):
        shutil.rmtree(pred_plots)
    os.makedirs(pred_plots)

    mpath = os.path.join(root, "model.pt")
    print(f"Saving model: '{mpath}'")
    torch.save(model.state_dict(), mpath)
    print("Generating predictions")
    pred = []
    for i, v in enumerate(tqdm.tqdm(val_dl, ncols=80)):
        pred.append(
            torch.softmax(model(v.to(device, dtype=torch.float)).detach(), 1)
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
        plt.savefig(pfmt.format(i + 1))
        plt.close()
    # TODO: validate


in_chan = 6
nclasses = 3
depth = 4
base_filters = 32
epochs = 30
batch_size = 10
learning_rate = 0.0005
lr_gamma = 0.89
learning_momentum = 0.9
l2_reg_weight = 0.1
lv_reg_weight = 0.2
land_reg_weight = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = ViewCopyTransform(15, 62, 12, 191)
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))
land_mask = ~water_mask
land_channel = land_mask.float()

root_data_dir = "../data/train/"
tb_ds = NCTbDataset(os.path.join(root_data_dir, "tb"), transform=transform)
# Tack on land mask as first channel
tb_ds = GridsStackDataset([RepeatDataset(land_channel, len(tb_ds)), tb_ds])
era08_ds = NpyDataset(
    os.path.join(root_data_dir, "era5_t2m/era5-t2m-bidaily-2008-ak.npy"),
)
era09_ds = NpyDataset(
    os.path.join(root_data_dir, "era5_t2m/era5-t2m-bidaily-2009-ak.npy"),
)
era_ds = torch.utils.data.ConcatDataset([era08_ds, era09_ds])
ds = ComposedDataset([tb_ds, era_ds])
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=batch_size, shuffle=True, drop_last=True
)

model = UNet(
    in_chan, nclasses, depth=depth, base_filter_bank_size=base_filters
)
model.to(device)
opt = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    weight_decay=l2_reg_weight,
)
sched = torch.optim.lr_scheduler.StepLR(opt, 1, lr_gamma)

stamp = str(dt.datetime.now()).replace(" ", "-")
run_dir = f"../runs/{stamp}"
os.makedirs(run_dir, exist_ok=True)
log_dir = os.path.join(run_dir, "logs")
writer = SummaryWriter(log_dir)
show_log_sh = os.path.join(run_dir, "show_log.sh")
with open(show_log_sh, "w") as fd:
    fd.write("#!/usr/bin/env bash\n")
    fd.write(f"tensorboard --logdir {os.path.abspath(log_dir)}\n")
    fd.flush()
st = os.stat(show_log_sh)
os.chmod(show_log_sh, st.st_mode | stat.S_IXUSR)

criterion = nn.CrossEntropyLoss()
iters = 0
for epoch in range(epochs):
    writer.add_scalar(
        "learning_rate", next(iter(opt.param_groups))["lr"], epoch
    )
    it = tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Epoch: {epoch + 1}/{epochs}",
    )
    for i, (input_data, label) in it:
        input_data = input_data.to(device, dtype=torch.float)
        # Compress 1-hot encoding to single channel
        label = label.argmax(dim=1).to(device)

        model.zero_grad()
        log_class_prob = model(input_data)
        class_prob = torch.softmax(log_class_prob, 1)

        loss = criterion(log_class_prob, label)
        # Minimize high frequency variation
        loss += lv_reg_weight * local_variation_loss(class_prob)
        # Minimize the probabilities of FT classes in water regions
        land_loss = class_prob[:, LABEL_FROZEN, water_mask].sum()
        land_loss += class_prob[:, LABEL_THAWED, water_mask].sum()
        # Minimize the probability of OTHER class in land regions
        land_loss += class_prob[:, LABEL_OTHER, land_mask].sum()
        loss += land_reg_weight * land_loss
        loss.backward()
        opt.step()

        step = (epoch * len(dataloader)) + i
        writer.add_scalar("training_loss", loss.item(), step)
        it.write(f"{loss.item()}")
        iters += 1
    sched.step()
writer.close()

val_ds = NCTbDataset("../data/val/tb", transform=transform)
val_ds = GridsStackDataset([RepeatDataset(land_channel, len(val_ds)), val_ds])
val_dl = torch.utils.data.DataLoader(val_ds)
write_results(run_dir, model, val_dl, device)
