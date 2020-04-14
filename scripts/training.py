from torch.utils.tensorboard import SummaryWriter
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
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
# opt = torch.optim.SGD(
opt = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    # momentum=learning_momentum,
    weight_decay=l2_reg_weight,
)
sched = torch.optim.lr_scheduler.StepLR(opt, 1, lr_gamma)

writer = SummaryWriter("../runs/ft_run")
loss_vec = []

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

        loss_vec.append(loss.item())
        step = (epoch * len(dataloader)) + i
        writer.add_scalar("training_loss", loss.item(), step)
        it.write(f"{loss.item()}")
        iters += 1
    sched.step()
writer.close()

fmt = "../models/unet-am-in_{}-nclass_{}-depth_{}-{}.pt"
torch.save(
    model.state_dict(),
    fmt.format(in_chan, nclasses, depth, dt.datetime.now().timestamp()),
)
val_ds = NCTbDataset("../data/val/tb", transform=transform)
val_ds = GridsStackDataset([RepeatDataset(land_channel, len(val_ds)), val_ds])
val_dl = torch.utils.data.DataLoader(val_ds)
pred = [
    torch.softmax(model(v.to(device, dtype=torch.float)).detach(), 1)
    .cpu()
    .squeeze()
    .numpy()
    .argmax(0)
    for v in val_dl
]
pred = np.array(pred)
np.save("../data/pred/pred.npy", pred)
plt.plot(loss_vec)
plt.show()
