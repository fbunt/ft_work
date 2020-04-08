import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import tqdm

from dataloading import (
    # AWSFuzzyLabelDataset,
    # DatabaseReference,
    ComposedDictDataset,
    KEY_ERA5_LABEL,
    KEY_TB_DATA,
    NCTbDataset,
    NCTbDatasetKeyedWrapper,
    NpyDataset,
    ViewCopyTransform,
)
from model import UNet, local_variation_loss
# from validation_db_orm import get_db_session


in_chan = 5
nclasses = 3
depth = 4
base_filters = 32
epochs = 30
batch_size = 10
learning_rate = 0.0005
learning_momentum = 0.9
l2_reg_weight = 0.1
lv_reg_weight = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = ViewCopyTransform(15, 62, 12, 191)
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))

root_data_dir = "../data/train/"
tb_ds = NCTbDatasetKeyedWrapper(
    NCTbDataset(root_data_dir, transform=transform)
)
era_ds = NpyDataset(
    KEY_ERA5_LABEL,
    os.path.join(root_data_dir, "era5_t2m/era5-t2m-bidaily-2009-ak.npy"),
)
ds = ComposedDictDataset([tb_ds, era_ds])
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=batch_size, shuffle=True, drop_last=True
)

# db = DatabaseReference("../data/dbs/wmo_gsod.db", get_db_session)
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

loss_vec = []

criterion = nn.CrossEntropyLoss()
iters = 0
for epoch in range(epochs):
    it = tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Epoch: {epoch + 1}/{epochs}",
    )
    for i, data in it:
        input_data = data[KEY_TB_DATA].to(device, dtype=torch.float)
        # Compress 1-hot encoding to single channel
        label = data[KEY_ERA5_LABEL].argmax(dim=1).to(device)

        model.zero_grad()
        output = model(input_data)
        loss = criterion(output, label) + lv_reg_weight * local_variation_loss(
            torch.sigmoid(output)
        )
        loss.backward()
        opt.step()
        loss_vec.append(loss.item())
        it.write(f"{loss.item()}")
        iters += 1
    if epoch > 20:
        for g in opt.param_groups:
            g["lr"] = learning_rate / 100.0
    elif epoch > 10:
        for g in opt.param_groups:
            g["lr"] = learning_rate / 10.0
fmt = "../models/unet-in_{}-nclass_{}-depth_{}-{}.pt"
torch.save(
    model.state_dict(),
    fmt.format(in_chan, nclasses, depth, dt.datetime.now().timestamp()),
)
plt.plot(loss_vec)
plt.show()
