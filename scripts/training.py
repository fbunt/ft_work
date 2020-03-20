import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from dataloading import (
    KEY_INPUT_DATA,
    KEY_VALIDATION_DATA,
    KEY_DIST_DATA,
    DatabaseReference,
    NCTbDataset,
    FTDataset,
    ValidationDataGenerator,
    ViewCopyTransform,
)
from model import UNet, local_variation_loss, ft_loss, LABEL_OTHER
from validation_db_orm import get_db_session


depth = 3
epochs = 10
batch_size = 8
learning_rate = 0.01
learning_momentum = 0.9
l2_reg_weight = 1
lv_reg_weight = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = ViewCopyTransform(15, 62, 12, 191)

water_mask = torch.tensor(
    transform(np.load("../data/masks/ft_esdr_water_mask.npy"))
)
root_data_dir = "../data/training/train"

db = DatabaseReference("../data/dbs/wmo_gsod.db", get_db_session)
model = UNet(6, 1, depth=depth)
model.to(device)
dataset = FTDataset(
    NCTbDataset(root_data_dir, transform=transform),
    ValidationDataGenerator(db, transform=transform),
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
)
opt = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=learning_momentum,
    weight_decay=l2_reg_weight,
)

loss_vec = []

iters = 0
print_period = 50
for epoch in range(epochs):
    print(f"Starting epoch: {epoch}")
    for i, data in tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Epoch: {epoch + 1}/{epochs}",
    ):
        input_data = data[KEY_INPUT_DATA]
        label = data[KEY_VALIDATION_DATA]
        label[..., water_mask] = LABEL_OTHER
        dist_data = data[KEY_DIST_DATA]
        # We are very certain that this is water
        dist_data[..., water_mask] = 1
        input_data = input_data.to(device, dtype=torch.float)

        model.zero_grad()
        output = model(input_data)

        label = label.to(device)
        dist_data = dist_data.to(device)
        loss = ft_loss(output, label, dist_data) + (
            lv_reg_weight * local_variation_loss(output)
        )
        loss.backward()
        opt.step()
        loss_vec.append(loss.item())

        # if iters % print_period == 0:
        #     print(f"{epoch}/{epochs}:{iters}: Loss: {loss}")
        iters += 1
torch.save(
    model.state_dict(), f"../models/unet-{dt.datetime.now().timestamp()}.pt"
)
