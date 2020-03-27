import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from dataloading import (
    AWSFuzzyLabelDataset,
    DatabaseReference,
    FTDataset,
    KEY_INPUT_DATA,
    KEY_LABEL_DATA,
    NCTbDataset,
    ViewCopyTransform,
)
from model import UNet, local_variation_loss, ft_loss, LABEL_OTHER
from validation_db_orm import get_db_session


in_chan = 5
nclasses = 3
depth = 4
base_filters = 32
epochs = 1
batch_size = 8
learning_rate = 0.01
learning_momentum = 0.9
l2_reg_weight = 1
lv_reg_weight = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = ViewCopyTransform(15, 62, 12, 191)

base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))
root_data_dir = "../data/training/train"

db = DatabaseReference("../data/dbs/wmo_gsod.db", get_db_session)
model = UNet(
    in_chan, nclasses, depth=depth, base_filter_bank_size=base_filters
)
model.to(device)
dataset = FTDataset(
    NCTbDataset(root_data_dir, transform=transform),
    AWSFuzzyLabelDataset(db, transform=transform, other_mask=base_water_mask),
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
)
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
print_period = 50
for epoch in range(epochs):
    it = tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Epoch: {epoch + 1}/{epochs}",
    )
    for i, data in it:
        input_data = data[KEY_INPUT_DATA]
        label = data[KEY_LABEL_DATA]
        it.write(f"{label.shape}")
        input_data = input_data.to(device, dtype=torch.float)

        model.zero_grad()
        output = model(input_data)
        label = label.to(device)
        loss = criterion(output, label) + (
            lv_reg_weight * local_variation_loss(torch.sigmoid(output))
        )
        loss.backward()
        opt.step()
        loss_vec.append(loss.item())
        it.write(f"{loss.item()}")
        iters += 1
fmt = "../models/unet-in_{}-nclass_{}-depth_{}-{}.pt"
torch.save(
    model.state_dict(),
    fmt.format(in_chan, nclasses, depth, dt.datetime.now().timestamp()),
)
plt.plot(loss_vec)
plt.show()
