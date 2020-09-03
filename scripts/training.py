from collections import namedtuple
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
    AK,
    N45W,
    REGION_TO_TRANS,
)
from utils import FT_CMAP
from validate import (
    RETRIEVAL_MIN,
    WMOValidationPointFetcher,
    WMOValidator,
    ft_model_zero_threshold,
    get_nearest_flat_idxs_and_values,
)
from validation_db_orm import get_db_session
import ease_grid as eg


def init_run_dir(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    # Dump configuration info
    with open(os.path.join(root_dir, "config"), "w") as fd:
        fd.write(f"{config}\n")
    log_dir = os.path.join(root_dir, "logs")
    summary = SummaryWriter(log_dir)
    show_log_sh = os.path.join(root_dir, "show_log.sh")
    # Create script to view logs
    with open(show_log_sh, "w") as fd:
        fd.write("#!/usr/bin/env bash\n")
        fd.write(f"tensorboard --logdir {os.path.abspath(log_dir)}\n")
        fd.flush()
    st = os.stat(show_log_sh)
    os.chmod(show_log_sh, st.st_mode | stat.S_IXUSR)
    return summary


def load_dates(path):
    dates = []
    with open(path) as fd:
        for line in fd:
            i, ds = line.strip().split(",")
            dates.append(dt.date.fromisoformat(ds))
    return dates


def get_predictions(input_ds, model, water_mask, water_label, device, config):
    pred = []
    for i, v in enumerate(tqdm.tqdm(input_ds, ncols=80)):
        if config.mask_water:
            p = torch.softmax(
                model(v.unsqueeze(0).to(device, dtype=torch.float)).detach(), 1
            )
            p = p.cpu().squeeze().numpy().argmax(0)
            p[..., water_mask] = water_label
        else:
            p = (
                torch.softmax(
                    model(
                        v.unsqueeze(0).to(device, dtype=torch.float)
                    ).detach(),
                    1,
                )
                .cpu()
                .squeeze()
                .numpy()
                .argmax(0)
            )
        pred.append(p)
    pred = np.array(pred)
    return pred


def validate_against_era5(pred, era_ds, valid_mask_ds, land_mask, config):
    if config.val_use_valid_mask:
        masked_pred = [
            p[land_mask & vmask] for p, vmask in zip(pred, val_mask_ds)
        ]
        era = [
            v.argmax(0)[land_mask & vmask]
            for v, vmask in zip(era_ds, val_mask_ds)
        ]
    else:
        masked_pred = [p[land_mask] for p in pred]
        era = [v.argmax(0)[land_mask] for v in era_ds]
    era_acc = np.array(
        [(p == e).sum() / p.size for p, e in zip(masked_pred, era)]
    )
    return era_acc * 100


def validate_against_aws_db(
    pred, db, dates, view_transform, valid_mask_ds, land_mask, config
):
    pf = WMOValidationPointFetcher(db, RETRIEVAL_MIN)
    elon, elat = [view_transform(i) for i in eg.v1_get_full_grid_lonlat(eg.ML)]
    aws_val = WMOValidator(pf)
    if config.val_use_valid_mask:
        if isinstance(valid_mask_ds[0], np.ndarray):
            mask = (land_mask & vmask for vmask in valid_mask_ds)
        else:
            mask = (land_mask & vmask.numpy() for vmask in valid_mask_ds)
    else:
        mask = land_mask
        if not isinstance(mask, np.ndarray):
            mask = mask.numpy()
    aws_acc = aws_val.validate_bounded(
        pred,
        dates,
        elon,
        elat,
        mask,
        show_progress=True,
        variable_mask=config.val_use_valid_mask,
    )
    return aws_acc * 100


def plot_results(predictions, era_acc, aws_acc, root_dir, pred_plot_dir):
    # ERA
    plt.figure()
    plt.plot(val_dates, era_acc, lw=1, label="ERA5")
    plt.ylim(0, 100)
    plt.title(f"ERA Accuracy: {era_acc.mean():.3}%")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root_dir, "acc_era_plot.png"), dpi=300)
    # AWS
    plt.figure()
    plt.plot(val_dates, aws_acc, lw=1, label="AWS")
    plt.ylim(0, 100)
    plt.title(f"AWS Accuracy: {aws_acc.mean():.3}%")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root_dir, "acc_aws_plot.png"), dpi=300)
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
    plt.savefig(os.path.join(root_dir, "acc_plot.png"), dpi=300)
    plt.close()
    # Save prediction plots
    print(f"Creating prediction plots: '{pred_plot_dir}'")
    pfmt = os.path.join(pred_plot_dir, "{:03}.png")
    for i, p in enumerate(tqdm.tqdm(predictions, ncols=80)):
        plt.figure()
        plt.imshow(p, cmap=FT_CMAP, vmin=LABEL_FROZEN, vmax=LABEL_OTHER)
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
    snow_path,
    tb_channels=None,
):
    datasets = []
    tb_ds = np.load(tb_path)
    if config.normalize:
        tb_ds = normalize(tb_ds)
    tb_ds = NpyDataset(tb_ds, channels=tb_channels)
    reduced_indices = list(range(1, len(tb_ds)))
    # Land channel
    if config.use_land_mask:
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
    if config.use_snow:
        ds = np.load(snow_path)
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


def run_model(
    model,
    device,
    iterator,
    optimizer,
    era_criterion,
    land_mask,
    water_mask,
    summary,
    epoch,
    config,
    is_train,
):
    base_step = epoch * len(iterator)
    for i, (input_data, batch_era, batch_idxs) in iterator:
        step = base_step + i
        input_data = input_data.to(device, dtype=torch.float)
        # Compress 1-hot encoding to single channel
        batch_era = batch_era.argmax(dim=1).to(device)

        if is_train:
            model.zero_grad()
        log_class_prob = model(input_data)
        class_prob = torch.softmax(log_class_prob, 1)
        #
        # ERA
        #
        era_loss = criterion(
            log_class_prob[..., land_mask], batch_era[..., land_mask]
        )
        era_loss *= config.era_weight
        #
        # AWS loss
        #
        batch_aws = [train_aws_data[idx] for idx in batch_idxs]
        batch_aws_flat_idxs = [v[0] for v in batch_aws]
        batch_aws_labels = [v[1] for v in batch_aws]
        aws_loss = aws_loss_func(
            log_class_prob,
            batch_aws_flat_idxs,
            batch_aws_labels,
            config,
            device,
        )
        aws_loss *= config.aws_loss_weight
        if not config.mask_water:
            #
            # Land/Water
            #
            # Minimize the probabilities of FT classes in water regions
            land_loss = class_prob[:, LABEL_FROZEN, water_mask].sum()
            land_loss += class_prob[:, LABEL_THAWED, water_mask].sum()
            # Minimize the probability of OTHER class in land regions
            land_loss += class_prob[:, LABEL_OTHER, land_mask].sum()
            land_loss *= config.land_reg_weight
        #
        # Local variation
        #
        # Minimize high frequency variation
        lv_loss = local_variation_loss(class_prob)
        lv_loss *= config.lv_reg_weight
        loss = era_loss
        loss += aws_loss
        if not config.mask_water:
            loss += land_loss
        loss += lv_loss
        if is_train:
            summary.add_scalar("ERA Loss", era_loss.item(), step)
            summary.add_scalar("AWS Loss", aws_loss.item(), step)
            if not config.mask_water:
                summary.add_scalar("Land Loss", land_loss.item(), step)
            summary.add_scalar("LV Loss", lv_loss.item(), step)
            summary.add_scalar("training_loss", loss.item(), step)
        else:
            summary.add_scalar("test_loss", loss.item(), step)
        if is_train:
            loss.backward()
            optimizer.step()


def test(
    model,
    device,
    dataloader,
    optimizer,
    era_criterion,
    land_mask,
    water_mask,
    summary,
    epoch,
    config,
):
    model.eval()
    it = tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Test: {epoch + 1}/{config.epochs}",
    )
    with torch.no_grad():
        run_model(
            model,
            device,
            it,
            optimizer,
            era_criterion,
            land_mask,
            water_mask,
            summary,
            epoch,
            config,
            False,
        )


def train(
    model,
    device,
    dataloader,
    optimizer,
    era_criterion,
    land_mask,
    water_mask,
    summary,
    epoch,
    config,
):
    model.train()
    it = tqdm.tqdm(
        enumerate(dataloader),
        ncols=80,
        total=len(dataloader),
        desc=f"Train: {epoch + 1}/{config.epochs}",
    )
    run_model(
        model,
        device,
        it,
        optimizer,
        era_criterion,
        land_mask,
        water_mask,
        summary,
        epoch,
        config,
        True,
    )


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
        "do_test",
        "normalize",
        "randomize_offset",
        "mask_water",
        "use_land_mask",
        "use_dem",
        "use_latitude",
        "use_day_of_year",
        "use_solar",
        "use_snow",
        "use_prior_day",
        "region",
        "l2_reg_weight",
        "era_weight",
        "aws_loss_weight",
        "land_reg_weight",
        "lv_reg_weight",
    ),
)


tb_channels = [0, 1, 2, 3, 4]
_use_land_mask = False
_use_dem = True
_use_lat = False
_use_day_of_year = False
_use_solar = False
_use_snow = False
_use_prior_day = False
config = Config(
    in_chan=len(tb_channels)
    + _use_dem
    + _use_lat
    + _use_day_of_year
    + _use_solar
    + _use_snow
    + (len(tb_channels) * _use_prior_day),
    n_classes=2,
    depth=4,
    base_filters=64,
    epochs=50,
    batch_size=16,
    batch_shuffle=False,
    drop_last=False,
    learning_rate=5e-4,
    lr_gamma=0.89,
    aws_use_valid_mask=False,
    val_use_valid_mask=False,
    optimizer=torch.optim.Adam,
    do_test=True,
    normalize=False,
    randomize_offset=False,
    mask_water=True,
    use_land_mask=_use_land_mask,
    use_dem=_use_dem,
    use_latitude=_use_lat,
    use_day_of_year=_use_day_of_year,
    use_solar=_use_solar,
    use_snow=_use_snow,
    use_prior_day=_use_prior_day,
    region=N45W,
    l2_reg_weight=1e-2,
    era_weight=1e0,
    aws_loss_weight=5e-2,
    land_reg_weight=1e-4,
    lv_reg_weight=5e-2,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.mask_water:
    assert (
        config.n_classes == 2
    ), "Can only have 2 output channels if masking water"
else:
    assert (
        config.n_classes == 3
    ), "Must have 3 output channels if not masking water"

transform = REGION_TO_TRANS[config.region]
base_water_mask = np.load("../data/masks/ft_esdr_water_mask.npy")
water_mask = torch.tensor(transform(base_water_mask))
land_mask = ~water_mask
land_mask_np = land_mask.numpy()
land_channel = torch.tensor(land_mask_np).float()
dem_channel = torch.tensor(transform(np.load("../data/z/dem.npy"))).float()
elon, elat = eg.v1_get_full_grid_lonlat(eg.ML)
lat_channel = torch.tensor(transform(elat)).float()
data_grid_shape = land_mask_np.shape


#
# Training data
#
train_input_ds = build_input_dataset(
    config,
    f"../data/cleaned/tb-D-2007-2010-{config.region}.npy",
    transform(np.load("../data/z/dem.npy")),
    land_channel,
    lat_channel,
    f"../data/cleaned/date_map-2007-2010-{config.region}.csv",
    data_grid_shape,
    f"../data/cleaned/solar_rad-AM-2007-2010-{config.region}.npy",
    f"../data/cleaned/snow_cover-2007-2010-{config.region}.npy",
    tb_channels=tb_channels,
)
# AWS
train_aws_data = get_aws_data(
    f"../data/cleaned/date_map-2007-2010-{config.region}.csv",
    f"../data/cleaned/tb_valid_mask-D-2007-2010-{config.region}.npy",
    "../data/dbs/wmo_gsod.db",
    land_mask_np,
    transform,
    RETRIEVAL_MIN,
    config,
)
# ERA
train_era_ds = NpyDataset(
    f"../data/cleaned/era5-t2m-am-2007-2010-{config.region}.npy"
)
if config.use_prior_day:
    train_era_ds = Subset(
        train_era_ds, list(range(1, len(train_input_ds) + 1))
    )
    train_idx_ds = IndexEchoDataset(len(train_input_ds), offset=1)
else:
    train_idx_ds = IndexEchoDataset(len(train_input_ds))
train_ds = ComposedDataset([train_input_ds, train_era_ds, train_idx_ds])

#
# Test Data
#
test_input_ds = build_input_dataset(
    config,
    f"../data/cleaned/tb-D-2015-{config.region}.npy",
    transform(np.load("../data/z/dem.npy")),
    land_channel,
    lat_channel,
    f"../data/cleaned/date_map-2015-{config.region}.csv",
    data_grid_shape,
    f"../data/cleaned/solar_rad-AM-2015-{config.region}.npy",
    f"../data/cleaned/snow_cover-2015-{config.region}.npy",
    tb_channels=tb_channels,
)
test_reduced_indices = list(range(1, len(test_input_ds) + 1))
# AWS
test_aws_data = get_aws_data(
    f"../data/cleaned/date_map-2015-{config.region}.csv",
    f"../data/cleaned/tb_valid_mask-D-2015-{config.region}.npy",
    "../data/dbs/wmo_gsod.db",
    land_mask_np,
    transform,
    RETRIEVAL_MIN,
    config,
)
# ERA
test_era_ds = NpyDataset(
    f"../data/cleaned/era5-t2m-am-2015-{config.region}.npy"
)
if config.use_prior_day:
    test_era_ds = Subset(test_era_ds, test_reduced_indices)
    test_idx_ds = IndexEchoDataset(len(test_input_ds), offset=1)
else:
    test_idx_ds = IndexEchoDataset(len(test_input_ds))
test_ds = ComposedDataset([test_input_ds, test_era_ds, test_idx_ds])

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
root_dir = f'../runs/{str(dt.datetime.now()).replace(" ", "-")}'
summary = init_run_dir(root_dir)

criterion = nn.CrossEntropyLoss()
if config.randomize_offset:
    rng = np.random.default_rng()
    day_indices = list(range(len(train_ds)))
else:
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.batch_shuffle,
        drop_last=config.drop_last,
    )
test_dataloader = torch.utils.data.DataLoader(
    test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False,
)
for epoch in range(config.epochs):
    summary.add_scalar(
        "learning_rate", next(iter(opt.param_groups))["lr"], epoch
    )
    if config.randomize_offset:
        offset = rng.choice(7, 1)[0]
        train_dataloader = torch.utils.data.DataLoader(
            Subset(train_ds, day_indices[offset:]),
            batch_size=config.batch_size,
            shuffle=config.batch_shuffle,
            drop_last=config.drop_last,
        )
    train(
        model,
        device,
        train_dataloader,
        opt,
        criterion,
        land_mask,
        water_mask,
        summary,
        epoch,
        config,
    )
    if config.do_test:
        test(
            model,
            device,
            test_dataloader,
            opt,
            criterion,
            land_mask,
            water_mask,
            summary,
            epoch,
            config,
        )
    sched.step()
summary.close()
# Free up data for GC
train_input_ds = None
train_era_ds = None
train_idx_ds = None
train_ds = None
train_dataloader = None

# Validation
val_dates = load_dates(f"../data/cleaned/date_map-2015-{config.region}.csv")
val_mask_ds = NpyDataset(
    f"../data/cleaned/tb_valid_mask-D-2015-{config.region}.npy"
)
if config.use_prior_day:
    val_dates = Subset(val_dates, test_reduced_indices)
    val_mask_ds = Subset(val_mask_ds, test_reduced_indices)
# Log results
os.makedirs(root_dir, exist_ok=True)
pred_plot_dir = os.path.join(root_dir, "pred_plots")
if os.path.isdir(pred_plot_dir):
    shutil.rmtree(pred_plot_dir)
os.makedirs(pred_plot_dir)
# Save model
mpath = os.path.join(root_dir, "model.pt")
print(f"Saving model: '{mpath}'")
torch.save(model.state_dict(), mpath)
# Create and save predictions for test data
print("Generating predictions")
pred = get_predictions(
    test_input_ds, model, ~land_mask, LABEL_OTHER, device, config
)
ppath = os.path.join(root_dir, "pred.npy")
print(f"Saving predictions: '{ppath}'")
np.save(ppath, pred)
# Validate against ERA5
print("Validating against ERA5")
era_acc = validate_against_era5(
    pred, test_era_ds, val_mask_ds, land_mask, config
)
# Validate against AWS DB
db = get_db_session("../data/dbs/wmo_gsod.db")
aws_acc = validate_against_aws_db(
    pred, db, val_dates, transform, val_mask_ds, land_mask, config
)
db.close()
# Write accuracies
acc_file = os.path.join(root_dir, "acc.csv")
with open(acc_file, "w") as fd:
    for d, ae, aa in zip(val_dates, era_acc, aws_acc):
        fd.write(f"{d},{ae},{aa}\n")

plot_results(pred, era_acc, aws_acc, root_dir, pred_plot_dir)
