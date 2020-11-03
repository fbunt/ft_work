from collections import namedtuple
from torch.nn import DataParallel
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import os
import stat
import torch
import tqdm

from datahandling import (
    ComposedDataset,
    GridsStackDataset,
    IndexEchoDataset,
    NpyDataset,
    RepeatDataset,
    SingleValueGridDataset,
    load_persisted_data_object,
    read_accuracies_file,
    write_accuracies_file,
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
)
from validation_db_orm import get_db_session
import ease_grid as eg


class MinMetricTracker:
    """Keeps track of the minimum value seen"""

    def __init__(self, initial_value):
        self.value = initial_value

    def update(self, new_value):
        ret = False
        if new_value <= self.value:
            # Err on the side of caution and return true for == case
            self.value = new_value
            ret = True
        return ret


class MaxMetricTracker:
    """Keeps track of the maximum value seen"""

    def __init__(self, initial_value):
        self.value = initial_value

    def update(self, new_value):
        ret = False
        if new_value >= self.value:
            # Err on the side of caution and return true for == case
            self.value = new_value
            ret = True
        return ret


def confusion(flat_labels, flat_predictions):
    """Compute the confusion matrix for the given data

    The confusion matrix will be of the form:
                       Actual
                       N     P
                     +----+----+
        Predicted  N | TN | FN |
                     +----+----+
                   P | FP | TP |
                     +----+----+
    This is slightly different but equivalent to the format used in the
    Wikipedia article. Note that a class size of 2 is assumed.

    ref: https://en.wikipedia.org/wiki/Confusion_matrix
    """
    cm = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            cm[j, i] = torch.sum(
                flat_predictions[flat_labels == i] == j
            ).item()
    return cm


MET_ACCURACY = "accuracy"
MET_PRECISION = "met_prec"
MET_RECALL = "met_recall"
MET_F1 = "met_f1"
MET_INFORMEDNESS = "met_informd"
MET_MCC = "met_mcc"


class ConfusionMatrix:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = np.zeros((2, 2), dtype=float)
        self.metrics = {
            MET_PRECISION: 0,
            MET_RECALL: 0,
            MET_F1: 0,
            MET_MCC: 0,
        }

    def update(self, flat_labels, flat_predictions):
        self.cm += confusion(flat_labels, flat_predictions)

    def compute(self):
        try:
            tn = self.cm[0, 0]
            tp = self.cm[1, 1]
            fn = self.cm[0, 1]
            fp = self.cm[1, 0]
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)
            f1 = 2 * precision * recall / (precision + recall)
            informd = precision + specificity - 1
            n = tn + tp + fn + fp
            s = (tp + fn) / n
            p = (tp + fp) / n
            mcc = ((tp / n) - (s * p)) / np.sqrt(p * s * (1 - s) * (1 - p))
            self.metrics[MET_ACCURACY] = accuracy
            self.metrics[MET_PRECISION] = precision
            self.metrics[MET_RECALL] = recall
            self.metrics[MET_F1] = f1
            self.metrics[MET_INFORMEDNESS] = informd
            self.metrics[MET_MCC] = mcc
        except ZeroDivisionError:
            pass

    def update_compute(self, flat_labels, flat_predictions):
        self.update(flat_labels, flat_predictions)
        self.compute()

    def set(self, flat_labels, flat_predictions):
        self.reset()
        self.update_compute(flat_labels, flat_predictions)


class MetricImprovementIndicator:
    def __init__(self, tracker, metric):
        self.tracker = tracker
        self.met = metric
        self.improved = False

    def check(self, cm):
        cm.compute()
        m = cm.metrics[self.met]
        return self.tracker.update(m)


FNAME_MODEL = "model.pt"
FNAME_PREDICTIONS = "pred.npy"


class SnapshotHandler:
    def __init__(self, root_dir, model, config):
        self.root_path = os.path.abspath(root_dir)
        self.model = model
        self.config = config
        self.model_path = os.path.join(self.root_path, FNAME_MODEL)
        self.pred_path = os.path.join(self.root_path, FNAME_PREDICTIONS)
        self.counter = 0

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def take_model_snapshot(self):
        print("\nTaking snapshot")
        self.save_model()
        self.counter += 1
        return True

    def load_best_model(self):
        model.load_state_dict(torch.load(self.model_path))
        return model


def log_metrics(writer, cm, step):
    for label, val in cm.metrics.items():
        writer.add_scalar(label, val, step)


def get_year_str(ya, yb):
    if ya == yb:
        return str(ya)
    else:
        ya, yb = sorted([ya, yb])
        return f"{ya}-{yb}"


def init_run_dir(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    # Dump configuration info
    with open(os.path.join(root_dir, "config"), "w") as fd:
        fd.write(f"{config}\n")
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    train_log_dir = os.path.join(log_dir, "training")
    test_log_dir = os.path.join(log_dir, "test")
    train_summary = SummaryWriter(train_log_dir)
    test_summary = SummaryWriter(test_log_dir)
    show_log_sh = os.path.join(root_dir, "show_log.sh")
    # Create script to view logs
    with open(show_log_sh, "w") as fd:
        fd.write("#!/usr/bin/env bash\n")
        fd.write(f"tensorboard --logdir {os.path.abspath(log_dir)}\n")
        fd.flush()
    st = os.stat(show_log_sh)
    os.chmod(show_log_sh, st.st_mode | stat.S_IXUSR)
    return train_summary, test_summary


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
        p = torch.sigmoid(
            model(v.unsqueeze(0).to(device, dtype=torch.float)).detach()
        )
        p = p.cpu().squeeze().numpy().argmax(0)
        p[..., water_mask] = water_label
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


def plot_accuracies(val_dates, era_acc, aws_acc, root_dir):
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


def plot_predictions(dates, predictions, root_dir, pred_plot_dir):
    # Save prediction plots
    print(f"Creating prediction plots: '{pred_plot_dir}'")
    pfmt = os.path.join(pred_plot_dir, "{:03}.png")
    for i, p in enumerate(tqdm.tqdm(predictions, ncols=80)):
        plt.figure()
        plt.imshow(p, cmap=FT_CMAP, vmin=LABEL_FROZEN, vmax=LABEL_OTHER)
        plt.title(f"Day: {i + 1}, {dates[i]}")
        plt.tight_layout(pad=2)
        plt.savefig(pfmt.format(i + 1), dpi=400)
        plt.close()


def add_plots_to_run_dir(root_dir, do_val_plots, do_pred_plots):
    if do_val_plots:
        dates, era_acc, aws_acc = read_accuracies_file(
            os.path.join(root_dir, "acc.csv")
        )
        plot_accuracies(dates, era_acc, aws_acc, root_dir)
    if do_pred_plots:
        dates, _, _ = read_accuracies_file(os.path.join(root_dir, "acc.csv"))
        preds = np.load(os.path.join(root_dir, "pred.npy"))
        plot_predictions(
            dates, preds, root_dir, os.path.join(root_dir, "pred_plots")
        )


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
        loss += binary_cross_entropy_with_logits(pred, labels)
    return loss


def normalize(x):
    if len(x.shape) < 4:
        return (x - x.min()) / (x.max() - x.min())
    else:
        x = x.copy()
        for i in range(x.shape[1]):
            chan = x[:, i]
            x[:, i] = (chan - chan.max()) / (chan.max() - chan.min())
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
    aws_data,
    land_mask,
    water_mask,
    summary,
    epoch,
    config,
    is_train,
    confusion_matrix=None,
):
    loss_sum = 0.0
    for i, (input_data, batch_era, batch_idxs, batch_bce_weights) in iterator:
        input_data = input_data.to(device, dtype=torch.float)

        if is_train:
            model.zero_grad()
        log_class_prob = model(input_data)
        class_prob = torch.sigmoid(log_class_prob)
        #
        # ERA/AWS
        #
        flat_era = batch_era.view(batch_era.size(0), batch_era.size(1), -1)
        flat_bce_weights = batch_bce_weights.view(
            batch_bce_weights.size(0), batch_bce_weights.size(1), -1
        )
        batch_aws = [aws_data[idx] for idx in batch_idxs]
        batch_aws_fzn_idxs = [v[0] for v in batch_aws]
        batch_aws_thw_idxs = [v[1] for v in batch_aws]
        for i in range(len(flat_era)):
            i_fzn = batch_aws_fzn_idxs[i]
            i_thw = batch_aws_thw_idxs[i]
            flat_era[i, LABEL_FROZEN, i_fzn] = 1
            flat_era[i, LABEL_THAWED, i_thw] = 1
            flat_bce_weights[i, :, i_fzn] = config.aws_bce_weight
            flat_bce_weights[i, :, i_thw] = config.aws_bce_weight
        batch_era = batch_era.to(device)
        batch_bce_weights = batch_bce_weights.to(device)
        comb_loss = binary_cross_entropy_with_logits(
            log_class_prob, batch_era, batch_bce_weights
        )
        comb_loss *= config.main_loss_weight

        #
        # Local variation
        #
        # Minimize high frequency variation
        lv_loss = local_variation_loss(class_prob)
        lv_loss *= config.lv_reg_weight
        loss = comb_loss
        loss += lv_loss
        if is_train:
            loss.backward()
            optimizer.step()
        loss_sum += loss

        if confusion_matrix is not None:
            flat_labels = batch_era.argmax(1)[..., land_mask].view(-1)
            flat_predictions = class_prob.argmax(1)[..., land_mask].view(-1)
            confusion_matrix.update(flat_labels, flat_predictions)
    loss_mean = loss_sum / len(iterator)
    summary.add_scalar("Loss", loss_mean.item(), epoch)
    return loss_mean


def test(
    model,
    device,
    dataloader,
    optimizer,
    aws_data,
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
        desc=f"-Test: {epoch + 1}/{config.epochs}",
    )
    cm = ConfusionMatrix()
    with torch.no_grad():
        loss = run_model(
            model,
            device,
            it,
            optimizer,
            aws_data,
            land_mask,
            water_mask,
            summary,
            epoch,
            config,
            False,
            cm,
        )
    return loss.item(), cm


def train(
    model,
    device,
    dataloader,
    optimizer,
    aws_data,
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
        aws_data,
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
        "lr_shed_multi",
        "learning_rate",
        "lr_milestones",
        "lr_gamma",
        "lr_step_gamma",
        "aws_use_valid_mask",
        "val_use_valid_mask",
        "optimizer",
        "do_val_plots",
        "do_pred_plots",
        "normalize",
        "randomize_offset",
        "use_land_mask",
        "use_dem",
        "use_latitude",
        "use_day_of_year",
        "use_solar",
        "use_snow",
        "use_prior_day",
        "region",
        "train_start_year",
        "train_end_year",
        "test_start_year",
        "test_end_year",
        "l2_reg_weight",
        "main_loss_weight",
        "aws_bce_weight",
        "lv_reg_weight",
    ),
)


if __name__ == "__main__":
    tb_channels = [0, 1, 2, 3, 4]
    _use_land_mask = False
    _use_dem = True
    _use_lat = False
    _use_day_of_year = False
    _use_solar = False
    _use_snow = False
    _use_prior_day = True
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
        epochs=500,
        batch_size=16,
        batch_shuffle=True,
        drop_last=False,
        lr_shed_multi=True,
        learning_rate=1e-4,
        lr_milestones=[100, 200, 300, 350, 400, 450],
        lr_gamma=0.89,
        lr_step_gamma=0.5,
        aws_use_valid_mask=False,
        val_use_valid_mask=False,
        optimizer=torch.optim.Adam,
        do_val_plots=True,
        do_pred_plots=False,
        normalize=False,
        randomize_offset=False,
        use_land_mask=_use_land_mask,
        use_dem=_use_dem,
        use_latitude=_use_lat,
        use_day_of_year=_use_day_of_year,
        use_solar=_use_solar,
        use_snow=_use_snow,
        use_prior_day=_use_prior_day,
        region=N45W,
        train_start_year=2005,
        train_end_year=2014,
        test_start_year=2015,
        test_end_year=2015,
        l2_reg_weight=1e-2,
        main_loss_weight=1e0,
        aws_bce_weight=5e0,
        lv_reg_weight=5e-2,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_year_str = get_year_str(
        config.train_start_year, config.train_end_year
    )
    test_year_str = get_year_str(config.test_start_year, config.test_end_year)

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
        f"../data/cleaned/tb-D-{train_year_str}-{config.region}.npy",
        transform(np.load("../data/z/dem.npy")),
        land_channel,
        lat_channel,
        f"../data/cleaned/date_map-{train_year_str}-{config.region}.csv",
        data_grid_shape,
        f"../data/cleaned/solar_rad-AM-{train_year_str}-{config.region}.npy",
        f"../data/cleaned/snow_cover-{train_year_str}-{config.region}.npy",
        tb_channels=tb_channels,
    )
    # AWS
    train_aws_data = load_persisted_data_object(
        f"../data/cleaned/aws_data-AM-{train_year_str}-{config.region}.pkl"
    )
    # ERA
    train_era_ds = NpyDataset(
        f"../data/cleaned/era5-ft-am-{train_year_str}-{config.region}.npy"
    )
    if config.use_prior_day:
        train_era_ds = Subset(
            train_era_ds, list(range(1, len(train_input_ds) + 1))
        )
        train_idx_ds = IndexEchoDataset(len(train_input_ds), offset=1)
    else:
        train_idx_ds = IndexEchoDataset(len(train_input_ds))
    ws = torch.zeros((1, *land_mask.shape), dtype=torch.float)
    ws[..., land_mask] = 1.0
    train_weights_ds = RepeatDataset(ws, len(train_input_ds))
    train_ds = ComposedDataset(
        [train_input_ds, train_era_ds, train_idx_ds, train_weights_ds]
    )

    #
    # Test Data
    #
    test_input_ds = build_input_dataset(
        config,
        f"../data/cleaned/tb-D-{test_year_str}-{config.region}.npy",
        transform(np.load("../data/z/dem.npy")),
        land_channel,
        lat_channel,
        f"../data/cleaned/date_map-{test_year_str}-{config.region}.csv",
        data_grid_shape,
        f"../data/cleaned/solar_rad-AM-{test_year_str}-{config.region}.npy",
        f"../data/cleaned/snow_cover-{test_year_str}-{config.region}.npy",
        tb_channels=tb_channels,
    )
    test_reduced_indices = list(range(1, len(test_input_ds) + 1))
    # AWS
    test_aws_data = load_persisted_data_object(
        f"../data/cleaned/aws_data-AM-{test_year_str}-{config.region}.pkl"
    )
    # ERA
    test_era_ds = NpyDataset(
        f"../data/cleaned/era5-ft-am-{test_year_str}-{config.region}.npy"
    )
    if config.use_prior_day:
        test_era_ds = Subset(test_era_ds, test_reduced_indices)
        test_idx_ds = IndexEchoDataset(len(test_input_ds), offset=1)
    else:
        test_idx_ds = IndexEchoDataset(len(test_input_ds))
    test_weights_ds = RepeatDataset(ws, len(test_input_ds))
    test_ds = ComposedDataset(
        [test_input_ds, test_era_ds, test_idx_ds, test_weights_ds]
    )

    model = UNet(
        config.in_chan,
        config.n_classes,
        depth=config.depth,
        base_filter_bank_size=config.base_filters,
    )
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device)
    opt = config.optimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_reg_weight,
    )
    if not config.lr_shed_multi:
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, config.lr_gamma)
    else:
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, config.lr_milestones, config.lr_step_gamma
        )

    # Create run dir and fill with info
    root_dir = f'../runs/{str(dt.datetime.now()).replace(" ", "-")}'
    train_summary, test_summary = init_run_dir(root_dir)
    mpath = os.path.join(root_dir, "model.pt")

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
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    snap_handler = SnapshotHandler(root_dir, model, config)
    metric_checker = MetricImprovementIndicator(
        MaxMetricTracker(-np.inf), MET_MCC
    )
    snap_handler.take_model_snapshot()
    try:
        for epoch in range(config.epochs):
            train_summary.add_scalar(
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
                train_aws_data,
                land_mask,
                water_mask,
                train_summary,
                epoch,
                config,
            )
            loss, cm = test(
                model,
                device,
                test_dataloader,
                opt,
                test_aws_data,
                land_mask,
                water_mask,
                test_summary,
                epoch,
                config,
            )
            if metric_checker.check(cm):
                snap_handler.take_model_snapshot()
            log_metrics(test_summary, cm, epoch)
            sched.step()
    except KeyboardInterrupt:
        print("Exiting training loop")
    except Exception as e:
        print(f"\n{e}")
        raise e
    finally:
        train_summary.close()
        test_summary.close()
        # Free up data for GC
        train_input_ds = None
        train_era_ds = None
        train_idx_ds = None
        train_ds = None
        train_dataloader = None

        # Validation
        val_dates = load_dates(
            f"../data/cleaned/date_map-{test_year_str}-{config.region}.csv"
        )
        val_mask_ds = NpyDataset(
            f"../data/cleaned/tb_valid_mask-D-{test_year_str}"
            f"-{config.region}.npy"
        )
        if config.use_prior_day:
            val_dates = Subset(val_dates, test_reduced_indices)
            val_mask_ds = Subset(val_mask_ds, test_reduced_indices)

        model = snap_handler.load_best_model()
        # Log results
        pred_plot_dir = os.path.join(root_dir, "pred_plots")
        os.makedirs(pred_plot_dir)
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
        write_accuracies_file(val_dates, era_acc, aws_acc, acc_file)
        print(f"Era Mean Acc: {era_acc.mean()}")
        print(f"AWS Mean Acc: {aws_acc.mean()}")
        add_plots_to_run_dir(
            root_dir, config.do_val_plots, config.do_pred_plots
        )
