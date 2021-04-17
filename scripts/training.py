import argparse
import datetime as dt
import numpy as np
import os
import shutil
import torch
import tqdm
from torch.nn import DataParallel
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Subset

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD = True
except ImportError:
    TENSORBOARD = False

from config import create_model, load_config
from datahandling import (
    ArrayDataset,
    ComposedDataset,
    GridsStackDataset,
    NpyDataset,
    RepeatDataset,
    SingleValueGridDataset,
    TilingDataset,
    dataset_to_array,
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
from utils import FT_CMAP, validate_dir_path, validate_file_path
from validate import validate_against_aws_db, validate_against_grid_stack
from validation_db_orm import get_db_session


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c",
        "--config_path",
        default="../config/config_default.yaml",
        type=validate_file_path,
        help="Path to config file. If not provided, the default file is used",
    )
    p.add_argument(
        "-R",
        "--resumable",
        action="store_true",
        help="Allow program to resume using run dir in the config file",
    )
    return p


class SummaryWriterDummy:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class MinMetricTracker:
    """Keeps track of the minimum value seen"""

    def __init__(self, initial_value):
        self.value = initial_value

    def update(self, new_value):
        ret = False
        if not np.isinf(new_value) and new_value <= self.value:
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
        if not np.isinf(new_value) and new_value >= self.value:
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


class MetricImprovementChecker:
    def __init__(self, tracker, metric):
        self.tracker = tracker
        self.met = metric
        self.improved = False

    def check(self, cm):
        cm.compute()
        m = cm.metrics[self.met]
        return self.tracker.update(m)


FNAME_MODEL = "model.pt"
FNAME_MODEL_TMP = "model.pt.tmp"
FNAME_PREDICTIONS = "pred.npy"
FNAME_PROBABILITIES = "prob.npy"
FNAME_FULL_SNAPSHOT = "snap_full.pt"
FNAME_FULL_SNAPSHOT_TMP = "snap_full.pt.tmp"
SNAP_KEY_EPOCH = "epoch"
SNAP_KEY_MODEL = "model"
SNAP_KEY_OPTIMIZER = "optimizer"
SNAP_KEY_LR_SCHED = "lr_sched"
SNAP_KEY_CHECKER_VAL = "checker_val"


class SnapshotHandler:
    def __init__(self, root_dir, model, optimizer, lr_sched, checker):
        self.root_path = os.path.abspath(root_dir)
        self.model = model
        self.opt = optimizer
        self.lr_sched = lr_sched
        self.checker = checker
        self.model_path = os.path.join(self.root_path, FNAME_MODEL)
        self.model_path_tmp = os.path.join(self.root_path, FNAME_MODEL_TMP)
        self.full_snap_path = os.path.join(self.root_path, FNAME_FULL_SNAPSHOT)
        self.full_snap_path_tmp = os.path.join(
            self.root_path, FNAME_FULL_SNAPSHOT_TMP
        )
        self.counter = 0

    def save_model(self):
        # Make sure that there is always a possible recovery mode in case of
        # early termination during file write.
        torch.save(self.model.state_dict(), self.model_path_tmp)
        os.replace(self.model_path_tmp, self.model_path)

    def can_resume(self):
        return os.path.isfile(self.full_snap_path)

    def take_model_snapshot(self):
        print("\nTaking snapshot")
        self.save_model()
        self.counter += 1
        return True

    def take_full_snapshot(self, epoch):
        snap = {
            SNAP_KEY_EPOCH: epoch,
            SNAP_KEY_MODEL: self.model.state_dict(),
            SNAP_KEY_OPTIMIZER: self.opt.state_dict(),
            SNAP_KEY_LR_SCHED: self.lr_sched.state_dict(),
            SNAP_KEY_CHECKER_VAL: self.checker.tracker.value,
        }
        # Make sure that there is always a possible recovery mode in case of
        # early termination during file write.
        torch.save(snap, self.full_snap_path_tmp)
        os.replace(self.full_snap_path_tmp, self.full_snap_path)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        return self.model

    def load_full_snapshot(self):
        print("Loading full snapshot")
        snap = torch.load(self.full_snap_path)
        epoch = snap[SNAP_KEY_EPOCH]
        self.model.load_state_dict(snap[SNAP_KEY_MODEL])
        self.opt.load_state_dict(snap[SNAP_KEY_OPTIMIZER])
        self.lr_sched.load_state_dict(snap[SNAP_KEY_LR_SCHED])
        self.checker.tracker.value = snap[SNAP_KEY_CHECKER_VAL]
        return epoch, self.model, self.opt, self.lr_sched, self.checker


def log_metrics(writer, cm, step):
    for label, val in cm.metrics.items():
        writer.add_scalar(label, val, step)


def get_year_str(ya, yb):
    if ya == yb:
        return str(ya)
    else:
        ya, yb = sorted([ya, yb])
        return f"{ya}-{yb}"


FNAME_CONFIG = "config.yaml"


def init_run_dir(root_dir, config_path, resume=False):
    log_dir = os.path.join(root_dir, "logs")
    if not resume:
        os.makedirs(root_dir, exist_ok=True)
        # Dump configuration info
        shutil.copyfile(config_path, os.path.join(root_dir, FNAME_CONFIG))
        os.makedirs(log_dir, exist_ok=True)
    train_log_dir = os.path.join(log_dir, "training")
    test_log_dir = os.path.join(log_dir, "test")
    summary_class = SummaryWriter if TENSORBOARD else SummaryWriterDummy
    train_summary = summary_class(train_log_dir)
    test_summary = summary_class(test_log_dir)
    return train_summary, test_summary


def load_dates(path):
    dates = []
    with open(path) as fd:
        for line in fd:
            i, ds = line.strip().split(",")
            dates.append(dt.date.fromisoformat(ds))
    return dates


def get_predictions(input_dl, model, water_mask, water_label, device, config):
    pred = []
    prob = []
    with torch.no_grad():
        for i, v in tqdm.tqdm(
            enumerate(input_dl), ncols=80, total=len(input_dl)
        ):
            output = model(v.to(device, dtype=torch.float))
            p = torch.sigmoid(output).cpu().numpy()
            prob.extend(p)
            predictions = p.argmax(1)
            predictions[..., water_mask] = water_label
            pred.extend(predictions)
    pred = np.array(pred)
    # Transform the class channels into a Bernoulli distribution. This is
    # necessary because each channel is 0 <= chan <= 1 but their sum is
    # 0 < sum < 2. Then take only the thawed channel to compress the data.
    prob = [(x / x.sum(0))[1] for x in prob]
    prob = np.array(prob)
    return pred, prob


def validate_against_era5(pred, era_ds, dates, land_mask):
    if not isinstance(land_mask, np.ndarray):
        land_mask = land_mask.numpy()
    df = validate_against_grid_stack(pred, era_ds, dates, land_mask)
    return df.acc.to_numpy() * 100


def validate_against_aws(
    pred, db, dates, lon_grid, lat_grid, land_mask, config
):
    if not isinstance(land_mask, np.ndarray):
        land_mask = land_mask.numpy()
    df = validate_against_aws_db(
        pred,
        db,
        dates,
        lon_grid,
        lat_grid,
        land_mask,
        am_pm=config.am_pm,
    )
    return df.acc.to_numpy() * 100


def plot_accuracies(val_dates, era_acc, aws_acc, root_dir):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tkr

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
        f"Mean Accuracy: ERA: {era_acc.mean():.3}% AWS: {aws_acc.mean():.3}%"
    )
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(5))
    plt.grid(True, which="both", alpha=0.7, lw=0.5, ls=":")
    plt.savefig(os.path.join(root_dir, "acc_plot.png"), dpi=300)
    plt.close()


def plot_predictions(dates, predictions, pred_plot_dir):
    import matplotlib.pyplot as plt

    if not os.path.isdir(pred_plot_dir):
        os.makedirs(pred_plot_dir, exist_ok=True)
    # Save prediction plots
    print(f"Creating prediction plots: '{pred_plot_dir}'")
    pfmt = os.path.join(pred_plot_dir, "{:03}.png")
    for i, p in enumerate(tqdm.tqdm(predictions, ncols=80)):
        plt.figure()
        plt.imshow(
            p,
            cmap=FT_CMAP,
            vmin=LABEL_FROZEN,
            vmax=max(LABEL_THAWED, LABEL_OTHER),
        )
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
        plot_predictions(dates, preds, os.path.join(root_dir, "pred_plots"))


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


def build_input_dataset_form_config(config, is_train):
    dem = np.load(config.dem_data_path)
    land_mask = torch.tensor(np.load(config.land_mask_path)).float()
    if is_train:
        return build_input_dataset(
            config,
            dem,
            land_mask,
            config.train_tb_data_path,
            config.train_date_map_path,
            config.train_solar_data_path,
            config.train_snow_data_path,
        )
    else:
        return build_input_dataset(
            config,
            dem,
            land_mask,
            config.test_tb_data_path,
            config.test_date_map_path,
            config.test_solar_data_path,
            config.test_snow_data_path,
        )


def build_input_dataset(
    config,
    dem,
    land_mask,
    tb_path,
    date_map_path,
    solar_path,
    snow_path,
):
    datasets = []
    tb_ds = np.load(tb_path)
    grid_shape = dem.shape
    if config.normalize:
        tb_ds = normalize(tb_ds)
    tb_ds = NpyDataset(tb_ds, channels=config.tb_channels)
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
        latitude_grid = torch.tensor(np.load(config.lat_grid_path)).float()
        if config.normalize:
            latitude_grid = normalize(latitude_grid)
        ds = RepeatDataset(latitude_grid, len(tb_ds))
        if config.use_prior_day:
            ds = Subset(ds, reduced_indices)
        datasets.append(ds)
    # Day of year channel
    if config.use_day_of_year:
        ds = build_day_of_year_ds(date_map_path, grid_shape, config)
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
    scaler,
    iterator,
    optimizer,
    land_mask,
    summary,
    epoch,
    config,
    is_train,
    confusion_matrix=None,
):
    loss_sum = 0.0
    for i, (input_data, labels, label_weights) in iterator:
        input_data = input_data.to(device, dtype=torch.float)
        labels = labels.to(device)
        label_weights = label_weights.to(device)

        if is_train:
            model.zero_grad()
        log_class_prob = model(input_data)
        class_prob = torch.sigmoid(log_class_prob)
        #
        # ERA/AWS
        #
        comb_loss = binary_cross_entropy_with_logits(
            log_class_prob, labels, label_weights
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item()

        if confusion_matrix is not None:
            flat_labels = labels.argmax(1)[..., land_mask].view(-1)
            flat_predictions = class_prob.argmax(1)[..., land_mask].view(-1)
            confusion_matrix.update(flat_labels, flat_predictions)
    loss_mean = loss_sum / len(iterator)
    summary.add_scalar("Loss", loss_mean, epoch)
    return loss_mean


def test(
    model,
    device,
    scaler,
    dataloader,
    optimizer,
    land_mask,
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
            scaler,
            it,
            optimizer,
            land_mask,
            summary,
            epoch,
            config,
            False,
            cm,
        )
    return loss, cm


def train(
    model,
    device,
    scaler,
    dataloader,
    optimizer,
    land_mask,
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
        scaler,
        it,
        optimizer,
        land_mask,
        summary,
        epoch,
        config,
        True,
    )


def build_full_dataset_from_config(config, land_mask, is_train):
    if is_train:
        aws_mask_path = config.train_aws_mask_path
        ft_label_data_path = config.train_ft_label_data_path
    else:
        aws_mask_path = config.test_aws_mask_path
        ft_label_data_path = config.test_ft_label_data_path
    input_ds = ArrayDataset(
        dataset_to_array(build_input_dataset_form_config(config, is_train))
    )
    # AWS
    aws_mask = np.load(aws_mask_path)
    if config.use_prior_day:
        aws_mask = aws_mask[1:]
    aws_mask = torch.tensor(aws_mask)
    if config.use_cold_constrained_weight_boost:
        cold_constrained_mask = torch.tensor(
            np.load(config.cold_constrained_mask_path)
        )
    ws = torch.zeros((len(input_ds), 1, *land_mask.shape), dtype=torch.float)
    ws[..., land_mask] = 1.0
    for i in tqdm.tqdm(range(len(ws)), ncols=80, desc="Weights"):
        ws[i, :, aws_mask[i]] *= config.aws_bce_weight
    if config.use_cold_constrained_weight_boost:
        ws[..., cold_constrained_mask] *= config.cold_constrained_weight_scale
        cold_constrained_mask = None
    weights_ds = ArrayDataset(ws)
    aws_mask = None
    # FT label
    label_ds = NpyDataset(ft_label_data_path)
    reduced_indices = list(range(1, len(input_ds) + 1))
    if config.use_prior_day:
        label_ds = Subset(label_ds, list(range(1, len(input_ds) + 1)))
    datasets = [input_ds, label_ds, weights_ds]
    if config.tile and is_train:
        datasets = [
            TilingDataset(d, land_mask.shape, config.tile_layout)
            for d in datasets
        ]
    ds = ComposedDataset(datasets)
    if not is_train:
        era_ds = Subset(
            NpyDataset(config.test_era5_ft_data_path), reduced_indices
        )
        return ds, input_ds, era_ds
    return ds


def main(config_path, resumable=False):
    config = load_config(config_path)
    device = torch.device("cuda:0")

    model = create_model(UNet, config)
    if torch.cuda.device_count() > 1:
        print("Using DataParallel")
        model = DataParallel(model)
    model = model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_reg_weight,
    )
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, config.lr_milestones, config.lr_step_gamma
    )
    grad_scaler = torch.cuda.amp.GradScaler()

    metric_checker = MetricImprovementChecker(
        MaxMetricTracker(-np.inf), MET_MCC
    )
    root_dir = config.run_dir
    snap_handler = SnapshotHandler(root_dir, model, opt, sched, metric_checker)
    resume = resumable and snap_handler.can_resume()
    if resume:
        print("Resuming")
    print(f"Initializing run dir: {root_dir}")
    train_summary, test_summary = init_run_dir(
        root_dir, config_path, resume=resume
    )

    last_epoch = 0
    if resume:
        (
            last_epoch,
            model,
            opt,
            sched,
            metric_checker,
        ) = snap_handler.load_full_snapshot()

    land_mask = torch.tensor(np.load(config.land_mask_path))
    #
    # Training data
    #
    train_ds = build_full_dataset_from_config(config, land_mask, True)
    #
    # Test Data
    #
    test_ds, test_input_ds, test_era_ds = build_full_dataset_from_config(
        config, land_mask, False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=config.drop_last,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    if not resume:
        snap_handler.take_model_snapshot()
    try:
        for epoch in range(last_epoch, config.epochs):
            train_summary.add_scalar(
                "learning_rate", next(iter(opt.param_groups))["lr"], epoch
            )
            train(
                model,
                device,
                grad_scaler,
                train_dataloader,
                opt,
                land_mask,
                train_summary,
                epoch,
                config,
            )
            loss, cm = test(
                model,
                device,
                grad_scaler,
                test_dataloader,
                opt,
                land_mask,
                test_summary,
                epoch,
                config,
            )
            if metric_checker.check(cm):
                snap_handler.take_model_snapshot()
            log_metrics(test_summary, cm, epoch)
            sched.step()
            if epoch % 3 == 0 and epoch != 0:
                snap_handler.take_full_snapshot(epoch)
    except KeyboardInterrupt:
        print("Exiting training loop")
    except Exception as e:
        print(f"\n{e}")
        raise e
    finally:
        train_summary.close()
        test_summary.close()
        # Free up data for GC
        train_ds = None
        train_dataloader = None

        # Validation
        val_dates = load_dates(config.test_date_map_path)
        if config.use_prior_day:
            val_dates = val_dates[1:]

        model = snap_handler.load_best_model()
        model.eval()
        # Create and save predictions for test data
        print("Generating predictions")
        test_loader = torch.utils.data.DataLoader(
            test_input_ds,
            batch_size=config.test_batch_size,
            shuffle=False,
            drop_last=False,
        )
        pred, raw_prob = get_predictions(
            test_loader, model, ~land_mask, LABEL_OTHER, device, config
        )
        predictions_path = os.path.join(root_dir, FNAME_PREDICTIONS)
        print(f"Saving predictions: '{predictions_path}'")
        np.save(predictions_path, pred)
        probabilities_path = os.path.join(root_dir, FNAME_PROBABILITIES)
        print(f"Saving probabilities: '{probabilities_path}'")
        np.save(probabilities_path, raw_prob)
        # Validate against ERA5
        print("Validating against ERA5")
        test_era_ds = dataset_to_array(test_era_ds).argmax(1).squeeze()
        era_acc = validate_against_era5(
            pred, test_era_ds, val_dates, land_mask
        )
        # Validate against AWS DB
        db = get_db_session(config.db_path)
        lon_grid = np.load(config.lon_grid_path)
        lat_grid = np.load(config.lat_grid_path)
        aws_acc = validate_against_aws(
            pred, db, val_dates, lon_grid, lat_grid, land_mask, config
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


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    config_path = args.config_path
    main(config_path, args.resumable)
