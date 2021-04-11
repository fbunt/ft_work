from collections import namedtuple
import yaml

ConfigV1 = namedtuple(
    "ConfigV1",
    (
        "version",
        "in_chan",
        "n_classes",
        "depth",
        "base_filters",
        "skips",
        "bndry_dropout",
        "bndry_dropout_p",
        "tb_channels",
        "use_land_mask",
        "use_dem",
        "use_latitude",
        "use_day_of_year",
        "use_solar",
        "use_snow",
        "use_prior_day",
        "normalize",
        "region",
        "train_start_year",
        "train_end_year",
        "test_start_year",
        "test_end_year",
        "epochs",
        "batch_size",
        "drop_last",
        "learning_rate",
        "lr_milestones",
        "lr_step_gamma",
        "l2_reg_weight",
        "main_loss_weight",
        "aws_bce_weight",
        "lv_reg_weight",
        "aws_use_valid_mask",
        "val_use_valid_mask",
        "do_val_plots",
        "do_pred_plots",
    ),
)
ConfigV2Plus = namedtuple(
    "ConfigV2Plus",
    (
        "version",
        "in_chan",
        "n_classes",
        "depth",
        "base_filters",
        "skips",
        "bndry_dropout",
        "bndry_dropout_p",
        "tb_channels",
        "use_land_mask",
        "use_dem",
        "use_latitude",
        "use_day_of_year",
        "use_solar",
        "use_snow",
        "use_prior_day",
        "normalize",
        "tile",  # v5
        "tile_layout",  # v7
        "region",
        "train_start_year",
        "train_end_year",
        "test_start_year",
        "test_end_year",
        "epochs",
        "train_batch_size",  # v5
        "test_batch_size",  # v5
        "am_pm",  # v6
        "drop_last",
        "learning_rate",
        "lr_milestones",
        "lr_step_gamma",
        "l2_reg_weight",
        "main_loss_weight",
        "aws_bce_weight",
        "lv_reg_weight",
        "aws_use_valid_mask",
        "val_use_valid_mask",
        "do_val_plots",
        "do_pred_plots",
        # Paths
        "runs_dir",  # v3
        "db_path",  # v3
        "land_mask_path",
        "dem_data_path",
        "lon_grid_path",
        "lat_grid_path",
        "train_aws_data_path",
        "train_aws_mask_path",  # v4
        "train_date_map_path",
        "train_tb_data_path",
        "train_era5_ft_data_path",
        "train_ft_label_data_path",  # v4
        "train_solar_data_path",
        "train_snow_data_path",
        "test_aws_data_path",
        "test_aws_mask_path",  # v4
        "test_date_map_path",
        "test_tb_data_path",
        "test_era5_ft_data_path",
        "test_ft_label_data_path",  # v4
        "test_solar_data_path",
        "test_snow_data_path",
    ),
)


def build_in_chan(cfg):
    cfg["in_chan"] = (
        len(cfg["tb_channels"])
        + cfg["use_dem"]
        + cfg["use_latitude"]
        + cfg["use_day_of_year"]
        + cfg["use_solar"]
        + cfg["use_snow"]
        + (len(cfg["tb_channels"]) * cfg["use_prior_day"])
    )
    return cfg


def build_v1_config(cfg):
    if "version" not in cfg:
        cfg["version"] = 1
    cfg = build_in_chan(cfg)
    return ConfigV1(**cfg)


def build_v2plus_config(cfg):
    cfg = build_in_chan(cfg)
    cfg["runs_dir"] = cfg.get("runs_dir", "../runs")
    cfg["db_path"] = cfg.get("db_path", "../data/dbs/wmo_gsod.db")
    # Handle v2-3 case where ft_label and aws_mask are not used
    cfg["train_ft_label_data_path"] = cfg.get("train_ft_label_data_path", None)
    cfg["test_ft_label_data_path"] = cfg.get("test_ft_label_data_path", None)
    cfg["train_aws_mask_path"] = cfg.get("train_aws_mask_path", None)
    cfg["test_aws_mask_path"] = cfg.get("test_aws_mask_path", None)
    # Handle addition of tile key in v5
    cfg["tile"] = cfg.get("tile", False)
    # Handle splitting of batch size into train and test inv v5
    if cfg["version"] < 5:
        bs = cfg.pop("batch_size")
        cfg["train_batch_size"] = bs
        cfg["test_batch_size"] = bs
    # Handle v6 am_pm field
    cfg["am_pm"] = cfg.get("am_pm", "AM")
    # Handle v7 tile_layout field
    cfg["tile_layout"] = tuple(cfg.get("tile_layout", (1, 3)))
    return ConfigV2Plus(**cfg)


class ConfigError(Exception):
    pass


def load_config(config_path):
    with open(config_path) as fd:
        cfg = yaml.safe_load(fd)["config"]
    version = cfg.get("version", 1)
    if version == 1:
        return build_v1_config(cfg)
    elif version >= 2:
        return build_v2plus_config(cfg)
    else:
        raise ConfigError("Invalid verion found in config file")


def create_model(model_class, config):
    model = model_class(
        config.in_chan,
        config.n_classes,
        depth=config.depth,
        base_filter_bank_size=config.base_filters,
        skip=config.skips,
        bndry_dropout=config.bndry_dropout,
        bndry_dropout_p=config.bndry_dropout_p,
    )
    return model
