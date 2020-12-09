import argparse
import numpy as np
import os
import torch

import utils
from model import LABEL_OTHER, UNet, UNetLegacy
from training import (
    build_input_dataset_form_config,
    get_predictions,
    load_config,
)
from transforms import REGION_TO_TRANS


def get_cli_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "target_dir",
        type=utils.validate_dir_path,
        help="The target directory to add probabilities to.",
    )
    p.add_argument(
        "-c",
        "--config",
        type=utils.validate_file_path,
        default=None,
        help=(
            "Path to alternate config file. Default is to use the config file "
            "in the target directory."
        ),
    )
    p.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=-1,
        help="Override config batch size",
    )
    p.add_argument(
        "-p",
        "--save_predictions",
        action="store_true",
        help="Causes FT predictions to be saved, if set",
    )
    p.add_argument(
        "-l",
        "--legacy_model",
        action="store_true",
        help="Use legacy UNet model",
    )
    return p


def model_load(model, state):
    sample_key = next(iter(state.keys()))
    input_module_prefixed = sample_key.startswith("module")
    sample_key = next(iter(model.state_dict().keys()))
    model_module_prefixed = sample_key.startswith("module")
    if model_module_prefixed:
        if input_module_prefixed:
            model.load_state_dict(state)
        else:
            new_dict = {}
            for k, v in state.items():
                k = "module." + k
                new_dict[k] = v
            model.load_state_dict(new_dict)
    else:
        if input_module_prefixed:
            new_dict = {}
            for k, v in state.items():
                # Remove leading "module." prefix
                k = k.partition("module.")[2]
                new_dict[k] = v
            model.load_state_dict(new_dict)
        else:
            model.load_state_dict(state)
    return model


def main(args):
    cfile = os.path.join(args.target_dir, "config.yaml")
    if args.config is not None:
        # Already validated by parser
        cfile = args.config
    else:
        try:
            utils.validate_file_path(cfile)
        except IOError:
            cfile = os.path.join(args.target_dir, "config")
            utils.validate_file_path(cfile)
    config = load_config(cfile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = REGION_TO_TRANS[config.region]
    model_path = os.path.join(args.target_dir, "model.pt")
    utils.validate_file_path(model_path)
    model_class = UNet
    if args.legacy_model:
        print("Using legacy model")
        model_class = UNetLegacy
    model = model_class(
        in_chan=config.in_chan,
        n_classes=config.n_classes,
        depth=config.depth,
        base_filter_bank_size=config.base_filters,
        skip=config.skips,
        bndry_dropout=config.bndry_dropout,
        bndry_dropout_p=config.bndry_dropout_p,
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model_dict = torch.load(model_path)
    model_load(model, model_dict)
    model.eval()
    input_ds = build_input_dataset_form_config(config, is_train=False)
    batch_size = args.batch_size if args.batch_size > 0 else config.batch_size
    input_dl = torch.utils.data.DataLoader(
        input_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    preds, probs = get_predictions(
        input_dl,
        model,
        transform(np.load("../data/masks/ft_esdr_water_mask.npy")),
        LABEL_OTHER,
        device,
        config,
    )
    # TODO
    pred_path = os.path.join(args.target_dir, "pred.npy")
    prob_path = os.path.join(args.target_dir, "prob.npy")
    print(f"Saving probabilities: '{prob_path}'")
    np.save(prob_path, probs)
    if args.save_predictions:
        print(f"Saving predictions: '{pred_path}'")
        np.save(preds, pred_path)


if __name__ == "__main__":
    args = get_cli_parser().parse_args()
    main(args)
