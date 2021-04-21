import argparse
import datetime as dt
import numpy as np
import torch
import tqdm

import datahandling as dh
import utils
from config import load_config, create_model
from model import UNet
from transforms import REGION_CODES


def validate_region(reg):
    if reg in REGION_CODES:
        return reg
    else:
        raise ValueError(f"Unknown region code: '{reg}'")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="Override config's batch size",
    )
    p.add_argument(
        "-p", "--parallel", action="store_true", help="Use all available GPUs"
    )
    ropts = (("{}, " * (len(REGION_CODES) - 1)) + "{}").format(
        *sorted(REGION_CODES)
    )
    p.add_argument(
        "region",
        type=validate_region,
        help=f"The region to pull data from. Options are: {ropts}",
    )
    p.add_argument("am_pm", type=str, help="AM or PM data")
    p.add_argument(
        "start_year", type=int, help="Start year of data to run through model"
    )
    p.add_argument(
        "end_year", type=int, help="End year of data to run through model"
    )
    p.add_argument(
        "config_path",
        type=utils.validate_file_path,
        help="Path to config file",
    )
    p.add_argument(
        "model_path",
        type=utils.validate_file_path,
        help="Path to saved model weights",
    )
    p.add_argument(
        "output_path_prefix",
        type=str,
        help=(
            "Output prefix path. Will be used for FT predictions and"
            " probabilities files."
        ),
    )
    return p


def get_predictions(input_dl, model, water_mask, water_label, device):
    pred = np.zeros((len(input_dl.dataset), *water_mask.shape), dtype=np.int8)
    prob = np.zeros((len(input_dl.dataset), *water_mask.shape))
    j = 0
    with torch.no_grad():
        for v in tqdm.tqdm(input_dl, ncols=80, total=len(input_dl)):
            output = model(v.to(device))
            p = torch.sigmoid(output).cpu().numpy()
            # Transform the class channels into a Bernoulli distribution. This
            # is necessary because each channel is 0 <= chan <= 1 but their sum
            # is 0 < sum < 2. Then take only the thawed channel to compress the
            # data.
            prob_batch = [(x / x.sum(0))[1] for x in p]
            pred_batch = p.argmax(1)
            pred_batch[..., water_mask] = water_label
            n = len(prob_batch)
            prob[j : j + n] = prob_batch
            pred[j : j + n] = pred_batch
            j += n
    return pred, prob


def main(
    region,
    am_pm,
    start_year,
    end_year,
    config_path,
    model_path,
    output_path_prefix,
    batch_size,
    parallel,
):
    config = load_config(config_path)
    batch_size = batch_size or config.test_batch_size
    device = torch.device("cuda:0")
    model = create_model(UNet, config)
    if parallel:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    n = 0
    tbdss = []
    for y in range(start_year, end_year + 1):
        fname = (
            f"../data/tb/gapfilled_{region}/tb_{y}_{am_pm}_{region}_filled.npy"
        )
        ni = (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days
        ds = dh.LazyLoadFastUnloadNpyDataset(fname, ni)
        tbdss.append(ds)
        n += ni
    tbdss = torch.utils.data.ConcatDataset(tbdss)
    zds = dh.RepeatDataset(np.load(config.dem_data_path), n)
    ds = dh.GridsStackDataset([zds, tbdss])
    if config.use_prior_day:
        ds = torch.utils.data.Subset(ds, list(range(1, n)))
    dloader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    water_mask = np.load(config.land_mask_path)
    pred, prob = get_predictions(dloader, model, water_mask, device)
    np.save(f"{output_path_prefix}_pred.npy", pred)
    np.save(f"{output_path_prefix}_prob.npy", pred)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
