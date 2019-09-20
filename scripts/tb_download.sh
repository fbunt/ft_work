#!/usr/bin/env bash
echo "Downloading SMMR"
python tb_download.py "sidads.colorado.edu" "pub/DATASETS/nsidc0071_smmr_ease_grid_tbs/global" "$1"

echo "Downloading SSMI"
python tb_download.py "sidads.colorado.edu" "pub/DATASETS/nsidc0032_ease_grid_tbs/global" "$2"
