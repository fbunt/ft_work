#!/usr/bin/env bash
wget -r -np -nH -w 0.2 -N --cut-dirs=4 -R "*.html*" \
    http://files.ntsg.umt.edu/data/FT_ESDR/DAILY_GEOTIFF/SSMI
