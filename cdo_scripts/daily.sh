#!/usr/bin/env bash
ls *.nc | parallel --jobs 8 --ungroup --verbose 'cdo -f nc4 -z zip_6 daymean {} {.}-daily.nc'
