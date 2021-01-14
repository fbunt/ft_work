#!/usr/bin/env bash
ls ./*.grib |
grep -oP ".*(?=.grib)" |
awk '{if(system("[ ! -f " $1 ".nc ]") == 0) {print $1}}' |
parallel --jobs 4 --ungroup --verbose 'cdo -f nc4 -z zip_6 copy {}.grib {}.nc'
