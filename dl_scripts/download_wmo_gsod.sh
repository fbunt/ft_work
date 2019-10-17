#!/usr/bin/env bash
for y in $(seq 1979 2018); do
    echo "Downloading ${y}..."
    if [ ! -d "${y}" ]; then
        mkdir "${y}"
        wget "ftp://ftp.ncdc.noaa.gov/pub/data/gsod/${y}/gsod_${y}.tar"
        tar -xf "gsod_${y}.tar" -C "${y}"
        rm "gsod_${y}.tar"
        sleep 1
        echo "done"
    else
        echo "${y} already present. Skipping"
    fi
done
