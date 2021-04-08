#!/usr/bin/env bash
# Train
python prep_data.py -t -e nh AM 2005 2014 "../data/cleaned"
# Test
python prep_data.py -t -e nh AM 2016 2016 "../data/cleaned"
