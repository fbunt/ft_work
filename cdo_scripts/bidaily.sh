parallel --verbose --ungroup --jobs 4 'foo={}; cdo -f nc4 -z zip_6 shifttime,30minute -timselmean,12 {} "${foo/hourly/bidaily}"' ::: *.nc
