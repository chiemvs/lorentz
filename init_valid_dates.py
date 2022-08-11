import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

basepath = Path('/data/volume_2/subseasonal/ecmwf')

dataframes = []
for forctype in ['hindcast','forecast']:
    init_paths = list((basepath / forctype).glob('*-t2m-*')) # Get a complete file list by searching for those of a single variable
    init_dates = [pd.Timestamp(p.parts[-1].split('-')[-1][:8]) for p in init_paths]
    df = pd.DataFrame({'type':forctype}, index = pd.Index(init_dates,name = 'init_date'))
    df['aggregation_start_inclusive'] = df.index + pd.Timedelta(15,'d') # 15th simulated day is the start, so 14 days (2 week) lead time.
    df['aggregation_end_inclusive'] = df['aggregation_start_inclusive'] + pd.Timedelta(27,'d') # Because the end is inclusive this will amount to 28 values: week 3 + 4 + 5 + 6
    df['path'] = [str(p) for p in init_paths]
    dataframes.append(df)

testfile = xr.open_dataarray(init_paths[0], decode_times = True)
selection = testfile.squeeze().swap_dims({'lead_time':'valid_time'}).sel(valid_time = slice(df.loc[init_dates[0],'aggregation_start_inclusive'],df.loc[init_dates[0],'aggregation_end_inclusive']))
print('testfile:')
print(selection.dims,' : ',selection.shape)

dataframes = pd.concat(dataframes, axis = 0)
dataframes.to_hdf(basepath / 'aggregated' / 'aggregation_timestamps.h5', key = 'aggregation_timestamps', mode = 'w')

