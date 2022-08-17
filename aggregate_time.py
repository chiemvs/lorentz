# Pre-computation of aggregated variables 
# Not rolling but single,fixed leadtime (week 3,4,5,6)
# tp: accumulation,
# gh, tcw, t2m, sm20, sm100, sst: mean 

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

from pathlib import Path

variables = ['gh', 'tcw', 't2m', 'sm20', 'sm100', 'sst','tp'] 

rawpath = Path('/data/volume_2/subseasonal/ecmwf/raw')
aggpath = Path('/data/volume_2/subseasonal/ecmwf/aggregated')

timestamp_frame = pd.read_hdf(aggpath / 'aggregation_timestamps.h5') # From init_valid_dates.py, leadtime = 14, 4-week aggregation

def load_and_agg_variable(var, filepath):
    array = xr.open_dataarray(filepath)
    initdate = pd.Timestamp(filepath.parts[-1].split('-')[-1][:8])
    start_inclusive = timestamp_frame.loc[initdate,'aggregation_start_inclusive']
    end_inclusive = timestamp_frame.loc[initdate,'aggregation_end_inclusive']
    selection = array.squeeze().swap_dims({'lead_time':'valid_time'}).sel(valid_time = slice(start_inclusive, end_inclusive))
    if var == 'tp': # deaccumulation, last minus first.
        last = selection.isel(valid_time = -1)
        first = selection.isel(valid_time = 0)
        aggregated = last - first 
    else:
        aggregated = selection.mean('valid_time')
    aggregated = aggregated.drop('forecast_time').expand_dims({'valid_time':[start_inclusive]}) # Timestamp the aggregation period with the first day
    array.close()
    return aggregated

for var in variables:
    for forctype in ['hindcast','forecast']:
        outpath = aggpath / forctype / f'ecmwf-{forctype}-{var}-week3456.nc'
        if not outpath.exists():
            filelist = list((rawpath / forctype).glob(f'*-{var}-*'))
            filelist.sort()
            init_times = [ pd.Timestamp(p.parts[-1].split('-')[-1][:8]) for p in filelist]
            example = xr.open_dataarray(filelist[0]).load()
            example.close()
            combined = xr.DataArray(np.full(shape = (len(filelist),example.shape[0])+ example.shape[-2:], fill_value = np.nan, dtype = np.float32), dims = ('valid_time','realization','latitude','longitude'))
            for key in ['realization','latitude','longitude']:
                combined.coords.update({key:example.coords[key]})
            combined.coords.update({'valid_time':timestamp_frame.loc[init_times,'aggregation_start_inclusive'].values}) # First days of the aggregation period
            combined.attrs = example.attrs
            for key in ['dtype','_FillValue','scale_factor','add_offset']:
                combined.encoding.update({key:example.encoding[key]}) # For more compressed storage on-disk
            del example # Cleaning up, low memory environment
            for filepath in filelist:
                aggregated = load_and_agg_variable(var = var, filepath = filepath)
                combined.loc[aggregated.valid_time,...] = aggregated
                print(f'aggregated: {filepath}')
            del aggregated
            combined.to_netcdf(outpath)
        else:
            print(f'aggregated {outpath} already exists')

# Target aggregation
chirps = xr.open_dataarray('/data/volume_2/observational/raw/chirps_tp_1980-2021_daily_0.25deg_africa.nc')
outpath = Path('/data/volume_2/observational/preprocessed/chirps_tp_2000-2020_4weekly_0.25deg_africa.nc')

if not outpath.exists():
    aggarray = xr.DataArray(np.full(shape = (len(timestamp_frame),) + chirps.shape[1:], fill_value = np.nan, dtype = np.float32), dims = ('valid_time','latitude','longitude'))
    for key in ['latitude','longitude']:
        aggarray.coords.update({key:chirps.coords[key]})
    aggarray.coords.update({'valid_time':timestamp_frame.loc[:,'aggregation_start_inclusive'].sort_values().values}) # First days of the aggregation period
    aggarray.attrs = chirps.attrs
    aggarray.attrs.update({'units':'mm','time_step':'4_week_accumulation'}) # Original daily units are mm/day, so we can just sum over 4 weeks.
    for i in range(len(timestamp_frame)):
        start_inclusive, end_inclusive = timestamp_frame[['aggregation_start_inclusive','aggregation_end_inclusive']].iloc[i]
        selection = chirps.sel(time = slice(start_inclusive, end_inclusive))
        assert selection.shape[0] == 28, 'length of time period should be 28, check if daily chirps contains all aggregation stamps.'
        aggarray.loc[start_inclusive,...] = selection.sum('time')
        print(f'aggregated from {start_inclusive.strftime("%Y-%m-%d")}, to {end_inclusive.strftime("%Y-%m-%d")} ')
    aggarray.to_netcdf(outpath)
