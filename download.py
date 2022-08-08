import os
import sys
import numpy as np 
import xarray as xr
import pandas as pd

from pathlib import Path

init_dates = pd.date_range('2020-01-01','2021-01-01', freq = 'W-THU') # Initialization of both the hindcasts (20 yr) and forecasts

center = 'ecmwf'

forecasttypes = ['hindcast','forecast']

variables = {'Total Precipitation':'tp',
        'Two meter temperature':'t2m',
        'Sea surface temperature':'sst',
        'Total column water':'sst',
        'Soil moisture 20cm':'sm20',
        'Soil moisture 100cm':'sm100',
        }
#'Geopotential height':'gh',

def create_filename(forecasttype, init_date, center, varname):
    init_date = init_date.strftime('%Y%m%d')
    return f'{center}-{forecasttype}-{varname}-{init_date}.nc'

def create_url(forecasttype, init_date, center, varname):
    filename = create_filename(forecasttype = forecasttype, init_date = init_date, center = center, varname = varname)
    url = 'https://storage.ecmwf.europeanweather.cloud/s2s-ai-challenge/data/'
    if forecasttype == 'hindcast':
        url += 'training-input/'
    else:
        url += 'test-input/'
    url += f'0.3.0/netcdf/{filename}'
    return url

def create_local_path(forecasttype, init_date, center, varname):
    filename = create_filename(forecasttype = forecasttype, init_date = init_date, center = center, varname = varname)
    path = Path('/scistor/ivm/data_catalogue/climate_models/s2s/')
    return path / center / forecasttype / filename

for forecasttype in []: #forecasttypes:
    for init_date in init_dates:
        for varname in variables.values():
            serverurl = create_url(forecasttype = forecasttype, init_date = init_date, center = center, varname = varname)
            localfile = create_local_path(forecasttype = forecasttype, init_date = init_date, center = center, varname = varname)
            # Procedure for forecasts
            if forecasttype == 'forecast':
                if localfile.exists():
                    print(f'{localfile} already exists, doing nothing')
                else:
                    os.system(f'curl {serverurl} -o {localfile}')
            else:
            # For hindcast we are creating multiple local files. as a single downloaded one contains 20 years.
                localdates = [pd.Timestamp(year = init_date.year - y, month = init_date.month, day = init_date.day) for y in range(1,21)]
                localfiles = [create_local_path(forecasttype = forecasttype, init_date = date, center = center, varname = varname) for date in localdates] 
                if np.all([l.exists() for l in localfiles]):
                    print(f'{localfiles} already exist, doing nothing')
                else:
                    os.system(f'curl {serverurl} -o {localfile}')
                    hinc = xr.open_dataarray(localfile, decode_times = True)
                    for date, path in zip(localdates, localfiles):
                        test = hinc.sel(forecast_time = [date])
                        test.to_netcdf(path)
                    hinc.close()
                    localfile.unlink() # Removes the 20 yr file
                    
