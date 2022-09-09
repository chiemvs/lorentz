import os
import sys
import numpy as np 
import xarray as xr
import pandas as pd

from pathlib import Path

init_dates = pd.date_range('2020-01-01','2021-01-01', freq = 'W-THU') # Initialization of both the hindcasts (20 yr) and forecasts

center = 'ecmwf'

forecasttypes = ['hindcast','forecast']

#variables = {'Total Precipitation':'tp',
#        'Two meter temperature':'t2m',
#        'Sea surface temperature':'sst',
#        }
#variables = {'Total column water':'tcw',
#        'Soil moisture 20cm':'sm20',
#        'Soil moisture 100cm':'sm100',
#        }
variables = {'Accumulated TOA thermal rad':'ttr'}

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
    #path = Path('/scistor/ivm/data_catalogue/climate_models/s2s/')
    path = Path('/data/volume_2/subseasonal/')
    return path / center / 'raw' / forecasttype / filename

def download(serverurl, localfile):
    if not localfile.exists():
        print('downloading: ', localfile)
        os.system(f'curl {serverurl} -o {localfile}')

def sellevel(localfile, varname, opened_array = None):
    """
    extracting a single pressure level for the multilevel variables, only if multilevel
    If loaded array is supplied, just return the subset
    If not supplied  then load the array here, select and overwrite
    """
    multilevel = {'gh':500} # Level in hPa
    if varname in multilevel.keys():
        if opened_array is None:
            array = xr.open_dataarray(localfile)
            print('multilevel selection: YES')
        else:
            array = opened_array
            print('multilevel selection: YES, file already open')
        single = array.sel(plev = multilevel[varname])
        if opened_array is None: 
            single.load() # Into memory
            array.close()
            localfile.unlink() # Remove old multilevel
            single.to_netcdf(localfile) # Replace by writing the the data in memory
        else:
            return single
    else:
        print('multilevel selection: NO')
        return opened_array


for forecasttype in forecasttypes:
    for init_date in init_dates:
        for varname in variables.values():
            serverurl = create_url(forecasttype = forecasttype, init_date = init_date, center = center, varname = varname)
            localfile = create_local_path(forecasttype = forecasttype, init_date = init_date, center = center, varname = varname)
            # Procedure for forecasts
            if forecasttype == 'forecast':
                if localfile.exists():
                    print(f'{localfile} already exists, doing nothing')
                else:
                    download(serverurl = serverurl, localfile = localfile)
                    sellevel(localfile = localfile, varname = varname)
            else:
            # For hindcast we are creating multiple local files. as a single downloaded one contains 20 years.
                localdates = [pd.Timestamp(year = init_date.year - y, month = init_date.month, day = init_date.day) for y in range(1,21)]
                localfiles = [create_local_path(forecasttype = forecasttype, init_date = date, center = center, varname = varname) for date in localdates] 
                if np.all([l.exists() for l in localfiles]):
                    print(f'{localfiles} already exist, doing nothing')
                else:
                    download(serverurl = serverurl, localfile = localfile)
                    hinc = xr.open_dataarray(localfile, decode_times = True)
                    for date, path in zip(localdates, localfiles):
                        test = hinc.sel(forecast_time = [date])
                        test = sellevel(localfile = localfile, varname = varname, opened_array = test)
                        test.to_netcdf(path)
                    hinc.close()
                    localfile.unlink() # Removes the 20 yr file
