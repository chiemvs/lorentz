import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# All functions and data concern a single lead time, and a single 4-week time aggregation (pre-aggregated)
# Only the observations/target is without leadtime
# Shared time axis is 'valid_time'

# Computation of ensemble mean
# Removal of seasonality (for any set, but based on training, not on test set)
# Selection of the right season
# Standardization (spatial / per gridcell?)
# patch size determined, possibility to grab only the target region, or get patches from everywhere (more samples, perhaps local standardization needed)
# Output TfRecords? At least split between forecast and hindcast, perhaps also separately validation (because we don't want non-target region patches there).


def remove_seasonality(array, expectation = None):
    """
    Monthly values assumed to have little sampling variability, normal is defined 2000-2019, namely the hindcast period 
    To deasonalize the 
    """
    if not ('month' in array.coords):
        array.coords['month'] = ('valid_time', array.coords['valid_time'].dt.month.values)
    grouped = array.groupby('month')
    if expectation is None:
        expectation = grouped.mean('valid_time')
        if 'realization' in expectation.dims:
            print('computing seasonality of individual members, members treated as extra samples for the climate')
            expectation = expectation.mean('realization')
        else:
            print('computing seasonality of ensemble mean')
    else:
        print('seasonal expectation is presupplied')
    anomalies = array - expectation.loc[array.coords['month'],...] # xarray does alignment
    return anomalies, expectation

def standardize_array(array, spatially = True, temporally = False, trained_scaler = None):
    """
    If large patches, spatial standardization makes sense?
    Possible to pre_supply the scaler
    Scalers work on (nsamples, nfeatures), where each feature is standardized separately
    Realization / members are always treated as samples, if present in the array. (otherwise a scaler trained on hindcasts (11 members) would not be applicable to forecasts (51 members))
    """
    assert (spatially or temporally) and (not(spatially and temporally)), 'choose one of spatial or temporal standardization'
    print(f'scaling is {"spatial" if spatially else "temporal"}')
    feature_dims = list(array.dims)
    if spatially:
        feature_dims.remove('latitude')
        feature_dims.remove('longitude')
    else:
        feature_dims.remove('valid_time')
    if 'realization' in feature_dims:
        feature_dims.remove('realization')
    sample_dims = [d for d in array.dims if (not d in feature_dims)]
    stacked = array.stack({'samples':sample_dims}).stack({'features':feature_dims})
    if trained_scaler is None:
        print(f'computing standard scaler on {stacked.shape}, members as samples = {"realization" in array.dims}')
        trained_scaler = StandardScaler()
        trained_scaler.fit(stacked) # (nsamples,nfeatures)
    else:
        print(f'using pre-trained standard scaler on {stacked.shape}, members as samples = {"realization" in array.dims}')
    stacked.values = trained_scaler.transform(stacked) # Actual transformation
    return stacked.unstack('samples').unstack('features'), trained_scaler

def select_centered_patch(array, patchsize: tuple = (40,40)):
    """
    Patchsize in degrees (nlon,nlat)
    """
    center_HoA = {'latitude':4.511,'longitude':40.496}
    latslice = slice(center_HoA['latitude'] + patchsize[1]/2, center_HoA['latitude'] - patchsize[1]/2) # Latitude is stored descending (90:-90)
    lonslice = slice(center_HoA['longitude'] - patchsize[0]/2, center_HoA['longitude'] + patchsize[0]/2) # Longitude is stored ascending (0:360)
    print(f'attempt patch selection lat:{latslice}, lon:{lonslice}')
    array = array.sel(latitude = latslice, longitude = lonslice)
    return array

def preprocess_main(var: str = var, rm_season: bool = True, ensmean: bool = False, standardize_space: bool = False, standardize_time: bool = False, fixed_patch: bool = True, patchsize: tuple = (40,40) ):
    """
    Patchsize in degrees (nlon,nlat), if fixed_patch, this is centered over the Horn of Africa
    Preprocessing should prevent data leakage from hindcasts to forecasts.
    """
    assert fixed_patch, 'currently only a fixed patch is supported' # To support variable patches, the order needs to be changed, e.g. rm season for all gridcells, later spatial subsetting
    datadir = Path( '/data/volume_2/subseasonal/ecmwf/aggregated/')
    var = 'tcw' 
    hindcast = xr.open_dataarray(datadir / 'hindcast' / f'ecmwf-hindcast-{var}-week3456.nc')
    forecast = xr.open_dataarray(datadir / 'forecast' / f'ecmwf-forecast-{var}-week3456.nc') 

    # Patch selection, currently one of the early steps to limit memory usage
    hindcast = select_centered_patch(hindcast, patchsize = patchsize)
    forecast = select_centered_patch(forecast, patchsize = patchsize)

    if rm_season:
        hindcast, exp = remove_seasonality(hindcast)
        forecast, _ = remove_seasonality(forecast, expectation = exp) # Will form the test set, so not used for seasonal expectations

    if ensmean:
        hindcast = hindcast.mean('realization')
        forecast = forecast.mean('realization')

    if standardize_space or standardize_time: # Standardizing spatially means that each valid time is its own feature, therefore no leakage from hindcast to forecast, and no supply of pretrained_scaler
        hindcast, trained_scaler = standardize_array(array = hindcast, spatially = standardize_space, temporally = standardize_time)
        forecast, _ = standardize_array(array = forecast, spatially = standardize_space, temporally = standardize_time, trained_scaler = None if standardize_space else trained_scaler)
        
    return hindcast, forecast

varlist = ['tp','sst','tcw']
for var in varlist[-1:]:
    if var == 'tp': # No seasonal anomalies
        hindcast, forecast = preprocess_main(var = var, rm_season = False, ensmean = False, standardize_space = True) # Perhaps rainfall should be min-max scaled, such that zero is really zero?
    else:
        hindcast, forecast = preprocess_main(var = var, rm_season = True, ensmean = False, standardize_space = True)

