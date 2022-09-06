import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# All functions and data concern a single lead time, and a single 4-week time aggregation (pre-aggregated)
# Only the observations/target is without leadtime
# Shared time axis between obs and forecast is 'valid_time'

# Options:
# Computation of ensemble mean 
# Removal of seasonality (for any set, but based on training/hindcast, not on test/forecast set)
# Standardization (spatial / per gridcell)
# patch size determined, possibility to grab patch centered on the target region

# Future TODO:
# Selection of the right season
# Output TfRecords?


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
    print(f'standardization is {"spatial" if spatially else "temporal"}')
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

def preprocess_ecmwf(var: str, rm_season: bool = True, ensmean: bool = False, standardize_space: bool = False, standardize_time: bool = False, fixed_patch: bool = True, patchsize: tuple = (40,40) ):
    """
    Patchsize in degrees (nlon,nlat), if fixed_patch, this is centered over the Horn of Africa
    Preprocessing should prevent data leakage from hindcasts to forecasts.
    """
    assert fixed_patch, 'currently only a fixed patch is supported' # To support variable patches, the order needs to be changed, e.g. rm season for all gridcells, later spatial subsetting
    datadir = Path( '/data/volume_2/subseasonal/ecmwf/aggregated/')
    #var = 'tcw'
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

def spatial_average_in_mask(array, maskname):
    """Finds masks in the observational directory """
    datapath = Path('/data/volume_2/observational/')
    mask = xr.open_dataarray(datapath / maskname)
    mask = mask.reindex_like(array, method = 'nearest') # Same resolution but slightly different lats and lons
    mask.name = 'mask' # True or False
    spatial_avg = array.groupby(mask).mean().sel(mask = True)
    return spatial_avg

def preprocess_target(maskname = 'era5_hoa_dry_mask_0.25deg.nc', quantile_edges = [0.33,0.66], return_edges: bool = False):
    """
    Currently not a target patch, just a scalar through spatial aggregation
    This scaler is classified according its tercile edges (estimated per month)
    """
    datapath = Path('/data/volume_2/observational/')
    chirps = xr.open_dataarray(datapath / 'preprocessed' / 'chirps_tp_2000-2020_4weekly_0.25deg_africa.nc')
    spatial_avg = spatial_average_in_mask(chirps, maskname = maskname)
    spatial_avg.coords['month'] = ('valid_time', spatial_avg.coords['valid_time'].dt.month.values)

    hindcast_forecast_split = pd.Timestamp('2020-01-16') # In terms of valid_time
    target_hindcast = spatial_avg.sel(valid_time = slice(None,hindcast_forecast_split))
    target_forecast = spatial_avg.sel(valid_time = slice(hindcast_forecast_split,None))
    
    tercile_edges = target_hindcast.groupby('month').quantile(quantile_edges) # Determined on hindcast (to prevent leakage to forecast), also per month to account for seasonality 

    def lookup_month_and_digitize(array, edges):
        """ 
        function to be mapped, returns an array with the same
        dimensions, only values replaced by integers.
        for instance if tercile edges are supplied:
        2 = upper, 1 = middle, 0 = lower tercile
        """
        month = int(np.unique(array.month.values))
        edges_month = edges.loc[month].values
        binned = np.digitize(x = array, bins = edges_month)
        return xr.DataArray(binned, dims = array.dims, coords = array.coords)

    target_hindcast_binned = target_hindcast.groupby('month').map(lookup_month_and_digitize, edges = tercile_edges)
    target_forecast_binned = target_forecast.groupby('month').map(lookup_month_and_digitize, edges = tercile_edges)

    # One hot encoding of the classes
    nclasses = len(quantile_edges) + 1
    target_hindcast_encoded = xr.DataArray(tf.one_hot(target_hindcast_binned, depth = nclasses), dims = target_hindcast_binned.dims + ('classes',), coords = target_hindcast_binned.coords) 
    target_hindcast_encoded.coords['classes'] = ('classes',pd.RangeIndex(nclasses))
    target_forecast_encoded = xr.DataArray(tf.one_hot(target_forecast_binned, depth = nclasses), dims = target_forecast_binned.dims + ('classes',), coords = target_forecast_binned.coords) 
    target_forecast_encoded.coords['classes'] = ('classes',pd.RangeIndex(nclasses))

    if return_edges:
        return target_hindcast_encoded, target_forecast_encoded, tercile_edges
    else:
        return target_hindcast_encoded, target_forecast_encoded

def preprocess_raw_forecasts(maskname = 'era5_hoa_dry_mask_0.25deg.nc', quantile_edges = [0.33,0.66], return_edges : bool = False):
    """
    Extracting the raw probability forecasts for our categorical precipitation target 
    not used as inputs for postprocessing, but instead as a benchmark
    edge estimation based on hindcast, and per month, all members used for estimating probability of a certain class
    """
    datadir = Path( '/data/volume_2/subseasonal/ecmwf/aggregated/')
    var = 'tcw' 
    hindcast = xr.open_dataarray(datadir / 'hindcast' / f'ecmwf-hindcast-{var}-week3456.nc')
    forecast = xr.open_dataarray(datadir / 'forecast' / f'ecmwf-forecast-{var}-week3456.nc') 

    hindcast_avg = spatial_average_in_mask(hindcast, maskname = maskname) 
    hindcast_avg.coords['month'] = ('valid_time', hindcast_avg.coords['valid_time'].dt.month.values) # Adding these coordinates because of per-month estimation
    forecast_avg = spatial_average_in_mask(forecast, maskname = maskname) 
    forecast_avg.coords['month'] = ('valid_time', forecast_avg.coords['valid_time'].dt.month.values)

    nclasses = len(quantile_edges) + 1
    tercile_edges = hindcast_avg.groupby('month').quantile(quantile_edges, dim = ['valid_time','realization']) # Determined on hindcast (to prevent leakage to forecast), also per month to account for seasonality, using all members as extra samples.

    def lookup_month_and_compute_probability(array, edges, nclasses: int):
        """ 
        function to be mapped, returns an array with realization dimension removed
        replaced by a dimension for the classes. e.g. 2 = upper, 1 = middle, 0 = lower tercile
        Looks up the edges belonging to the month
        counts the number of members per bin, and normalizes to probability
        """
        month = int(np.unique(array.month.values))
        edges_month = edges.loc[month].values
        pre_allocated_probabilities = xr.DataArray(np.full(array.shape[:1] + (nclasses,), np.nan), dims = array.dims[:1] + ('classes',), coords = array.coords[array.dims[0]].coords)
        pre_allocated_probabilities.coords['classes'] = ('classes',pd.RangeIndex(nclasses))
        binned = np.digitize(x = array, bins = edges_month)
        for i in range(nclasses): # Counting the number of members assigned to that class
            pre_allocated_probabilities.loc[:,i] = (binned == i).mean(axis = 1) # Mean over boolean is same as counting True's and dividing by length of axis
        return pre_allocated_probabilities 

    hindcast_probabilities = hindcast_avg.groupby('month').map(lookup_month_and_compute_probability, edges = tercile_edges, nclasses = nclasses)
    forecast_probabilities = forecast_avg.groupby('month').map(lookup_month_and_compute_probability, edges = tercile_edges, nclasses = nclasses)

    if return_edges:
        return hindcast_probabilities, forecast_probabilities, tercile_edges
    else:
        return hindcast_probabilities, forecast_probabilities

if __name__  == '__main__': # Running as script, not calling from a notebook.
    outdir = Path('/scratch/')
    experiment_name = 'trial2_ensmean'
    ensmean = True
    varlist = ['tp','sst','tcw']

    # Construction of inputs
    training_inputs = {key:[] for key in varlist}  # These will be hindcasts
    testing_inputs = {key:[] for key in varlist}  # These will be forecasts (though probably more testing data will be generated through crossvalidation of hindcasts)
    for var in varlist:
        if var == 'tp': # No seasonal anomalies
            hindcast, forecast = preprocess_ecmwf(var = var, rm_season = False, ensmean = ensmean, standardize_space = False, standardize_time = True) # No seasonal removal for rainfall? Perhaps rainfall should be min-max scaled, such that zero is really zero?
        else:
            hindcast, forecast = preprocess_ecmwf(var = var, rm_season = True, ensmean = ensmean, standardize_space = False, standardize_time = True)
        training_inputs[var] = hindcast.expand_dims({'variable':[var]})
        testing_inputs[var] = forecast.expand_dims({'variable':[var]})
    
    training_inputs = xr.concat(training_inputs.values(), dim = 'variable') # This already stacks the arrays into nchannels = nvariables 
    testing_inputs = xr.concat(testing_inputs.values(), dim = 'variable')
    
    # Extra stacking of channels, as multiple members available per variable
    if not ensmean:
        nmembers = len(training_inputs.coords['realization'])
        training_inputs = training_inputs.stack({'channels':['realization','variable']})
        testing_inputs = testing_inputs.sel(realization = np.random.choice(testing_inputs.realization.values, size = nmembers, replace = False)) # Testing now needs to be matched in training, so downsampling the members to 11 (always select control?)
        testing_inputs = testing_inputs.stack({'channels':['realization','variable']})
    else:
        training_inputs = training_inputs.rename({'variable':'channels'})
        testing_inputs = testing_inputs.rename({'variable':'channels'})
    
    # processing target, checking correspondence of time axes
    target_h, target_f = preprocess_target(quantile_edges = [0.33,0.66], return_edges = False)
    assert np.all(np.equal(target_h.valid_time.values, training_inputs.valid_time.values)), 'training timestamps must match'
    assert np.all(np.equal(target_f.valid_time.values, testing_inputs.valid_time.values)), 'testing timestamps must match'
    
    hindcast_benchmark, forecast_benchmark = preprocess_raw_forecasts(quantile_edges = [0.33,0.66], return_edges = False)
    
    # Re-ordering and writing inputs to disk (only array, no coordinates, so directly readable with numpy (and tensorflow)
    # (nsamples,nlat,nlon,nchannels)
    np.save(file = outdir / f'{experiment_name}.training_inputs.npy', arr = training_inputs.transpose('valid_time','latitude','longitude','channels').values)
    np.save(file = outdir / f'{experiment_name}.testing_inputs.npy', arr = testing_inputs.transpose('valid_time','latitude','longitude','channels').values)
    
    # writing targets to disk (nsamples, nclasses)
    np.save(file = outdir / f'{experiment_name}.training_terciles.npy', arr = target_h.values)
    np.save(file = outdir / f'{experiment_name}.testing_terciles.npy', arr = target_f.values)
    
    # Writing benchmarks to disk (nsamples, nclasses), not as numpy but as pandas because no feeding into neural network
    hindcast_benchmark.to_pandas().to_hdf(outdir / f'{experiment_name}.training_benchmark.h5', key = 'benchmark', mode = 'w')
    forecast_benchmark.to_pandas().to_hdf(outdir / f'{experiment_name}.testing_benchmark.h5', key = 'benchmark', mode = 'w')
    
    # Some extra time information
    target_h.valid_time.to_pandas().to_hdf(outdir / f'{experiment_name}.training_timestamps.h5', key = 'timestamps', mode = 'w')
    target_f.valid_time.to_pandas().to_hdf(outdir / f'{experiment_name}.testing_timestamps.h5', key = 'timestamps', mode = 'w')
