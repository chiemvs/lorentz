import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

# Checking available experiments (each the name-giver of a set of pre-processed arrays written on the /scratch/)
scratchdir = Path('/scratch')
available_experiments = np.unique([p.parts[-1].split('.')[0] for p in scratchdir.glob('*.npy')])
print('available experiments: ', available_experiments)

# Loading of preprocessed data, check preprocessing.py for the options like removing seasonal cycle and normalization in time or space
chosen_experiment = 'trial2_ensmean'
print('chosen experiment: ', chosen_experiment)

train_inputs = np.load(scratchdir / f'{chosen_experiment}.training_inputs.npy') # (nsamples,nchannels,nlat,nlon), if ensmean then channels is just the number of variables, nlat & nlon depend on patch size
train_target = np.load(scratchdir / f'{chosen_experiment}.training_terciles.npy') # Spatial average 4-week rainfall classified into terciles (2,1,0 = high,mid,low)

test_inputs = np.load(scratchdir / f'{chosen_experiment}.testing_inputs.npy') # these are the forecasts that have been kept separate, technically we will probably not use them as test data, as it is only one year. We will do crossvalidation instead (perhaps after adding this extra year?)
test_target = np.load(scratchdir / f'{chosen_experiment}.testing_terciles.npy')

# CNN & Crossvalidation code goes below
