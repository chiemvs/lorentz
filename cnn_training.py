import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

"""
generating constant climatological probabilities
"""

def generate_climprob_inputs(patchinputs, climprobs:np.ndarray):
    """
    Belonging to a set of patches this function generates the accompanying
    constant array of climatological probabilies, with as first axis length the amount of samples
    The logarithm over those probabilities is taken for easier adding in the cnn
    """
    assert np.isclose(climprobs.sum(),1.0), 'sum of climatological probabilities over classes should be close to 1'
    probs = np.repeat(climprobs[np.newaxis,...], repeats = patchinputs.shape[0], axis = 0)
    return np.log(probs) 

"""
CNN & Crossvalidation code goes below
"""
def earlystop(patience: int = 10, monitor: str = 'val_loss'):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, min_delta=0, patience=patience,
        verbose=1, mode='auto', restore_best_weights=True)

DEFAULT_FIT = dict(batch_size = 32, epochs = 10, shuffle = True,  callbacks = [earlystop(patience = 7, monitor = 'val_loss')])

DEFAULT_COMPILE = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics = ['accuracy'],
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False))

DEFAULT_CONSTRUCT = dict(n_classes = 2, n_initial_filters = 4, n_conv_blocks = 3, n_hidden_fcn_layers = 0, n_hidden_nodes = 10, dropout_rate = 0.3)

def construct_climdev_cnn(n_classes: int, n_initial_filters: int, n_conv_blocks: int, n_hidden_fcn_layers: int, n_hidden_nodes: int, inputshape: tuple, dropout_rate: float = 0.3):
    """
    Creates a two-branch Classifier model (n_classes)
    Branch one (simple) receives the (logarithm of) climatological probability for each class 
    Branch two (complex) is a convolutional network taking in a patch of inputs.
    the amount of filters gets doubled with each block
    A fully connected ending translates the output from convolutions to learned multiplications of the climatological probabilities
    """
    assert n_conv_blocks >= 1, 'at least one convolutional block is needed to produce a cnn'
    log_p_clim = tf.keras.layers.Input((n_classes,)) # These are going to be constant, supplied with a data-generator, they are logarithms of the climatological probability
    patch_input = tf.keras.layers.Input(inputshape)
    weights_initializer = tf.keras.initializers.GlorotUniform() # initialization of weights should be optimal for the activation function
    bias_initializer = tf.keras.initializers.Zeros() # initialization of weights should be optimal for the activation function
    x = patch_input
    for i in range(n_conv_blocks):
        x = tf.keras.layers.Conv2D(n_initial_filters * (i+1), (3,3), # nfilters, kernelsize (x,y) 
                kernel_initializer= weights_initializer,
                bias_initializer= bias_initializer,
                activation = 'elu')(x) 
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    for i in range(n_hidden_fcn_layers): # Fully connected
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(units = n_hidden_nodes, activation='elu', kernel_initializer = weights_initializer, bias_initializer = bias_initializer)(x) # was units = n_features, but not logical to scale with predictors. 10 is choice in Scheuerer 2020 (with 20 outputs)
    x = tf.keras.layers.Dense(units = n_classes, activation='elu', kernel_initializer = weights_initializer, bias_initializer = bias_initializer)(x) # Last fully connected building up to output
    pre_activation = tf.keras.layers.Add()([log_p_clim, x]) # addition: x + log_p_clim. outputs the logarithm of class probablity. Multiplicative nature seen in e.g. softmax: exp(x + log_p_clim) = exp(x)*p_clim
    prob_dist = tf.keras.layers.Activation('softmax')(pre_activation) # normalized to sum to 1

    return tf.keras.models.Model(inputs = [patch_input, log_p_clim], outputs = prob_dist)


class ModelRegistry(object):
    """
    Creation of multiple models, training them on data (xdata, ydata)
    xdata is a list of potentially multiple arrays (if network has multiple branches)
    bookkeeping on the parts of the data seen
    All models are created with the same hyperparameters
    """
    def __init__(self, xdata: list, ydata: np.ndarray, timestamps: pd.DatetimeIndex, compile_kwargs = DEFAULT_COMPILE, construct_kwargs = DEFAULT_CONSTRUCT, fit_kwargs = DEFAULT_FIT):
        self.xdata = xdata # list of arrays, All data in memory (small datasets so no bottleneck)
        self.ydata = ydata # Array
        self.timestamps = timestamps
        self.construct_kwargs = construct_kwargs
        self.construct_kwargs.update({'inputshape':xdata[0].shape[1:]}) # Model needs knowledge about the patch size (first branch) to be constructed
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.registry = [] # Containing the model objects
        self.train_indices = [] # Containing the indices of samples on which a model gets trained
        self.val_indices = [] # Containing the indices of validation samples
        self.test_indices = [] # Containing the indices of test samples
        self.histories = [] 

    def __repr__(self):
        return f'Registry of {len(self.registry)} models, containing \n {len(np.concatenate(self.train_indices))} training samples, \n {len(np.concatenate(self.val_indices))} validation samples, \n {len(np.concatenate(self.test_indices))} testing samples \n (with double counting)'

    def initialize_untrained_model(self, train_indices: np.ndarray, val_indices: np.ndarray, test_indices: np.ndarray):
        """
        You already need to supply the indices of the portion of data you want to train/validate it on
        test indices are for future reference
        returns the index of the new model in the registry
        """
        model = construct_climdev_cnn(**self.construct_kwargs)
        model.compile(**self.compile_kwargs)
        self.registry.append(model)
        self.train_indices.append(train_indices)
        self.val_indices.append(val_indices)
        self.test_indices.append(test_indices)
        self.histories.append(None)
        return len(self.registry) - 1

    def _get_indices(self, modelindex: int, what: str):
        """
        hidden method, what can be ['train', 'val', 'test']
        """
        indices = getattr(self, f'{what}_indices')[modelindex]
        return indices

    def train_model(self, modelindex: int):
        """trains a model of choice"""
        model = self.registry[modelindex]
        train_idx = self._get_indices(modelindex,'train')
        val_idx = self._get_indices(modelindex,'val')
        xtrain = [data[train_idx,...] for data in self.xdata ] # List of arrays
        xval = [data[val_idx,...] for data in self.xdata ] # List of arrays
        ytrain = self.ydata[train_idx,...]
        yval = self.ydata[val_idx,...]

        history = model.fit(x = xtrain,
                y = ytrain,
                validation_data = (xval, yval),
                **self.fit_kwargs)
        self.histories[modelindex] = history.history

    def build_curves(self):
        """
        Dataframe to gather the evolutions of validation and training losses / accuracies 
        with the progressing of epochts. Uses sequences saved in the histories
        """
        dataframes = []
        for modelindex in range(len(self.registry)):
            history = self.histories[modelindex]
            if not (history is None):
                df = pd.DataFrame(history)
                df.index = pd.MultiIndex.from_product([[modelindex],df.index + 1], names = ['modelindex','epoch'])
                dataframes.append(df)
        return pd.concat(dataframes, axis = 0)

    def single_model_prediction(self, modelindex: int, what: str = 'val'):
        """Single model method"""
        model = self.registry[modelindex]
        indices = self._get_indices(modelindex, what)
        x = [data[indices,...] for data in self.xdata] # List of arrays
        return model(x)

    def make_predictions(self, what: str = 'val'):
        """ all models at once """
        predictions = [self.single_model_prediction(modelindex, what = what) for modelindex in range(len(self.registry))]
        predictions = np.concatenate(predictions)
        combined_indices = np.concatenate(getattr(self, f'{what}_indices'))
        combined_stamps = self.timestamps[combined_indices]
        return pd.DataFrame(predictions, index = combined_stamps)

class DoubleCV(object):
    def __init__(self, xdata: list, ydata: np.ndarray, timestamps: pd.DatetimeIndex, ntest: int = 1, nval: int = 3):
        """
        number of leave out test years, number of leave out validation years
        """
        self.xdata = xdata # list of arrays, All data in memory (small datasets so no bottleneck)
        self.ydata = ydata # Array
        self.timestamps = timestamps
        self.years = timestamps.year.unique().sort_values()
        self.testyears = np.split(self.years, np.arange(ntest, len(self.years),ntest)) # list of pandas int_index
        self.valyears = []  # list of lists
        for testyears in self.testyears:
            remaining_years = self.years.drop(testyears)
            self.valyears.append(np.split(remaining_years, np.arange(nval, len(remaining_years),nval)))
        self.registries = [None] * len(self.testyears)

    def _find_index_of_block(self, testyear: int):
        contains = [(testyear in int_index) for int_index in self.testyears]
        return contains.index(True)

    def train_inner_loop(self, testyear: int, compile_kwargs = DEFAULT_COMPILE, construct_kwargs = DEFAULT_CONSTRUCT, fit_kwargs = DEFAULT_FIT, overwrite: bool = False):
        """
        Properly initialize the registry, if already present overwrite (if set to True)
        """
        idx = self._find_index_of_block(testyear = testyear)
        testyear_list = self.testyears[idx] 
        test_indices = np.where(self.timestamps.year.map(lambda y: y in testyear_list))[0] # From boolean to numeric index
        if (not (self.registries[idx] is None)) and (not overwrite):
            raise AttributeError(f'registry for inner loop with test: {testyear}, already exists, and overwrite is False')
        registry = ModelRegistry(xdata = self.xdata, ydata = self.ydata, timestamps = self.timestamps, 
                compile_kwargs = compile_kwargs, construct_kwargs = construct_kwargs, fit_kwargs = fit_kwargs)  # A single registry has one set of parameters.
        for valyear_list in self.valyears[idx]:
            val_indices = np.where(self.timestamps.year.map(lambda y: y in valyear_list))[0]
            train_indices = np.setdiff1d(np.arange(len(self.timestamps)),np.concatenate([val_indices,test_indices]), assume_unique = True) 
            modelindex = registry.initialize_untrained_model(train_indices = train_indices, val_indices = val_indices, test_indices = test_indices)
        #registry.train_model(modelindex = modelindex)
        self.registries[idx] = registry

    def optimize_in_inner_loops(self):
        for test_year in self.years:
            self.train_inner_loop(testyear = test_year)


if __name__ == '__main__':
    print(tf.config.list_physical_devices("GPU"))
    scratchdir = Path('/scratch/cvanstraat')
    # Loading of preprocessed data, check preprocessing.py for the options like removing seasonal cycle and normalization in time or space
    chosen_experiment = 'trial5_ensmean_something'
    """
    Data loading, merging of the dataset. (for later crossvalidation)
    """
    train_inputs = np.load(scratchdir / f'{chosen_experiment}.training_inputs.npy') # (nsamples,nlat,nlon,nchannels), if ensmean then channels is just the number of variables, nlat & nlon depend on patch size
    train_target = np.load(scratchdir / f'{chosen_experiment}.training_terciles.npy') # Spatial average 4-week rainfall classified into terciles, one hot encoded (low, mid, high)
    train_timestamps = pd.read_hdf(scratchdir / f'{chosen_experiment}.training_timestamps.h5')
    
    test_inputs = np.load(scratchdir / f'{chosen_experiment}.testing_inputs.npy') # these are the forecasts that have been kept separate, technically we will probably not use them as test data, as it is only one year. We will do crossvalidation instead (perhaps after adding this extra year?)
    test_target = np.load(scratchdir / f'{chosen_experiment}.testing_terciles.npy')
    test_timestamps = pd.read_hdf(scratchdir / f'{chosen_experiment}.testing_timestamps.h5')
    
    full_inputs = np.concatenate([train_inputs, test_inputs], axis = 0) # Stacking along valid_time/sample dimension
    full_target = np.concatenate([train_target, test_target], axis = 0) # Stacking along valid_time/sample dimension
    full_timestamps = pd.concat([train_timestamps, test_timestamps])
    
    n_classes = full_target.shape[-1] 
    full_clim_logprobs = generate_climprob_inputs(full_inputs, climprobs = full_target.mean(axis = 0)) # 3 classes are assumed to be equiprobable terciles
    
    registry = ModelRegistry(xdata = [full_inputs, full_clim_logprobs], 
            ydata = full_target, 
            timestamps = full_timestamps.index,
            compile_kwargs = DEFAULT_COMPILE, construct_kwargs = DEFAULT_CONSTRUCT, fit_kwargs = DEFAULT_FIT)

    self = DoubleCV(xdata = [full_inputs, full_clim_logprobs], 
            ydata = full_target, 
            timestamps = full_timestamps.index, ntest = 2, nval = 5)
    
    # Leave one year out cross-validation (for testing)
    # This leaves 21 years remaining, although there are not really 22 full years (2021 has only three samples)
    # nested within this, a leave-three-years-out crossvalidation (meaning 7 folds)
    #years = full_timestamps.index.year.unique()
    #for test_year in years:
        #remaining_years = years.drop(test_year).sort_values() # sort to make sure that the next leaving out is blockwise
        #for validation_years in np.split(remaining_years, np.arange(3, len(remaining_years),3)):
            #training_years = years.drop(test_year).drop(validation_years)
            #print('test', test_year)
            #print('validation', validation_years)
            #print('training', training_years)
            #test_indices = np.where(full_timestamps.index.year == test_year)[0] # From boolean to numeric index
            #val_indices = np.where(full_timestamps.index.year.map(lambda y: y in validation_years))[0]
            #train_indices = np.where(full_timestamps.index.year.map(lambda y: y in training_years))[0]
            #modelindex = registry.initialize_untrained_model(train_indices = train_indices, val_indices = val_indices, test_indices = test_indices)
            #print(modelindex)
            # Normally you would call train here. But there are a lot of models (6 min of training with 10 epochs)
            #registry.train_model(modelindex = modelindex)
    
    # Two examples of training
    #registry.train_model(modelindex = 0)
    #registry.train_model(modelindex = 5)
    # And an example of the training curves, which are scores computed per set of predictions
    #curves = registry.build_curves()
    
    # But of course we are doing crossvalidation, which means that we first want to make a complete (joint) set of predictions, and compute scores once
    # Verification is possible for a certain set all in one go
    #valpreds = registry.make_predictions('val') # This contains predictions by untrained models
    #verifying_obs = pd.DataFrame(full_target, index = full_timestamps.index, columns = pd.RangeIndex(n_classes, name = 'classes')) # This is the fullset, not yet one-hot-encoded
    #verifying_obs = verifying_obs.reindex(valpreds.index) # Samples are in multiple validation sets so reindex to make obs align with preds
