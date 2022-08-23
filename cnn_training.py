import tensorflow as tf
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

"""
Data loading, generating constant climatological probabilities
"""
train_inputs = np.load(scratchdir / f'{chosen_experiment}.training_inputs.npy') # (nsamples,nlat,nlon,nchannels), if ensmean then channels is just the number of variables, nlat & nlon depend on patch size
train_target = np.load(scratchdir / f'{chosen_experiment}.training_terciles.npy') # Spatial average 4-week rainfall classified into terciles (2,1,0 = high,mid,low)

test_inputs = np.load(scratchdir / f'{chosen_experiment}.testing_inputs.npy') # these are the forecasts that have been kept separate, technically we will probably not use them as test data, as it is only one year. We will do crossvalidation instead (perhaps after adding this extra year?)
test_target = np.load(scratchdir / f'{chosen_experiment}.testing_terciles.npy')

def generate_climprob_inputs(patchinputs, climprobs:np.ndarray):
    """
    Belonging to a set of patches this function generates the accompanying
    constant array of climatological probabilies, with as first axis length the amount of samples
    The logarithm over those probabilities is taken for easier adding in the cnn
    """
    assert np.isclose(climprobs.sum(),1.0), 'sum of climatological probabilities over classes should be close to 1'
    probs = np.repeat(climprobs[np.newaxis,...], repeats = patchinputs.shape[0], axis = 0)
    return np.log(probs) 

n_classes = len(np.unique(train_target))
clim_logprobs_train = generate_climprob_inputs(train_inputs, climprobs = np.repeat(1/n_classes, n_classes)) # 3 classes are assumed to be equiprobable terciles
clim_logprobs_test = generate_climprob_inputs(test_inputs, climprobs = np.repeat(1/n_classes, n_classes))

"""
CNN & Crossvalidation code goes below
"""
def earlystop(patience: int = 10, monitor: str = 'val_loss'):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, min_delta=0, patience=patience,
        verbose=1, mode='auto', restore_best_weights=True)


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
        x = tf.keras.layers.Dense(units = n_hidden_nodes, activation='elu', kernel_initializer = weights_initializer)(x) # was units = n_features, but not logical to scale with predictors. 10 is choice in Scheuerer 2020 (with 20 outputs)
    x = tf.keras.layers.Dense(units = n_classes, activation='elu', kernel_initializer = weights_initializer)(x) # Last fully connected building up to output
    pre_activation = tf.keras.layers.Add()([log_p_clim, x]) # addition: x + log_p_clim. outputs the logarithm of class probablity. Multiplicative nature seen in e.g. softmax: exp(x + log_p_clim) = exp(x)*p_clim
    prob_dist = tf.keras.layers.Activation('softmax')(pre_activation) # normalized to sum to 1

    return tf.keras.models.Model(inputs = [patch_input, log_p_clim], outputs = prob_dist)


DEFAULT_FIT = dict(batch_size = 32, epochs = 100, shuffle = True,  callbacks = [earlystop(patience = 7, monitor = 'val_loss')])

DEFAULT_COMPILE = dict(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics = ['accuracy'],
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False))

cnn = construct_climdev_cnn(n_classes = n_classes, n_initial_filters = 4, n_conv_blocks = 3, n_hidden_fcn_layers = 0, n_hidden_nodes = 10, inputshape = train_inputs.shape[1:], dropout_rate = 0.3)

cnn.compile(**DEFAULT_COMPILE)

history = cnn.fit(x = [train_inputs, clim_logprobs_train],
                y = train_target,
                validation_split=0.3,
                **DEFAULT_FIT)
