import os
import pandas as pd

def log_experiment(experiment_name: str, 
                   params: tuple,# = [[experiment_name,varlist, patchsize, rm_season, ensmean, standardize_space, standardize_time]], 
                   param_names: list,# = ['experiment_name','varlist', 'patchsize', 'rm_season', 'ensmean', 'standardize_space','standardize_time']
                  ):
    """ 
    save information on preprocessing and box size and variables used
    
    params: list of all parameter to log
    param_names: list of strings containing the names of the parameters
    
    RETURNS: logs  are saved to {experiment_name}.csv, if file already exists, new entries are appended.
    
    TODO: maybe need sth like model_name
    """
    
    setup = pd.DataFrame(params,
                         columns=param_names)
    #print(setup)

    if os.path.isfile(f'{experiment_name}.csv'):
        setup.to_csv(f'{experiment_name}.csv', sep=',', index=False, mode = 'a', header=None)
    else:
        setup.to_csv(f'{experiment_name}.csv', sep=',', index=False, mode = 'a')
        
def log_training(registry, experiment_name):
    """ save the training scores to csv """
    import os
    curves = registry.build_curves()
    curves = curves.reset_index()
    if os.path.isfile(f'{experiment_name}_curves.csv'):
        curves.to_csv(f'{experiment_name}_curves.csv', sep=',', index=False, mode = 'a', header=None)
    else:
        curves.to_csv(f'{experiment_name}_curves.csv', sep=',', index=False, mode = 'a')