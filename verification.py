import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------
#   Define function to compute Brier skill score and apply to data set
#   The assumption is that we have a pd.DataFrame with one-hot encoded observations
#   (below normal, normal, above normal) and another one with predicted probabilities

def brier_score(y, x):
    return ((x-y)**2).sum(1).mean(0)

def brier_skill_score(y, x, x0):
    return 1. - brier_score(y,x) / brier_score(y,x0)

def l1yocv_climatology(y):
    prob = pd.DataFrame(0.0, index=y.index, columns=y.columns)
    years = y.index.year.unique()
    for yr in years:
        prob_yr = y.loc[y.index.year!=yr].mean(0)
        prob.loc[y.index.year==yr] = prob.loc[y.index.year==yr].add(prob_yr)
    return prob

#----------------------------------------------------------------------------------------
#  Define function to plot reliability diagrams for the three categories

def reliability_diagram(y, x, nbins=11, nmin=50):
    fig = plt.figure(figsize=(14,4.5))
    for icat in range(x.shape[-1]):
        category = x.columns[icat]
        xy_cat = pd.concat([y[category].rename('obs'),x[category].rename('prob'),(x[category]*(nbins-1)).round().rename('stratum')], axis=1)
        freq = xy_cat.groupby(['stratum']).size()
        relia = xy_cat.groupby(['stratum']).mean()
        relia[freq<nmin] = np.nan
        ax = fig.add_subplot(1,3,1+icat)
        rel = plt.plot(relia['prob'].dropna(), relia['obs'].dropna(), '-o', c='royalblue')
        plt.plot([0,1], [0,1], c='k')
        plt.axvline(0.33, c='k', ls=':', lw=1, ymin=0.05, ymax=0.58)
        plt.title(f'Reliability for "{category}"\n',fontsize=14)
        ins = ax.inset_axes([0.03,0.70,0.35,0.25])
        ins.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ins.set_xlabel('Frequency of usage', fontsize=11)
        ins.bar(np.arange(nbins), freq.reindex(np.arange(nbins)).fillna(0), 0.6, color='royalblue')
    plt.tight_layout()
    return fig, ax

if __name__ == '__main__':
    # Just some testing
    obs_binary = np.load('/scratch/trial2_ensmean.training_terciles.npy') # Spatial average 4-week rainfall classified into terciles, one hot encoded (low, mid, high)
    fcst_prob = pd.read_hdf('/scratch/trial2_ensmean.training_benchmark.h5') # Same, but as forecast by the ecmwf model (own terciles used).
    obs_binary = pd.DataFrame(obs_binary, index = fcst_prob.index, columns = fcst_prob.columns)

    clm_prob = l1yocv_climatology(obs_binary)
    brier_skill_score(obs_binary, fcst_prob, clm_prob)
