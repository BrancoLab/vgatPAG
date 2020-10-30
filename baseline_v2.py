# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from tqdm import tqdm

from scipy.stats import zscore
from fcutils.maths.filtering import smooth_hanning
from skimage.filters import threshold_otsu, threshold_li
from Analysis  import (
    mouse_sessions,
    sessions, 
    get_mouse_session_data
)

from fcutils.plotting.utils import clean_axes, save_figure

fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\baseline2')

# %%
"""
    Use image thresholding algorithms to compute a threshold which divides a ROIs activity
    between baseline and active. 

"""
for mouse, sess, sessname in tqdm(mouse_sessions):
    # get data
    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
    if is_rec is None: continue
    # Plot signals 
    f, axarr = plt.subplots(ncols=2, nrows=nrois, figsize=(20, 3*nrois))

    for n, sig in enumerate(signals):
        sig = sig[is_rec == 1]
        smooth_sig = smooth_hanning(sig, window_len=20)

        th = threshold_li(sig, initial_guess=0,) # np.min(smooth_sig), tolerance=.1)
        # th = np.percentile(smooth_sig, 25)

        x = np.arange(len(smooth_sig))
        axarr[n, 0].plot(x[smooth_sig > th], smooth_sig[smooth_sig > th], color='g')
        axarr[n, 0].plot(x[smooth_sig <= th], smooth_sig[smooth_sig <= th], color=[.6, .6, .6])
        axarr[n, 0].axhline(th,  lw=4, color='m')

        bins = np.linspace(-50, 150, 75)
        axarr[n, 1].hist(smooth_sig, color='k', bins=bins, histtype='step')
        axarr[n, 1].hist(smooth_sig[smooth_sig > th], bins=bins, color='g', alpha=.6, label='above threshold')
        axarr[n, 1].hist(smooth_sig[smooth_sig <= th], bins=bins, color=[.6, .6, .6], alpha=.6, label='below threshold')
        axarr[n, 1].axvline(th, lw=4, color='m', label='threshold')

        if n < len(sig)-2:
            axarr[n, 0].set(xticks=[])
            axarr[n, 1].set(xticks=[])
        elif n == 0:
            axarr[n, 1].legend()
        axarr[n, 0].set(ylabel=f'ROI {n}')

    clean_axes(f)
    save_figure(f, str(fld/f'{sessname}_LI_thresholds'))

    break

# %%
" for each ROI plot the histogram of signal + a normal distribution with same mean and std"

from fcutils.plotting.plot_distributions import plot_distribution
from fcutils.maths.distributions import get_distribution
from scipy.stats import entropy, normaltest
import scipy.stats
from brainrender.colors import colorMap

alpha = 1e-6
bins = np.linspace(-50, 150, 100)

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions

    see: https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

all_dists = []
for mouse, sess, sessname in tqdm(mouse_sessions):
    # get data
    tracking, ang_vel, speed, shelter_distance, signals, nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)
    if is_rec is None: continue
    # Plot signals 
    # f, axarr = plt.subplots(nrows=5, ncols=6, figsize=(20, 20), sharex=True, sharey=False)
    # axarr = axarr.flatten()

    for n, sig in enumerate(signals):
        # get smoothed signal
        sig = sig[is_rec == 1]
        smooth_sig = smooth_hanning(sig, window_len=20)

        # Get KL divergence
        norm_sig = get_distribution('normal', np.mean(smooth_sig), np.std(smooth_sig), n_samples=len(smooth_sig)*5)

        p = np.histogram(smooth_sig, bins=bins, density=True)[0]
        q = np.histogram(norm_sig, bins=bins, density=True)[0]

        dist = jensen_shannon_distance(p, q)
        all_dists.append(dist)
        color = colorMap(dist, name='bwr', vmin=0, vmax=.6)
        # axarr[n].set(title=f'ROI {n} | dist: {round(dist, 3)} | ')



        # # plot
        # axarr[n].hist(smooth_sig, bins=bins, density=True, histtype='stepfilled', ec=[.4, .4, .4],
        #                 color=color, alpha=.9, lw=2)

        # plot_distribution(np.mean(smooth_sig), np.std(smooth_sig), dist_type='normal', ax=axarr[n], 
        #         plot_kwargs=dict(lw=2, color='m', alpha=1))

        
        # break
    # break



# %%

# %%
