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

from skimage.filters.thresholding import _cross_entropy

ce = []
ths = np.linspace(-200, 200, 200)
for ith in ths:
    ce.append(_cross_entropy(sig, ith))

plt.plot(ths, ce)
plt.axvline(th)
# %%
