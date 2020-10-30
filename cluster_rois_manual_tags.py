# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from brainrender.colors import makePalette, colorMap
import numpy as np
from scipy.signal import medfilt
from pathlib import Path
from palettable.cmocean.sequential import Matter_8 as CMAP
from palettable.cmocean.sequential import Speed_8 as CMAP2
from palettable.cmocean.sequential import Deep_8 as CMAP3
from matplotlib.lines import Line2D
from tqdm import tqdm
from scipy.stats.stats import pearsonr   
from numba import jit
import scipy
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


from fcutils.plotting.utils import calc_nrows_ncols, set_figure_subplots_aspect, clean_axes, save_figure
from fcutils.plotting.plot_elements import plot_mean_and_error

from Analysis  import (
        mice,
        sessions,
        recordings,
        recording_names,
        stimuli,
        clean_stimuli,
        get_mouse_session_data,
        sessions_fps,
        mouse_sessions,
        pxperframe_to_cmpersec,
)
from Analysis.tag_aligned import (
    manual_tags,
    get_tags_by,
    get_next_tag,
    get_last_tag,
)



# %%

# ---------------------------------------------------------------------------- #
#                              CLUSTER CORR MATRIX                             #
# ---------------------------------------------------------------------------- #


SMOOTH_REG = 0.0   # Strength of roughness penalty
WARP_REG = 0.0      # Strength of penalty on warp magnitude
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
MAXLAG = .1        # Maximum amount of shift allowed.

# shift_model = ShiftWarping(
#     maxlag=MAXLAG,
#     smoothness_reg_scale=SMOOTH_REG,
#     warp_reg_scale=WARP_REG,
#     l2_reg_scale=L2_REG, 
# )


# dend = sch.dendrogram(sch.linkage(corr, method='ward'))

# cluster
N_CLUSTERS = 5
cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='euclidean', linkage='ward')
_ = cluster.fit_predict(corr)


# plot traces
f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(16, 12), sharex=True, sharey=True)
axarr = axarr.flatten()

for ax, clust_id in zip(axarr, set(cluster.labels_)):
    idxs = np.argwhere(cluster.labels_ == clust_id)
    clust_sigs = cache.iloc[idxs.ravel()]

    sigs = []
    for i,t in clust_sigs.iterrows():
        # min, max = t.signal.min(), t.signal.max()
        # sigs.append((t.signal-min)/(max-min))
        sigs.append(t.signal - np.mean(t.signal[:n_frames_pre-1]))
    signals = np.vstack(sigs)

    # affine warp all signals
    # warp_signs = signals[np.newaxis, :, :].transpose((1, 2, 0))
    # shift_model.fit(warp_signs)
    # warped = shift_model.transform(warp_signs).squeeze(2)

    ax.plot(signals.T, color='k', lw=1, alpha=.5)

    mean, std = np.mean(signals, 0), np.std(signals, 0)
    plot_mean_and_error(mean, std, ax, zorder=99, color='r')

    ax.axvline(n_frames_pre, lw=2, color='g')
    ax.set(title=f'Cluster {clust_id} - {len(clust_sigs)} trials', ylim=[-150, 150])

    


# Plot correlation matrix (sorted)
srtd = np.sort(cluster.labels_)
changes = [np.where(srtd == n)[0][0] for n in range(N_CLUSTERS)]

corr_clust = corr.copy()
corr_clust[:] = corr_clust[:, np.argsort(cluster.labels_)]
corr_clust[:] = corr_clust[np.argsort(cluster.labels_), :]
axarr[-1].axis('off')
clean_axes(f)

fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(corr_clust, cmap='RdYlGn', )
fig.colorbar(cax, ax=ax, shrink=0.9)

for change in changes:
    ax.axvline(change, lw=2, color='m')
    ax.axhline(change, lw=2, color='m')

# %%
# Plot individual trials for sanity check
f, axarr = plt.subplots(ncols=2, figsize=(12, 8))

for (idx1, idx2) in zip(*np.where((corr > .9)&(corr<1))):
    t1, t2 = cache.iloc[idx1], cache.iloc[idx2]
    if idx1 < 500: continue

    if t1.mouse != t2.mouse:

        axarr[0].plot(t1.x, t1.y)
        axarr[0].plot(t2.x, t2.y)
        axarr[0].set(title='XY tracking')

        axarr[1].plot(t1.signal)
        axarr[1].plot(t2.signal)
        axarr[1].set(title=f'Correlation: {round(corr[idx1, idx2], 2)}')

        break

# %%
traces = cache.loc[(cache.mouse == 'BF161p1')&(cache.session == '19JUN03')]

all_sigs = []
for roin in traces.roi_n.unique():
    roi_traces = traces.loc[traces.roi_n == roin]
    roi_sigs = np.vstack([t.signal for i,t in roi_traces.iterrows()])
    all_sigs.append(roi_sigs)

all_sigs = np.array(all_sigs).transpose((1, 2, 0))
all_sigs.shape # n trials - n samples - n rois
    
# %%


# Fit to binned spike times.
shift_model.fit(all_sigs, iterations=50)

shift_aligned_data = shift_model.transform(all_sigs)


f, axarr = plt.subplots(ncols=4, nrows=4, figsize=(24, 18))
axarr = axarr.flatten()

for n, ax in enumerate(axarr):
    # pot not-warped traces
    ax.plot(all_sigs[:, :, n].T, color='k', alpha=.3 ) 
    # plot_mean_and_error(np.mean(all_sigs[:, :, n], 0), np.std(all_sigs[:, :, n], 0), ax, color='k')

    # pot warped traces
    ax.plot(shift_aligned_data[:, :, n].T, color='r', alpha=.3)
    # plot_mean_and_error(np.mean(shift_aligned_data[:, :, n], 0), np.std(shift_aligned_data[:, :, n], 0), ax, color='r')
# 
    ax.axvline(n_frames_pre, lw=2, color='g')
    # break
# %%
