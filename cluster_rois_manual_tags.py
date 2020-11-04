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

fps = 30

n_sec_pre = 4 # rel escape onset
n_sec_post = 6 # # rel escape onset
n_frames_pre = n_sec_pre * fps
n_frames_post = n_sec_post * fps

# cluster
N_CLUSTERS = 4
cluster = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='euclidean', linkage='ward')

fld = Path('D:\\Dropbox (UCL - SWC)\\Project_vgatPAG\\analysis\\doric\\Fede\\plots\\ManualTagsAligned')

corr = np.load(os.path.join(fld, 'cached_corr_mtx.npy'))
cache = pd.read_hdf(os.path.join(fld, 'cached_traces.h5'), key='hdf')

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
    ax.set(title=f'Cluster {clust_id} - {len(clust_sigs)} trials', ylim=[-5, 5])

    


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
    